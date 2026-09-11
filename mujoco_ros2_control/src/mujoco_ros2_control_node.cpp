// Copyright (c) 2025 Sangtaek Lee
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <chrono>
#include <thread>

#include "mujoco/mujoco.h"
#include "rclcpp/rclcpp.hpp"

#include "mujoco_ros2_control/mujoco_cameras.hpp"
#include "mujoco_ros2_control/mujoco_lidar.hpp"
#include "mujoco_ros2_control/mujoco_rendering.hpp"
#include "mujoco_ros2_control/mujoco_ros2_control.hpp"

#include <pluginlib/class_loader.hpp>
#include "mujoco_ros2_control/mujoco_system_interface.hpp"
// MuJoCo data structures
mjModel *mujoco_model = nullptr;
mjData *mujoco_data = nullptr;

// main function
int main(int argc, const char **argv)
{
  rclcpp::init(argc, argv);

  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared(
    "mujoco_ros2_control_node",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  RCLCPP_INFO_STREAM(node->get_logger(), "Initializing mujoco_ros2_control node...");
  auto model_path = node->get_parameter("mujoco_model_path").as_string();

  // load and compile model
  char error[1000] = "Could not load binary model";
  if (
    std::strlen(model_path.c_str()) > 4 &&
    !std::strcmp(model_path.c_str() + std::strlen(model_path.c_str()) - 4, ".mjb"))
  {
    mujoco_model = mj_loadModel(model_path.c_str(), 0);
  }
  else
  {
    mujoco_model = mj_loadXML(model_path.c_str(), 0, error, 1000);
  }
  if (!mujoco_model)
  {
    mju_error("Load model error: %s", error);
  }

  RCLCPP_INFO_STREAM(node->get_logger(), "Mujoco model has been successfully loaded !");
  // make data
  mujoco_data = mj_makeData(mujoco_model);

  // initialize mujoco control
  auto mujoco_control = mujoco_ros2_control::MujocoRos2Control(node, mujoco_model, mujoco_data);

  if (!mujoco_control.init())
  {
    // init() leaves controller_manager_ null on failure and update() dereferences it, so
    // there is nothing to carry on with.  It used to fall through into the loop.
    RCLCPP_FATAL_STREAM(node->get_logger(), "Mujoco ros2 controller failed to initialize");
    mj_deleteData(mujoco_data);
    mj_deleteModel(mujoco_model);
    rclcpp::shutdown();
    return 1;
  }
  RCLCPP_INFO_STREAM(
    node->get_logger(), "Mujoco ros2 controller has been successfully initialized !");

  // initialize mujoco visualization environment for rendering and cameras
  if (!glfwInit())
  {
    mju_error("Could not initialize GLFW");
  }
  auto rendering = mujoco_ros2_control::MujocoRendering::get_instance();
  rendering->init(mujoco_model, mujoco_data, node);
  RCLCPP_INFO_STREAM(node->get_logger(), "Mujoco rendering has been successfully initialized !");

  auto cameras = std::make_unique<mujoco_ros2_control::MujocoCameras>(node);
  cameras->init(mujoco_model);

  // Ray-cast lidar, if the model carries one.  Unlike the cameras it needs no GL context
  // and no offscreen buffer - it is mj_multiRay over mjData - so it can run on the sim
  // cadence below rather than being throttled down to keep the renderer alive.
  auto lidar = std::make_unique<mujoco_ros2_control::MujocoLidar>(node);
  lidar->init(mujoco_model);

  // Pace the loop against the wall clock.  The frame below advances a fixed 1/60 s of sim
  // and never sleeps, so without this it runs at whatever rate the frame costs - profiled
  // on an M-series mac at 2.8-3.0x real time, a frame being 70 % render, 23 % camera and
  // 7-8 % physics-plus-ros2_control.  Faster than real time is not free: everything on
  // screen moves at that multiple, /clock outruns the wall clock so any wall-clock timeout
  // outside the sim fires early relative to what the robot has done, and a gait tuned by
  // eye gets tuned against a robot that does not exist.
  //
  // real_time_factor 1.0 (default) tracks the wall clock; >1 runs that multiple faster; <= 0
  // restores the old unthrottled behaviour, which is what long headless sweeps want.
  double real_time_factor = 1.0;
  if (!node->has_parameter("real_time_factor"))
  {
    node->declare_parameter("real_time_factor", 1.0);
  }
  try
  {
    real_time_factor = node->get_parameter("real_time_factor").as_double();
  }
  catch (const rclcpp::exceptions::InvalidParameterTypeException &e)
  {
    RCLCPP_WARN_STREAM(
      node->get_logger(),
      "real_time_factor must be a float (try real_time_factor:=2.0, not 2): " << e.what()
        << " - falling back to 1.0");
  }
  RCLCPP_INFO_STREAM(
    node->get_logger(), "Pacing the simulation at " << real_time_factor << "x real time"
      << (real_time_factor > 0.0 ? "" : " (unthrottled)"));

  // How far the sim may fall behind the wall clock before the pacing gives up and resyncs.
  // Without this a machine that cannot keep up accumulates debt it can never repay and then
  // sprints through the backlog the moment it gets a fast frame.
  const double max_lag_s = 0.1;
  auto wall_ref = std::chrono::steady_clock::now();
  mjtNum sim_ref = mujoco_data->time;

  // run main loop, target real-time simulation and 60 fps rendering with cameras around 6 hz
  mjtNum last_cam_update = mujoco_data->time;
  while (rclcpp::ok() && !rendering->is_close_flag_raised())
  {
    // advance interactive simulation for 1/60 sec
    //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
    //  this loop will finish on time for the next frame to be rendered at 60 fps.
    //  Otherwise add a cpu timer and exit this loop when it is time to render.
    mjtNum simstart = mujoco_data->time;
    while (mujoco_data->time - simstart < 1.0 / 60.0)
    {
      mujoco_control.update();
    }
    rendering->update();
    // self-throttles to the frame rate in the model; calling it per rendered frame just
    // means a cloud is never more than one frame late
    lidar->update(mujoco_model, mujoco_data);

    // Updating cameras at ~6 Hz
    // TODO(eholum): Break control and rendering into separate processes
    if (simstart - last_cam_update > 1.0 / 6.0)
    {
      cameras->update(mujoco_model, mujoco_data);
      last_cam_update = simstart;
    }

    if (real_time_factor > 0.0)
    {
      double sim_elapsed = (mujoco_data->time - sim_ref) / real_time_factor;
      double wall_elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_ref).count();
      double ahead = sim_elapsed - wall_elapsed;
      if (ahead > 0.0)
      {
        std::this_thread::sleep_for(std::chrono::duration<double>(ahead));
      }
      else if (-ahead > max_lag_s)
      {
        wall_ref = std::chrono::steady_clock::now();
        sim_ref = mujoco_data->time;
      }
    }
  }

  rendering->close();
  cameras->close();

  // free MuJoCo model and data
  mj_deleteData(mujoco_data);
  mj_deleteModel(mujoco_model);

  // This is the SUCCESS path: it returned 1, so every launch file that watches this node's
  // exit code saw a crash on a clean quit.  And rclcpp::shutdown() was never called, so
  // the context was torn down by the process exiting rather than by ROS.
  rclcpp::shutdown();
  return 0;
}
