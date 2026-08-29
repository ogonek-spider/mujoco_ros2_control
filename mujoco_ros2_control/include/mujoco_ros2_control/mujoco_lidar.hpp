// Copyright (c) 2026 the SmallDog project
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
//
// A ray-cast lidar, published as sensor_msgs/PointCloud2.
//
// MuJoCo has no lidar sensor: <rangefinder> is a single fixed ray per site, which cannot
// express a scanning sensor at all.  So this casts the rays itself with mj_multiRay, on
// the same 1/frame_hz cadence a real one would deliver frames on.
//
// It configures itself entirely out of the compiled model, and adds nothing to a model
// that does not ask for it: a <site name="lidar"> plus a <custom><numeric name="lidar_*">
// block turn it on, and their absence leaves it silent.  That is on purpose - the model
// is generated from CAD (see smalldog_description/scripts/generate_model.py and
// 3d/lidar.py in the SmallDog repo), so the sensor's pose, cone, point rate, range and
// noise all come from the same place the robot's geometry does, and there is no launch
// parameter to drift out of step with it.
//
// The scan pattern is a Risley pair - two counter-rotating wedge prisms, each deflecting
// the beam by half the cone angle - which is what makes the pattern non-repetitive, so a
// stationary sensor keeps filling in its field of view.  3d/lidar.py has the same pattern
// in Python and the argument for it; the two are separate implementations of one model,
// and they have to be changed together.
#pragma once

#include <random>
#include <string>
#include <vector>

#include "mujoco/mujoco.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace mujoco_ros2_control
{

class MujocoLidar
{
public:
  explicit MujocoLidar(rclcpp::Node::SharedPtr &node);

  // Look for the sensor in the model.  Quiet no-op if it is not there.
  void init(const mjModel *mujoco_model);

  // Call as often as convenient; it emits a cloud every 1/frame_hz of *simulated* time.
  void update(const mjModel *mujoco_model, mjData *mujoco_data);

private:
  bool read_parameters(const mjModel *mujoco_model);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

  bool enabled_{false};
  int site_{-1};
  int body_{0};                  // the sensor's own body: it does not see itself
  std::string frame_id_;

  // all of these come from the model's <custom> numerics
  double cone_{0.0};             // full cone angle from the axis, rad
  double rate_{0.0};             // points per second
  double range_min_{0.0}, range_max_{0.0};
  double sigma_{0.0};            // range noise, 1 sigma, m
  double spin_[2]{0.0, 0.0};     // prism rates, Hz
  double frame_hz_{10.0};
  mjtByte groups_[mjNGROUP]{};   // which geom groups a ray may hit

  double last_frame_{0.0};
  bool started_{false};

  std::vector<mjtNum> local_, vec_, dist_;
  std::vector<int> geomid_;
  std::mt19937 rng_{0};
  std::normal_distribution<double> noise_;
};

}  // namespace mujoco_ros2_control
