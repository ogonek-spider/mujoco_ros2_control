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

#include "hardware_interface/component_parser.hpp"
#include "hardware_interface/resource_manager.hpp"
#include "hardware_interface/system_interface.hpp"

#include "mujoco_ros2_control/mujoco_ros2_control.hpp"

namespace mujoco_ros2_control
{

class MJResourceManager : public hardware_interface::ResourceManager
{
  public:
  MJResourceManager(hardware_interface::ResourceManagerParams & params, mjModel *mujoco_model, mjData *mujoco_data)
  : mj_model_(mujoco_model), 
    mj_data_(mujoco_data),
    logger_(params.logger),
    hardware_interface::ResourceManager(params, false) {
  }

bool load_and_initialize_components(
    const hardware_interface::ResourceManagerParams & params) override
  {
    components_are_loaded_and_initialized_ = true;
    
    try
    {
      robot_hw_sim_loader_.reset(new pluginlib::ClassLoader<MujocoSystemInterface>(
        "mujoco_ros2_control", "mujoco_ros2_control::MujocoSystemInterface"));
    }
    catch (pluginlib::LibraryLoadException &ex)
    {
      RCLCPP_ERROR_STREAM(params.logger, "Failed to create hardware interface loader:  " << ex.what());
      return false;
    }

    const auto hardware_info =
      hardware_interface::parse_control_resources_from_urdf(params.robot_description);    

    for (const auto &hardware : hardware_info)
    {
      std::string robot_hw_sim_type_str_ = hardware.hardware_plugin_name;
      RCLCPP_INFO(params.logger, "Trying to load plugin %s", robot_hw_sim_type_str_.c_str());

      std::unique_ptr<MujocoSystemInterface> mujoco_system;
      try
      {
        mujoco_system = std::unique_ptr<MujocoSystemInterface>(
          robot_hw_sim_loader_->createUnmanagedInstance(robot_hw_sim_type_str_));
      }
      catch (pluginlib::PluginlibException &ex)
      {
        RCLCPP_ERROR_STREAM(params.logger, "The plugin failed to load. Error: " << ex.what());
        continue;
      }

      urdf::Model urdf_model;
      urdf_model.initString(params.robot_description);
      if (!mujoco_system->init_sim(mj_model_, mj_data_, urdf_model, hardware))
      {
        RCLCPP_FATAL(params.logger, "Could not initialize robot simulation interface");
        return false;
      }

      import_component(std::move(mujoco_system), hardware);

      rclcpp_lifecycle::State state(
        lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE,
        hardware_interface::lifecycle_state_names::ACTIVE);
      set_component_state(hardware.name, state);
    }
    
    return components_are_loaded_and_initialized_;
  }

private:
  std::shared_ptr<pluginlib::ClassLoader<MujocoSystemInterface>> robot_hw_sim_loader_;

  mjModel *mj_model_;
  mjData *mj_data_;

  rclcpp::Logger logger_;
};


MujocoRos2Control::MujocoRos2Control(
  rclcpp::Node::SharedPtr &node, mjModel *mujoco_model, mjData *mujoco_data)
    : node_(node),
      mj_model_(mujoco_model),
      mj_data_(mujoco_data),
      logger_(rclcpp::get_logger(node_->get_name() + std::string(".mujoco_ros2_control"))),
      control_period_(rclcpp::Duration(1, 0)),
      last_update_sim_time_ros_(0, 0, RCL_ROS_TIME)
{
}

MujocoRos2Control::~MujocoRos2Control()
{
  stop_cm_thread_ = true;
  cm_executor_->remove_node(controller_manager_);
  cm_executor_->cancel();

  if (cm_thread_.joinable()) cm_thread_.join();
}

std::string MujocoRos2Control::get_robot_description()
{
  // Getting robot description from parameter first. If not set trying from topic
  std::string robot_description;

  auto node = std::make_shared<rclcpp::Node>(
    "robot_description_node",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  if (node->has_parameter("robot_description"))
  {
    robot_description = node->get_parameter("robot_description").as_string();
    return robot_description;
  }

  RCLCPP_WARN(
    logger_,
    "Failed to get robot_description from parameter. Will listen on the ~/robot_description "
    "topic...");

  auto robot_description_sub = node->create_subscription<std_msgs::msg::String>(
    "robot_description", rclcpp::QoS(1).transient_local(),
    [&](const std_msgs::msg::String::SharedPtr msg)
    {
      if (!msg->data.empty() && robot_description.empty()) robot_description = msg->data;
    });

  while (robot_description.empty() && rclcpp::ok())
  {
    rclcpp::spin_some(node);
    RCLCPP_INFO(node->get_logger(), "Waiting for robot description message");
    rclcpp::sleep_for(std::chrono::milliseconds(500));
  }

  return robot_description;
}

void MujocoRos2Control::init()
{  
  clock_publisher_ = node_->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
  cm_executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();

  std::string urdf_string = this->get_robot_description();

  RCLCPP_INFO(node_->get_logger(), "1. setup actuators and mechanism control node");
  // setup actuators and mechanism control node.
  std::vector<hardware_interface::HardwareInfo> control_hardware_info;
  try
  {
    control_hardware_info = hardware_interface::parse_control_resources_from_urdf(urdf_string);
  }
  catch (const std::runtime_error &ex)
  {
    RCLCPP_ERROR_STREAM(logger_, "Error parsing URDF : " << ex.what());
    return;
  }
  RCLCPP_INFO(node_->get_logger(), "2. Creating resource manager");
  hardware_interface::ResourceManagerParams params;
  params.robot_description = urdf_string;
  params.clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  params.logger = rclcpp::get_logger(node_->get_name() + std::string(".resourcemanager"));
  params.executor = cm_executor_;
  // The joint limiters live inside the resource manager, and they size their per-cycle
  // allowance from *this* rate (ResourceStorage::cm_update_rate_), not from the control
  // loop below.  It defaults to 100 Hz, so leaving it unset enforces every limit -
  // position, velocity, effort - scaled by the ratio of the two rates, silently.  The
  // controller_manager cannot supply it: it does not exist yet, since it takes this
  // resource manager by value.  So it is read off our own node, and the check after the
  // controller_manager is constructed is what stops the two from drifting apart.
  params.update_rate = static_cast<unsigned int>(
    node_->has_parameter("update_rate") ? node_->get_parameter("update_rate").as_int() : 100);

  std::unique_ptr<MJResourceManager> resource_manager = std::make_unique<MJResourceManager>(params, mj_model_, mj_data_);

  // Create the controller manager
  RCLCPP_INFO(logger_, "3. Loading controller_manager");
  controller_manager_ = std::make_shared<controller_manager::ControllerManager>(
    std::move(resource_manager), cm_executor_, "controller_manager", node_->get_namespace());
  cm_executor_->add_node(controller_manager_);

  if (!controller_manager_->has_parameter("update_rate"))
  {
    RCLCPP_ERROR_STREAM(logger_, "controller manager doesn't have an update_rate parameter");
    return;
  }

  auto update_rate = controller_manager_->get_parameter("update_rate").as_int();
  if (static_cast<unsigned int>(update_rate) != params.update_rate)
  {
    RCLCPP_ERROR(
      logger_,
      "update_rate mismatch: the controller_manager loop runs at %ld Hz, but the joint "
      "limiters were built for %u Hz, so every command limit is enforced at %.2fx its real "
      "value.  Set 'update_rate' on the %s node to %ld as well.",
      update_rate, params.update_rate,
      static_cast<double>(update_rate) / static_cast<double>(params.update_rate),
      node_->get_name(), update_rate);
  }
  control_period_ = rclcpp::Duration(std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(1.0 / static_cast<double>(update_rate))));

  // Force setting of use_sime_time parameter
  controller_manager_->set_parameter(
    rclcpp::Parameter("use_sim_time", rclcpp::ParameterValue(true)));

  stop_cm_thread_ = false;
  auto spin = [this]()
  {
    while (rclcpp::ok() && !stop_cm_thread_)
    {
      cm_executor_->spin_once();
    }
  };
  cm_thread_ = std::thread(spin);
}

void MujocoRos2Control::update()
{
  // Get the simulation time and period
  auto sim_time = mj_data_->time;
  int sim_time_sec = static_cast<int>(sim_time);
  int sim_time_nanosec = static_cast<int>((sim_time - sim_time_sec) * 1000000000);

  rclcpp::Time sim_time_ros(sim_time_sec, sim_time_nanosec, RCL_ROS_TIME);
  rclcpp::Duration sim_period = sim_time_ros - last_update_sim_time_ros_;

  publish_sim_time(sim_time_ros);

  mj_step1(mj_model_, mj_data_);

  if (sim_period >= control_period_)
  {
    controller_manager_->read(sim_time_ros, sim_period);
    controller_manager_->update(sim_time_ros, sim_period);
    last_update_sim_time_ros_ = sim_time_ros;
  }

  // use same time as for read and update call - this is how it is done in ros2_control_node
  controller_manager_->write(sim_time_ros, sim_period);

  mj_step2(mj_model_, mj_data_);
}

void MujocoRos2Control::publish_sim_time(rclcpp::Time sim_time)
{
  // TODO(sangteak601)
  rosgraph_msgs::msg::Clock sim_time_msg;
  sim_time_msg.clock = sim_time;
  clock_publisher_->publish(sim_time_msg);
}

}  // namespace mujoco_ros2_control
