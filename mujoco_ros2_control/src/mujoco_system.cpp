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

#include "mujoco_ros2_control/mujoco_system.hpp"

namespace mujoco_ros2_control
{
MujocoSystem::MujocoSystem() : logger_(rclcpp::get_logger("")) {}

std::vector<hardware_interface::StateInterface> MujocoSystem::export_state_interfaces()
{
  return std::move(state_interfaces_);
}

std::vector<hardware_interface::CommandInterface> MujocoSystem::export_command_interfaces()
{
  return std::move(command_interfaces_);
}

hardware_interface::return_type MujocoSystem::read(
  const rclcpp::Time & /* time */, const rclcpp::Duration & /* period */)
{
  // Joint states
  for (auto &joint_state : joint_states_)
  {
    joint_state.position = mj_data_->qpos[joint_state.mj_pos_adr];
    joint_state.velocity = mj_data_->qvel[joint_state.mj_vel_adr];
    // qfrc_applied is an INPUT: its only writers were in the dead block write() used to
    // carry, so this read was always 0 and /joint_states effort was always 0 with it.
    // qfrc_actuator is what the position actuators actually put into the joint.
    joint_state.effort = mj_data_->qfrc_actuator[joint_state.mj_vel_adr];
  }

  // IMU Sensor data
  for (auto &data : imu_sensor_data_)
  {
    data.orientation.data.w() = mj_data_->sensordata[data.orientation.mj_sensor_index];
    data.orientation.data.x() = mj_data_->sensordata[data.orientation.mj_sensor_index + 1];
    data.orientation.data.y() = mj_data_->sensordata[data.orientation.mj_sensor_index + 2];
    data.orientation.data.z() = mj_data_->sensordata[data.orientation.mj_sensor_index + 3];

    data.angular_velocity.data.x() = mj_data_->sensordata[data.angular_velocity.mj_sensor_index];
    data.angular_velocity.data.y() =
      mj_data_->sensordata[data.angular_velocity.mj_sensor_index + 1];
    data.angular_velocity.data.z() =
      mj_data_->sensordata[data.angular_velocity.mj_sensor_index + 2];

    data.linear_acceleration.data.x() =
      mj_data_->sensordata[data.linear_acceleration.mj_sensor_index];
    data.linear_acceleration.data.y() =
      mj_data_->sensordata[data.linear_acceleration.mj_sensor_index + 1];
    data.linear_acceleration.data.z() =
      mj_data_->sensordata[data.linear_acceleration.mj_sensor_index + 2];
  }

  // FT Sensor data
  for (auto &data : ft_sensor_data_)
  {
    data.force.data.x() = -mj_data_->sensordata[data.force.mj_sensor_index];
    data.force.data.y() = -mj_data_->sensordata[data.force.mj_sensor_index + 1];
    data.force.data.z() = -mj_data_->sensordata[data.force.mj_sensor_index + 2];

    data.torque.data.x() = -mj_data_->sensordata[data.torque.mj_sensor_index];
    data.torque.data.y() = -mj_data_->sensordata[data.torque.mj_sensor_index + 1];
    data.torque.data.z() = -mj_data_->sensordata[data.torque.mj_sensor_index + 2];
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type MujocoSystem::write(
  const rclcpp::Time & /* time */, const rclcpp::Duration & /* period */)
{
  // update mimic joint
  for (auto &joint_state : joint_states_)
  {
    if (joint_state.is_mimic)
    {
      joint_state.position_command =
        joint_state.mimic_multiplier *
        joint_states_.at(joint_state.mimicked_joint_index).position_command;
      joint_state.velocity_command =
        joint_state.mimic_multiplier *
        joint_states_.at(joint_state.mimicked_joint_index).velocity_command;
      joint_state.effort_command =
        joint_state.mimic_multiplier *
        joint_states_.at(joint_state.mimicked_joint_index).effort_command;
    }
  }
  // Joint states.
  //
  // ONE path, and it is the only one that ever ran: write the position command to the
  // MuJoCo position actuator and let the actuator's own gains close the loop.  What used
  // to be here was that line followed by an unconditional `continue` and ~45 lines of
  // position/velocity/effort handling behind it - dead from the first statement, along
  // with the two control_toolbox::Pid per joint that register_joints() built from the
  // xacro's position_kp/kd/i_max and that nothing ever called.  Advertising gains the
  // code ignores is worse than not having them: the gains in smalldog-mujoco.urdf.xacro
  // read as tuning knobs and turning them did nothing at all.  Deleted 2026-09-11; the
  // actuator's kp/dampratio in the MJCF (mini_dog.py's MJ_KP / MJ_DAMPRATIO) are the real
  // ones.
  for (auto &joint_state : joint_states_)
  {
    // actuator_id is mj_name2id's -1 when the model has no actuator of that name, and
    // ctrl[-1] is a write one double before the array.  register_joints() drops such a
    // joint now, so this should not fire; it is cheap and it is a memory write.
    if (joint_state.actuator_id < 0)
    {
      continue;
    }
    mj_data_->ctrl[joint_state.actuator_id] = joint_state.position_command;
  }
  return hardware_interface::return_type::OK;
}

bool MujocoSystem::init_sim(
  mjModel *mujoco_model, mjData *mujoco_data, const urdf::Model &urdf_model,
  const hardware_interface::HardwareInfo &hardware_info)
{
  mj_model_ = mujoco_model;
  mj_data_ = mujoco_data;

  logger_ = rclcpp::get_logger("mujoco_system");
  register_joints(urdf_model, hardware_info);
  register_sensors(urdf_model, hardware_info);

  set_initial_pose();
  RCLCPP_INFO(logger_, "Mujoco system initialized");
  return true;
}

void MujocoSystem::register_joints(
  const urdf::Model &urdf_model, const hardware_interface::HardwareInfo &hardware_info)
{
  // reserve(), not resize(): a joint the model does not have is SKIPPED rather than left
  // in the vector value-initialised.  It used to be resized up front and `continue`d over,
  // which left an all-zero JointState behind - read() then indexed qpos[0]/qvel[0] (the
  // free joint's first component) and reported it as that joint's position forever.
  // Reserving also pins the addresses the state/command interfaces below point into.
  joint_states_.clear();
  joint_states_.reserve(hardware_info.joints.size());

  for (size_t joint_index = 0; joint_index < hardware_info.joints.size(); joint_index++)
  {
    auto joint = hardware_info.joints.at(joint_index);
    int mujoco_joint_id = mj_name2id(mj_model_, mjtObj::mjOBJ_JOINT, joint.name.c_str());

    if (mujoco_joint_id == -1)
    {
      RCLCPP_ERROR_STREAM(
        logger_, "Failed to find joint in mujoco model, joint name: " << joint.name);
      continue;
    }

    // save information in joint_states_ variable
    JointState joint_state;
    joint_state.name = joint.name;
    joint_state.mj_joint_type = mj_model_->jnt_type[mujoco_joint_id];
    joint_state.mj_pos_adr = mj_model_->jnt_qposadr[mujoco_joint_id];
    joint_state.mj_vel_adr = mj_model_->jnt_dofadr[mujoco_joint_id];
    joint_state.actuator_id = mj_name2id(mj_model_, mjtObj::mjOBJ_ACTUATOR, joint.name.c_str());

    if (joint_state.actuator_id == -1)
    {
      RCLCPP_ERROR_STREAM(
        logger_, "Failed to find actuator in mujoco model, joint name: " << joint.name
                                                                         << " - skipping it");
      continue;
    }

    joint_states_.push_back(joint_state);
    JointState &last_joint_state = joint_states_.back();

    // get joint limit from urdf
    get_joint_limits(urdf_model.getJoint(last_joint_state.name), last_joint_state.joint_limits);

    // check if mimicked
    if (joint.parameters.find("mimic") != joint.parameters.end())
    {
      const auto mimicked_joint = joint.parameters.at("mimic");
      const auto mimicked_joint_it = std::find_if(
        hardware_info.joints.begin(), hardware_info.joints.end(),
        [&mimicked_joint](const hardware_interface::ComponentInfo &info)
        { return info.name == mimicked_joint; });
      if (mimicked_joint_it == hardware_info.joints.end())
      {
        throw std::runtime_error(std::string("Mimicked joint '") + mimicked_joint + "' not found");
      }
      last_joint_state.is_mimic = true;
      // index into joint_states_, which is no longer 1:1 with hardware_info.joints now
      // that joints the model does not have are skipped: find the mimicked one by name.
      const auto kept = std::find_if(
        joint_states_.begin(), joint_states_.end(),
        [&mimicked_joint](const JointState &js) { return js.name == mimicked_joint; });
      if (kept == joint_states_.end())
      {
        throw std::runtime_error(
          std::string("Mimicked joint '") + mimicked_joint + "' is not in the mujoco model");
      }
      last_joint_state.mimicked_joint_index = std::distance(joint_states_.begin(), kept);

      auto param_it = joint.parameters.find("multiplier");
      if (param_it != joint.parameters.end())
      {
        last_joint_state.mimic_multiplier = std::stod(joint.parameters.at("multiplier"));
      }
      else
      {
        last_joint_state.mimic_multiplier = 1.0;
      }
    }

    auto get_initial_value = [this](const hardware_interface::InterfaceInfo &interface_info)
    {
      if (!interface_info.initial_value.empty())
      {
        double value = std::stod(interface_info.initial_value);
        return value;
      }
      else
      {
        return 0.0;
      }
    };

    // state interfaces
    for (const auto &state_if : joint.state_interfaces)
    {
      if (state_if.name == hardware_interface::HW_IF_POSITION)
      {
        state_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_POSITION, &last_joint_state.position);
        last_joint_state.position = get_initial_value(state_if);
      }
      else if (state_if.name == hardware_interface::HW_IF_VELOCITY)
      {
        state_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_VELOCITY, &last_joint_state.velocity);
        last_joint_state.velocity = get_initial_value(state_if);
      }
      else if (state_if.name == hardware_interface::HW_IF_EFFORT)
      {
        state_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_EFFORT, &last_joint_state.effort);
        last_joint_state.effort = get_initial_value(state_if);
      }
    }

    auto get_min_value = [this](const hardware_interface::InterfaceInfo &interface_info)
    {
      if (!interface_info.min.empty())
      {
        double value = std::stod(interface_info.min);
        return value;
      }
      else
      {
        return -1 * std::numeric_limits<double>::max();
      }
    };

    auto get_max_value = [this](const hardware_interface::InterfaceInfo &interface_info)
    {
      if (!interface_info.max.empty())
      {
        double value = std::stod(interface_info.max);
        return value;
      }
      else
      {
        return std::numeric_limits<double>::max();
      }
    };

    // command interfaces
    // overwrite joint limit with min/max value
    for (const auto &command_if : joint.command_interfaces)
    {
      if (command_if.name.find(hardware_interface::HW_IF_POSITION) != std::string::npos)
      {
        command_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_POSITION, &last_joint_state.position_command);
        last_joint_state.is_position_control_enabled = true;
        last_joint_state.position_command = last_joint_state.position;
        // TODO(sangteak601): These are not used at all. Potentially can be removed.
        last_joint_state.min_position_command = get_min_value(command_if);
        last_joint_state.max_position_command = get_max_value(command_if);
      }
      else if (command_if.name.find(hardware_interface::HW_IF_VELOCITY) != std::string::npos)
      {
        command_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_VELOCITY, &last_joint_state.velocity_command);
        last_joint_state.is_velocity_control_enabled = true;
        last_joint_state.velocity_command = last_joint_state.velocity;
        // TODO(sangteak601): These are not used at all. Potentially can be removed.
        last_joint_state.min_velocity_command = get_min_value(command_if);
        last_joint_state.max_velocity_command = get_max_value(command_if);
      }
      else if (command_if.name == hardware_interface::HW_IF_EFFORT)
      {
        command_interfaces_.emplace_back(
          joint.name, hardware_interface::HW_IF_EFFORT, &last_joint_state.effort_command);
        last_joint_state.is_effort_control_enabled = true;
        last_joint_state.effort_command = last_joint_state.effort;
        last_joint_state.min_effort_command = get_min_value(command_if);
        last_joint_state.max_effort_command = get_max_value(command_if);
      }

    }
    // No PID here any more.  write() has one path - ctrl[] = position_command - so the two
    // control_toolbox::Pid this used to build from the xacro's position_kp/kd/i_max were
    // constructed, stored and never read.  See write().
  }
}

void MujocoSystem::register_sensors(
  const urdf::Model & /* urdf_model */, const hardware_interface::HardwareInfo &hardware_info)
{
  // Assuming force/torque sensor end with "_fts" in the name,
  // and IMU sensor end with "_imu" in the name
  //
  // reserve() FIRST: the loop below push_back()s into these two vectors and then hands
  // state_interfaces_ the address of a field inside the element it just added.  Without
  // the reserve, the next push_back reallocates and every interface handed out so far
  // points into freed memory - it happens to survive with one IMU and no FT sensor, which
  // is exactly the kind of bug that waits for a second sensor to appear.
  ft_sensor_data_.reserve(hardware_info.sensors.size());
  imu_sensor_data_.reserve(hardware_info.sensors.size());

  for (size_t sensor_index = 0; sensor_index < hardware_info.sensors.size(); sensor_index++)
  {
    auto sensor = hardware_info.sensors.at(sensor_index);
    std::string sensor_name = sensor.name;
    sensor_name = sensor_name.substr(0, sensor_name.rfind('_'));

    if (sensor.name.find("_fts") != std::string::npos)
    {
      FTSensorData sensor_data;
      sensor_data.name = sensor_name;
      sensor_data.force.name = sensor_name + "_force";
      sensor_data.torque.name = sensor_name + "_torque";

      int force_sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, sensor_data.force.name.c_str());
      int torque_sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, sensor_data.torque.name.c_str());

      if (force_sensor_id == -1 || torque_sensor_id == -1)
      {
        RCLCPP_ERROR_STREAM(
          logger_,
          "Failed to find force/torque sensor in mujoco model, sensor name: " << sensor.name);
        continue;
      }

      sensor_data.force.mj_sensor_index = mj_model_->sensor_adr[force_sensor_id];
      sensor_data.torque.mj_sensor_index = mj_model_->sensor_adr[torque_sensor_id];

      ft_sensor_data_.push_back(sensor_data);
      auto &last_sensor_data = ft_sensor_data_.back();

      for (const auto &state_if : sensor.state_interfaces)
      {
        if (state_if.name == "force.x")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.force.data.x());
        }
        else if (state_if.name == "force.y")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.force.data.y());
        }
        else if (state_if.name == "force.z")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.force.data.z());
        }
        else if (state_if.name == "torque.x")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.torque.data.x());
        }
        else if (state_if.name == "torque.y")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.torque.data.y());
        }
        else if (state_if.name == "torque.z")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.torque.data.z());
        }
      }
    }

    else if (sensor.name.find("_imu") != std::string::npos)
    {
      IMUSensorData sensor_data;
      sensor_data.name = sensor_name;
      sensor_data.orientation.name = sensor_name + "_quat";
      sensor_data.angular_velocity.name = sensor_name + "_gyro";
      sensor_data.linear_acceleration.name = sensor_name + "_accel";

      int quat_id = mj_name2id(mj_model_, mjOBJ_SENSOR, sensor_data.orientation.name.c_str());
      int gyro_id = mj_name2id(mj_model_, mjOBJ_SENSOR, sensor_data.angular_velocity.name.c_str());
      int accel_id =
        mj_name2id(mj_model_, mjOBJ_SENSOR, sensor_data.linear_acceleration.name.c_str());

      if (quat_id == -1 || gyro_id == -1 || accel_id == -1)
      {
        RCLCPP_ERROR_STREAM(
          logger_, "Failed to find IMU sensor in mujoco model, sensor name: " << sensor.name);
        continue;
      }

      sensor_data.orientation.mj_sensor_index = mj_model_->sensor_adr[quat_id];
      sensor_data.angular_velocity.mj_sensor_index = mj_model_->sensor_adr[gyro_id];
      sensor_data.linear_acceleration.mj_sensor_index = mj_model_->sensor_adr[accel_id];

      imu_sensor_data_.push_back(sensor_data);
      auto &last_sensor_data = imu_sensor_data_.back();

      for (const auto &state_if : sensor.state_interfaces)
      {
        if (state_if.name == "orientation.x")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.orientation.data.x());
        }
        else if (state_if.name == "orientation.y")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.orientation.data.y());
        }
        else if (state_if.name == "orientation.z")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.orientation.data.z());
        }
        else if (state_if.name == "orientation.w")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.orientation.data.w());
        }
        else if (state_if.name == "angular_velocity.x")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.angular_velocity.data.x());
        }
        else if (state_if.name == "angular_velocity.y")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.angular_velocity.data.y());
        }
        else if (state_if.name == "angular_velocity.z")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.angular_velocity.data.z());
        }
        else if (state_if.name == "linear_acceleration.x")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.linear_acceleration.data.x());
        }
        else if (state_if.name == "linear_acceleration.y")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.linear_acceleration.data.y());
        }
        else if (state_if.name == "linear_acceleration.z")
        {
          state_interfaces_.emplace_back(
            sensor.name, state_if.name, &last_sensor_data.linear_acceleration.data.z());
        }
      }
    }
  }
}

void MujocoSystem::set_initial_pose()
{
  for (auto &joint_state : joint_states_)
  {
    mj_data_->qpos[joint_state.mj_pos_adr] = joint_state.position;
  }
}

void MujocoSystem::get_joint_limits(
  urdf::JointConstSharedPtr urdf_joint, joint_limits::JointLimits &joint_limits)
{
  if (urdf_joint->limits)
  {
    joint_limits.min_position = urdf_joint->limits->lower;
    joint_limits.max_position = urdf_joint->limits->upper;
    joint_limits.max_velocity = urdf_joint->limits->velocity;
    joint_limits.max_effort = urdf_joint->limits->effort;
  }
}

}  // namespace mujoco_ros2_control

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(mujoco_ros2_control::MujocoSystem, mujoco_ros2_control::MujocoSystemInterface)
