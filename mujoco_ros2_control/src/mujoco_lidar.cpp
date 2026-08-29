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

#include "mujoco_ros2_control/mujoco_lidar.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mujoco_ros2_control
{

namespace
{
constexpr const char *kSiteName = "lidar";

// One <numeric> out of the compiled model, by name.  Returns how many values it holds.
int read_numeric(const mjModel *m, const std::string &name, double *out, int max_values)
{
  const int id = mj_name2id(m, mjOBJ_NUMERIC, name.c_str());
  if (id < 0)
  {
    return 0;
  }
  const int n = std::min(m->numeric_size[id], max_values);
  for (int i = 0; i < n; ++i)
  {
    out[i] = m->numeric_data[m->numeric_adr[id] + i];
  }
  return n;
}
}  // namespace

MujocoLidar::MujocoLidar(rclcpp::Node::SharedPtr &node) : node_(node) {}

void MujocoLidar::init(const mjModel *mujoco_model)
{
  site_ = mj_name2id(mujoco_model, mjOBJ_SITE, kSiteName);
  if (site_ < 0)
  {
    RCLCPP_INFO(
      node_->get_logger(), "No site named \"%s\" in the model - no lidar is published.",
      kSiteName);
    return;
  }
  if (!read_parameters(mujoco_model))
  {
    // The site alone is not enough: without the numerics we would have to invent a cone
    // and a point rate, and an invented sensor is worse than none.
    RCLCPP_WARN(
      node_->get_logger(),
      "Site \"%s\" is in the model but its <custom><numeric name=\"%s_*\"> block is not."
      " No lidar is published - regenerate the model (smalldog_description/scripts/"
      "generate_model.py).",
      kSiteName, kSiteName);
    return;
  }

  body_ = mujoco_model->site_bodyid[site_];
  frame_id_ = std::string(kSiteName) + "_link";
  publisher_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
    std::string("~/") + kSiteName + "/points", rclcpp::SensorDataQoS());
  enabled_ = true;

  RCLCPP_INFO(
    node_->get_logger(),
    "Lidar on site \"%s\" (body \"%s\"): %.0f deg cone, %.0f points/s -> %.0f per frame at"
    " %.0f Hz, %.2f..%.1f m, %.0f mm noise. Publishing %s in frame %s.",
    kSiteName, mj_id2name(mujoco_model, mjOBJ_BODY, body_), cone_ * 180.0 / M_PI, rate_,
    rate_ / frame_hz_, frame_hz_, range_min_, range_max_, sigma_ * 1000.0,
    publisher_->get_topic_name(), frame_id_.c_str());
}

bool MujocoLidar::read_parameters(const mjModel *mujoco_model)
{
  const std::string p = std::string(kSiteName) + "_";
  double range[2] = {0.0, 0.0};
  double groups[mjNGROUP] = {};
  if (
    read_numeric(mujoco_model, p + "cone", &cone_, 1) != 1 ||
    read_numeric(mujoco_model, p + "rate", &rate_, 1) != 1 ||
    read_numeric(mujoco_model, p + "range", range, 2) != 2 ||
    read_numeric(mujoco_model, p + "sigma", &sigma_, 1) != 1 ||
    read_numeric(mujoco_model, p + "spin", spin_, 2) != 2 ||
    read_numeric(mujoco_model, p + "frame_hz", &frame_hz_, 1) != 1)
  {
    return false;
  }
  range_min_ = range[0];
  range_max_ = range[1];
  if (read_numeric(mujoco_model, p + "groups", groups, mjNGROUP) != mjNGROUP)
  {
    return false;
  }
  for (int i = 0; i < mjNGROUP; ++i)
  {
    groups_[i] = static_cast<mjtByte>(groups[i] != 0.0);
  }
  noise_ = std::normal_distribution<double>(0.0, sigma_);
  return rate_ > 0.0 && frame_hz_ > 0.0 && cone_ > 0.0 && range_max_ > range_min_;
}

void MujocoLidar::update(const mjModel *mujoco_model, mjData *mujoco_data)
{
  if (!enabled_)
  {
    return;
  }
  const double t1 = mujoco_data->time;
  if (!started_)
  {
    // The first call defines where the first window ends, not where it starts: a sim that
    // has been running for a while would otherwise open with one enormous frame.
    started_ = true;
    last_frame_ = t1;
    return;
  }
  if (t1 - last_frame_ < 1.0 / frame_hz_)
  {
    return;
  }
  const double t0 = last_frame_;
  last_frame_ = t1;

  const int nray = static_cast<int>(std::lround(rate_ * (t1 - t0)));
  if (nray <= 0)
  {
    return;
  }
  local_.resize(3 * nray);
  vec_.resize(3 * nray);
  dist_.resize(nray);
  geomid_.assign(nray, -1);

  // The Risley pair, in closed form.  Two wedges of half-angle cone/2 spinning at spin_[0]
  // and spin_[1] Hz: the angle from the sensor axis comes out as 0..cone and the azimuth
  // wraps with the first prism.  Same expression as directions() in 3d/lidar.py.
  const double a = 0.5 * cone_;
  const double sa = std::sin(a), ca = std::cos(a);
  const mjtNum *site_mat = mujoco_data->site_xmat + 9 * site_;
  for (int i = 0; i < nray; ++i)
  {
    const double t = t0 + (i + 0.5) * (t1 - t0) / nray;
    const double p1 = 2.0 * M_PI * spin_[0] * t;
    const double p2 = 2.0 * M_PI * spin_[1] * t;
    const double x1 = sa * ca * (1.0 + std::cos(p2));
    const double y1 = sa * std::sin(p2);
    const double z1 = ca * ca - sa * sa * std::cos(p2);
    const double x = x1 * std::cos(p1) - y1 * std::sin(p1);
    const double y = x1 * std::sin(p1) + y1 * std::cos(p1);
    local_[3 * i + 0] = x;
    local_[3 * i + 1] = y;
    local_[3 * i + 2] = z1;
    for (int r = 0; r < 3; ++r)      // sensor frame -> world
    {
      vec_[3 * i + r] = site_mat[3 * r] * x + site_mat[3 * r + 1] * y + site_mat[3 * r + 2] * z1;
    }
  }

  // flg_static = 1: the ground and the obstacle course are static geoms and are most of
  // what there is to see.  bodyexclude = the sensor's own body, so the chassis it is
  // bolted to does not fill a third of the cloud; the legs are NOT excluded, because a
  // real sensor does see them swing through the forward-down cone and masking them is the
  // perception side's job.
  mj_multiRay(
    mujoco_model, mujoco_data, mujoco_data->site_xpos + 3 * site_, vec_.data(), groups_, 1,
    body_, geomid_.data(), dist_.data(), nray, range_max_);

  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = frame_id_;
  // simulated time, like /clock and like the joint states - not the wall clock, which at
  // this node's real-time factor is a different clock entirely
  cloud.header.stamp = rclcpp::Time(
    static_cast<int32_t>(t1), static_cast<uint32_t>((t1 - std::floor(t1)) * 1e9),
    RCL_ROS_TIME);
  cloud.height = 1;
  cloud.is_bigendian = false;
  cloud.is_dense = true;
  cloud.point_step = 12;
  // No intensity field: mj_multiRay returns a distance and a geom id, and nothing about
  // the surface it hit.  A made-up reflectance is worse than none - see 3d/lidar.py.
  cloud.fields.resize(3);
  const char *axes[3] = {"x", "y", "z"};
  for (int i = 0; i < 3; ++i)
  {
    cloud.fields[i].name = axes[i];
    cloud.fields[i].offset = 4 * i;
    cloud.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[i].count = 1;
  }
  cloud.data.resize(static_cast<size_t>(nray) * cloud.point_step);

  uint32_t hits = 0;
  for (int i = 0; i < nray; ++i)
  {
    if (geomid_[i] < 0)
    {
      continue;
    }
    double d = dist_[i];
    if (d < range_min_ || d > range_max_)
    {
      continue;
    }
    if (sigma_ > 0.0)
    {
      d += noise_(rng_);
      if (d <= 0.0)
      {
        continue;
      }
    }
    // in the sensor's own frame, which is what frame_id_ names
    const float p[3] = {
      static_cast<float>(local_[3 * i + 0] * d), static_cast<float>(local_[3 * i + 1] * d),
      static_cast<float>(local_[3 * i + 2] * d)};
    std::memcpy(&cloud.data[hits * cloud.point_step], p, sizeof(p));
    ++hits;
  }
  cloud.width = hits;
  cloud.row_step = hits * cloud.point_step;
  cloud.data.resize(cloud.row_step);
  publisher_->publish(cloud);
}

}  // namespace mujoco_ros2_control
