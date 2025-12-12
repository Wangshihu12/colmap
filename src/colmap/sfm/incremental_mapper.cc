// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/sfm/incremental_mapper.h"

#include "colmap/estimators/generalized_pose.h"
#include "colmap/estimators/pose.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/scene/projection.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sfm/incremental_mapper_impl.h"
#include "colmap/util/misc.h"

#include <array>
#include <fstream>

namespace colmap {

bool IncrementalMapper::Options::Check() const {
  CHECK_OPTION_GT(init_min_num_inliers, 0);
  CHECK_OPTION_GT(init_max_error, 0.0);
  CHECK_OPTION_GE(init_max_forward_motion, 0.0);
  CHECK_OPTION_LE(init_max_forward_motion, 1.0);
  CHECK_OPTION_GE(init_min_tri_angle, 0.0);
  CHECK_OPTION_GE(init_max_reg_trials, 1);
  CHECK_OPTION_GT(abs_pose_max_error, 0.0);
  CHECK_OPTION_GT(abs_pose_min_num_inliers, 0);
  CHECK_OPTION_GE(abs_pose_min_inlier_ratio, 0.0);
  CHECK_OPTION_LE(abs_pose_min_inlier_ratio, 1.0);
  CHECK_OPTION_GE(local_ba_num_images, 2);
  CHECK_OPTION_GE(local_ba_min_tri_angle, 0.0);
  CHECK_OPTION_GE(min_focal_length_ratio, 0.0);
  CHECK_OPTION_GE(max_focal_length_ratio, min_focal_length_ratio);
  CHECK_OPTION_GE(max_extra_param, 0.0);
  CHECK_OPTION_GE(filter_max_reproj_error, 0.0);
  CHECK_OPTION_GE(filter_min_tri_angle, 0.0);
  CHECK_OPTION_GE(max_reg_trials, 1);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_GE(random_seed, -1);
  return true;
}

IncrementalMapper::IncrementalMapper(
    std::shared_ptr<const DatabaseCache> database_cache)
    : database_cache_(std::move(database_cache)),
      reconstruction_(nullptr),
      obs_manager_(nullptr),
      triangulator_(nullptr) {}

/**
 * [功能描述]：开始一个新的增量式三维重建过程，初始化重建所需的核心组件和状态。
 * @param reconstruction：指向 Reconstruction 对象的共享指针，表示要进行的三维重建实例。
 * @return 无返回值。
 */
void IncrementalMapper::BeginReconstruction(
    const std::shared_ptr<class Reconstruction>& reconstruction) {
  // 确保当前没有正在进行的重建，避免重复初始化
  THROW_CHECK(reconstruction_ == nullptr);

  // 保存重建对象的引用，并从数据库缓存中加载已有的重建数据
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);

  // 创建观测管理器，用于管理图像观测点与3D点之间的对应关系
  obs_manager_ = std::make_shared<class ObservationManager>(
      *reconstruction_, database_cache_->CorrespondenceGraph());

  // 创建增量三角化器，用于在重建过程中进行特征点的三角化
  triangulator_ = std::make_shared<IncrementalTriangulator>(
      database_cache_->CorrespondenceGraph(), *reconstruction_, obs_manager_);

  // 初始化注册统计信息：共享注册图像数量置零，清空每个相机的注册图像计数
  reg_stats_.num_shared_reg_images = 0;
  reg_stats_.num_reg_images_per_camera.clear();

  // 遍历重建中已注册的所有帧ID，触发帧注册事件以更新统计信息
  for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
    RegisterFrameEvent(frame_id);
  }

  // 将已注册的帧ID存储到集合中，用于后续快速查找已存在的帧
  existing_frame_ids_ =
      std::unordered_set<image_t>(reconstruction->RegFrameIds().begin(),
                                  reconstruction->RegFrameIds().end());

  // 清空过滤帧集合和注册尝试次数记录，为新的重建过程做准备
  filtered_frames_.clear();
  reg_stats_.num_reg_trials.clear();
}

void IncrementalMapper::EndReconstruction(const bool discard) {
  THROW_CHECK_NOTNULL(reconstruction_);

  if (discard) {
    for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
      DeRegisterFrameEvent(frame_id);
    }
  }

  triangulator_.reset();
  obs_manager_.reset();
  reconstruction_->TearDown();
  reconstruction_ = nullptr;
}

bool IncrementalMapper::FindInitialImagePair(const Options& options,
                                             image_t& image_id1,
                                             image_t& image_id2,
                                             Rigid3d& cam2_from_cam1) {
  return IncrementalMapperImpl::FindInitialImagePair(
      options,
      *database_cache_,
      *reconstruction_,
      reg_stats_.init_num_reg_trials,
      reg_stats_.num_registrations,
      reg_stats_.init_image_pairs,
      image_id1,
      image_id2,
      cam2_from_cam1);
}

std::vector<image_t> IncrementalMapper::FindNextImages(const Options& options) {
  return IncrementalMapperImpl::FindNextImages(
      options, *obs_manager_, filtered_frames_, reg_stats_.num_reg_trials);
}

/**
 * [功能描述]：注册初始图像对，作为增量式SfM重建的起点。
 *            将第一张图像设为世界坐标系原点，第二张图像根据两视图几何关系设置位姿。
 * @param options：增量式重建的配置选项。
 * @param image_id1：第一张图像的ID，将被设置为世界坐标系原点。
 * @param image_id2：第二张图像的ID。
 * @param cam2_from_cam1：从相机1坐标系到相机2坐标系的刚体变换（相对位姿）。
 * @return 无返回值。
 */
void IncrementalMapper::RegisterInitialImagePair(
    const Options& options,
    const image_t image_id1,
    const image_t image_id2,
    const Rigid3d& cam2_from_cam1) {
  // 确保重建对象和观测管理器已初始化
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(obs_manager_);
  // 确保当前没有已注册的帧，即这是第一对初始图像
  THROW_CHECK_EQ(reconstruction_->NumRegFrames(), 0);

  // 验证配置选项的有效性
  THROW_CHECK(options.Check());

  // 更新两张图像的初始化注册尝试次数和总注册尝试次数
  reg_stats_.init_num_reg_trials[image_id1] += 1;
  reg_stats_.init_num_reg_trials[image_id2] += 1;
  reg_stats_.num_reg_trials[image_id1] += 1;
  reg_stats_.num_reg_trials[image_id2] += 1;

  // 生成图像对ID并记录到初始图像对集合中
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  reg_stats_.init_image_pairs.insert(pair_id);

  // 获取两张图像的引用
  Image& image1 = reconstruction_->Image(image_id1);
  Image& image2 = reconstruction_->Image(image_id2);

  //////////////////////////////////////////////////////////////////////////////
  // Apply two-view geometry
  // 应用两视图几何关系
  //////////////////////////////////////////////////////////////////////////////

  // 将第一张图像的相机位姿设为单位变换（世界坐标系原点）
  image1.FramePtr()->SetCamFromWorld(image1.CameraId(), Rigid3d());
  // 将第二张图像的相机位姿设为相对于第一张图像的变换
  image2.FramePtr()->SetCamFromWorld(image2.CameraId(), cam2_from_cam1);

  //////////////////////////////////////////////////////////////////////////////
  // Update Reconstruction
  // 更新重建数据结构
  //////////////////////////////////////////////////////////////////////////////

  // 在重建中注册两个帧
  reconstruction_->RegisterFrame(image1.FrameId());
  reconstruction_->RegisterFrame(image2.FrameId());

  // 触发帧注册事件，更新注册统计信息
  RegisterFrameEvent(image1.FrameId());
  RegisterFrameEvent(image2.FrameId());
}

bool IncrementalMapper::RegisterNextImage(const Options& options,
                                          const image_t image_id) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(obs_manager_);
  THROW_CHECK_GT(reconstruction_->NumRegFrames(), 0);
  THROW_CHECK(options.Check());

  Image& image = reconstruction_->Image(image_id);
  Camera& camera = *image.CameraPtr();

  for (const auto& [_, sensor_from_rig] :
       image.FramePtr()->RigPtr()->Sensors()) {
    THROW_CHECK(sensor_from_rig.has_value())
        << "Registration only implemented for frames with known "
           "sensor_from_rig poses";
  }

  // Use central camera pose estimation for trivial frames and when we don't
  // have a good estimate of the camera's focal length, because we don't have a
  // focal length estimator for non-central/generalized cameras.
  if (image.FramePtr()->RigPtr()->NumSensors() > 1) {
    bool all_cameras_have_good_focal_length = true;
    for (const data_t& data_id : image.FramePtr()->ImageIds()) {
      const Image& frame_image = reconstruction_->Image(data_id.id);
      if ((!frame_image.CameraPtr()->has_prior_focal_length &&
           reg_stats_.num_reg_images_per_camera[frame_image.CameraId()] == 0) ||
          frame_image.CameraPtr()->HasBogusParams(
              options.min_focal_length_ratio,
              options.max_focal_length_ratio,
              options.max_extra_param)) {
        all_cameras_have_good_focal_length = false;
        break;
      }
    }
    if (all_cameras_have_good_focal_length) {
      VLOG(2) << "Registering image using generalized pose estimation";
      return RegisterNextGeneralFrame(options, *image.FramePtr());
    }
  }

  reg_stats_.num_reg_trials[image_id] += 1;

  // Check if enough 2D-3D correspondences.
  if (obs_manager_->NumVisiblePoints3D(image_id) <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    VLOG(2) << "Image observes insufficient number of points for registration ("
            << obs_manager_->NumVisiblePoints3D(image_id) << " < "
            << options.abs_pose_min_num_inliers << ")";
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Search for 2D-3D correspondences
  //////////////////////////////////////////////////////////////////////////////

  std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
  std::vector<Eigen::Vector2d> tri_points2D;
  std::vector<Eigen::Vector3d> tri_points3D;

  const std::shared_ptr<const CorrespondenceGraph> correspondence_graph =
      database_cache_->CorrespondenceGraph();

  std::unordered_set<point3D_t> corr_point3D_ids;
  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);

    corr_point3D_ids.clear();
    const auto corr_range =
        correspondence_graph->FindCorrespondences(image_id, point2D_idx);
    for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
      const Image& corr_image = reconstruction_->Image(corr->image_id);
      if (!corr_image.HasPose()) {
        continue;
      }

      const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
      if (!corr_point2D.HasPoint3D()) {
        continue;
      }

      // Avoid duplicate correspondences.
      if (corr_point3D_ids.count(corr_point2D.point3D_id) > 0) {
        continue;
      }

      const Camera& corr_camera = *corr_image.CameraPtr();

      // Avoid correspondences to images with bogus camera parameters.
      if (corr_camera.HasBogusParams(options.min_focal_length_ratio,
                                     options.max_focal_length_ratio,
                                     options.max_extra_param)) {
        continue;
      }

      const Point3D& point3D =
          reconstruction_->Point3D(corr_point2D.point3D_id);

      tri_corrs.emplace_back(point2D_idx, corr_point2D.point3D_id);
      corr_point3D_ids.insert(corr_point2D.point3D_id);
      tri_points2D.push_back(point2D.xy);
      tri_points3D.push_back(point3D.xyz);
    }
  }

  // The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()`
  // can only differ, when there are images with bogus camera parameters, and
  // hence we skip some of the 2D-3D correspondences.
  if (tri_points2D.size() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    VLOG(2) << "Insufficient number of 2D-3D correspondences for registration ("
            << tri_points2D.size() << " < " << options.abs_pose_min_num_inliers
            << ")";
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 2D-3D estimation
  //////////////////////////////////////////////////////////////////////////////

  // Only refine / estimate focal length, if no focal length was specified
  // (manually or through EXIF) and if it was not already estimated previously
  // from another image (when multiple images share the same camera parameters).

  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio =
      options.abs_pose_min_inlier_ratio;
  abs_pose_options.ransac_options.random_seed = options.random_seed;

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  if (options.constant_cameras.count(image.CameraId()) > 0) {
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
    abs_pose_refinement_options.refine_extra_params = false;
  } else {
    if (reg_stats_.num_reg_images_per_camera[image.CameraId()] > 0) {
      // Camera already refined from another image with the same camera.
      if (camera.HasBogusParams(options.min_focal_length_ratio,
                                options.max_focal_length_ratio,
                                options.max_extra_param)) {
        abs_pose_options.estimate_focal_length = !camera.has_prior_focal_length;
        abs_pose_refinement_options.refine_focal_length = true;
        abs_pose_refinement_options.refine_extra_params = true;
      } else {
        abs_pose_options.estimate_focal_length = false;
        abs_pose_refinement_options.refine_focal_length = false;
        abs_pose_refinement_options.refine_extra_params = false;
      }
    } else {
      // Camera not refined before. Note that the camera parameters might have
      // been changed before but the image was filtered, so we explicitly reset
      // the camera parameters and try to re-estimate them.
      camera.params = database_cache_->Camera(image.CameraId()).params;
      abs_pose_options.estimate_focal_length = !camera.has_prior_focal_length;
      abs_pose_refinement_options.refine_focal_length = true;
      abs_pose_refinement_options.refine_extra_params = true;
    }

    if (!options.abs_pose_refine_focal_length) {
      abs_pose_options.estimate_focal_length = false;
      abs_pose_refinement_options.refine_focal_length = false;
    }

    if (!options.abs_pose_refine_extra_params) {
      abs_pose_refinement_options.refine_extra_params = false;
    }
  }

  // If any of the cameras in the same rig has bogus cameras, reset them to the
  // original values from the database, so we have a chance of recovering from
  // previous failed estimations. Notice that this function will be called for
  // non-trivial frames, when there is no good estimate for the focal length or
  // one of the rig's cameras has bogus parameters.
  for (const data_t& data_id : image.FramePtr()->ImageIds()) {
    Image& frame_image = reconstruction_->Image(data_id.id);
    if (frame_image.CameraPtr()->HasBogusParams(options.min_focal_length_ratio,
                                                options.max_focal_length_ratio,
                                                options.max_extra_param)) {
      VLOG(2) << "Resetting camera " << frame_image.CameraId()
              << "'s bogus parameters";
      frame_image.CameraPtr()->params =
          database_cache_->Camera(frame_image.CameraId()).params;
    }
  }

  size_t num_inliers;
  std::vector<char> inlier_mask;
  Rigid3d cam_from_world;
  if (!EstimateAbsolutePose(abs_pose_options,
                            tri_points2D,
                            tri_points3D,
                            &cam_from_world,
                            &camera,
                            &num_inliers,
                            &inlier_mask)) {
    VLOG(2) << "Absolute pose estimation failed";
    return false;
  }

  if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    VLOG(2) << "Absolute pose estimation failed due to insufficient inliers ("
            << num_inliers << " < " << options.abs_pose_min_num_inliers << ")";
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Pose refinement
  //////////////////////////////////////////////////////////////////////////////

  if (!RefineAbsolutePose(abs_pose_refinement_options,
                          inlier_mask,
                          tri_points2D,
                          tri_points3D,
                          &cam_from_world,
                          &camera)) {
    VLOG(2) << "Absolute pose refinement failed";
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Continue tracks
  //////////////////////////////////////////////////////////////////////////////

  VLOG(2) << "Continuing tracks for " << num_inliers
          << " inlier 2D-3D correspondences";

  image.FramePtr()->SetCamFromWorld(image.CameraId(), cam_from_world);

  reconstruction_->RegisterFrame(image.FrameId());
  RegisterFrameEvent(image.FrameId());

  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const auto [point2D_idx, point3D_id] = tri_corrs[i];
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (!point2D.HasPoint3D()) {
        const TrackElement track_el(image_id, point2D_idx);
        obs_manager_->AddObservation(point3D_id, track_el);
        triangulator_->AddModifiedPoint3D(point3D_id);
      }
    }
  }

  return true;
}

bool IncrementalMapper::RegisterNextGeneralFrame(const Options& options,
                                                 Frame& frame) {
  // Only call this method for frames with more than
  THROW_CHECK_GT(frame.RigPtr()->NumSensors(), 1);

  struct Corr {
    point2D_t point2D_idx;
    image_t image_id;
    point3D_t point3D_id;
  };

  std::vector<Corr> tri_corrs;
  std::vector<Eigen::Vector2d> tri_points2D;
  std::vector<Eigen::Vector3d> tri_points3D;
  std::vector<size_t> tri_camera_idxs;

  std::vector<Rigid3d> cams_from_rig;
  cams_from_rig.reserve(frame.RigPtr()->NumSensors());
  std::vector<Camera> cameras;
  cameras.reserve(frame.RigPtr()->NumSensors());

  const std::shared_ptr<const CorrespondenceGraph> correspondence_graph =
      database_cache_->CorrespondenceGraph();

  for (const data_t& data_id : frame.ImageIds()) {
    const image_t image_id = data_id.id;
    const Image& image = reconstruction_->Image(image_id);
    const Camera& camera = *image.CameraPtr();

    const size_t camera_idx = cameras.size();
    if (frame.RigPtr()->IsRefSensor(camera.SensorId())) {
      cams_from_rig.push_back(Rigid3d());
    } else {
      cams_from_rig.push_back(frame.RigPtr()->SensorFromRig(camera.SensorId()));
    }
    cameras.push_back(camera);

    reg_stats_.num_reg_trials[image_id] += 1;

    std::unordered_set<point3D_t> corr_point3D_ids;
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const Point2D& point2D = image.Point2D(point2D_idx);

      corr_point3D_ids.clear();
      const auto corr_range =
          correspondence_graph->FindCorrespondences(image_id, point2D_idx);
      for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
        const Image& corr_image = reconstruction_->Image(corr->image_id);
        if (!corr_image.HasPose()) {
          continue;
        }

        const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
        if (!corr_point2D.HasPoint3D()) {
          continue;
        }

        // Avoid duplicate correspondences.
        if (corr_point3D_ids.count(corr_point2D.point3D_id) > 0) {
          continue;
        }

        const Camera& corr_camera = *corr_image.CameraPtr();

        // Avoid correspondences to images with bogus camera parameters.
        if (corr_camera.HasBogusParams(options.min_focal_length_ratio,
                                       options.max_focal_length_ratio,
                                       options.max_extra_param)) {
          continue;
        }

        const Point3D& point3D =
            reconstruction_->Point3D(corr_point2D.point3D_id);

        tri_corrs.push_back(
            Corr{point2D_idx, image_id, corr_point2D.point3D_id});
        corr_point3D_ids.insert(corr_point2D.point3D_id);
        tri_points2D.push_back(point2D.xy);
        tri_points3D.push_back(point3D.xyz);
        tri_camera_idxs.push_back(camera_idx);
      }
    }
  }

  // The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()`
  // can only differ, when there are images with bogus camera parameters, and
  // hence we skip some of the 2D-3D correspondences.
  if (tri_points2D.size() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    VLOG(2) << "Insufficient number of 2D-3D correspondences for registration ("
            << tri_points2D.size() << " < " << options.abs_pose_min_num_inliers
            << ")";
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 2D-3D estimation
  //////////////////////////////////////////////////////////////////////////////

  // Only refine focal length, if no focal length was specified
  // (manually or through EXIF) and if it was not already estimated previously
  // from another image (when multiple images share the same camera parameters).

  RANSACOptions abs_pose_options;
  abs_pose_options.max_error = options.abs_pose_max_error;
  abs_pose_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;
  abs_pose_options.random_seed = options.random_seed;

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  abs_pose_refinement_options.refine_focal_length = false;
  abs_pose_refinement_options.refine_extra_params = false;

  size_t num_inliers;
  std::vector<char> inlier_mask;
  Rigid3d rig_from_world;
  if (!EstimateGeneralizedAbsolutePose(abs_pose_options,
                                       tri_points2D,
                                       tri_points3D,
                                       tri_camera_idxs,
                                       cams_from_rig,
                                       cameras,
                                       &rig_from_world,
                                       &num_inliers,
                                       &inlier_mask)) {
    VLOG(2) << "Absolute pose estimation failed";
    return false;
  }

  if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    VLOG(2) << "Absolute pose estimation failed due to insufficient inliers ("
            << num_inliers << " < " << options.abs_pose_min_num_inliers << ")";
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Pose refinement
  //////////////////////////////////////////////////////////////////////////////

  if (!RefineGeneralizedAbsolutePose(abs_pose_refinement_options,
                                     inlier_mask,
                                     tri_points2D,
                                     tri_points3D,
                                     tri_camera_idxs,
                                     cams_from_rig,
                                     &rig_from_world,
                                     &cameras)) {
    VLOG(2) << "Absolute pose refinement failed";
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Continue tracks
  //////////////////////////////////////////////////////////////////////////////

  VLOG(2) << "Continuing tracks for " << num_inliers
          << " inlier 2D-3D correspondences";

  frame.SetRigFromWorld(rig_from_world);

  reconstruction_->RegisterFrame(frame.FrameId());
  RegisterFrameEvent(frame.FrameId());

  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const Corr& corr = tri_corrs[i];
      const Image& image = reconstruction_->Image(corr.image_id);
      const Point2D& point2D = image.Point2D(corr.point2D_idx);
      if (!point2D.HasPoint3D()) {
        const TrackElement track_el(corr.image_id, corr.point2D_idx);
        obs_manager_->AddObservation(corr.point3D_id, track_el);
        triangulator_->AddModifiedPoint3D(corr.point3D_id);
      }
    }
  }

  return true;
}

/**
 * [功能描述]：对指定图像进行三角化，通过与其他已注册图像的特征匹配生成新的3D点。
 * @param tri_options：三角化的配置选项，控制三角化的行为和阈值。
 * @param image_id：要进行三角化的图像ID。
 * @return 新增的三角化观测点数量。
 */
size_t IncrementalMapper::TriangulateImage(
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id) {
  // 确保重建对象已初始化
  THROW_CHECK_NOTNULL(reconstruction_);

  // 输出该图像当前已关联的3D点数量（继承自之前的观测）
  VLOG(1) << "=> Continued observations: "
          << reconstruction_->Image(image_id).NumPoints3D();

  // 调用三角化器对该图像进行三角化，返回新增的观测点数量
  const size_t num_tris =
      triangulator_->TriangulateImage(tri_options, image_id);

  // 输出新增的观测点数量
  VLOG(1) << "=> Added observations: " << num_tris;

  return num_tris;
}

size_t IncrementalMapper::Retriangulate(
    const IncrementalTriangulator::Options& tri_options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  return triangulator_->Retriangulate(tri_options);
}

size_t IncrementalMapper::CompleteTracks(
    const IncrementalTriangulator::Options& tri_options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  return triangulator_->CompleteAllTracks(tri_options);
}

size_t IncrementalMapper::MergeTracks(
    const IncrementalTriangulator::Options& tri_options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  return triangulator_->MergeAllTracks(tri_options);
}

size_t IncrementalMapper::CompleteAndMergeTracks(
    const IncrementalTriangulator::Options& tri_options) {
  const size_t num_completed_observations = CompleteTracks(tri_options);
  VLOG(1) << "=> Completed observations: " << num_completed_observations;
  const size_t num_merged_observations = MergeTracks(tri_options);
  VLOG(1) << "=> Merged observations: " << num_merged_observations;
  return num_completed_observations + num_merged_observations;
}

IncrementalMapper::LocalBundleAdjustmentReport
IncrementalMapper::AdjustLocalBundle(
    const Options& options,
    const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id,
    const std::unordered_set<point3D_t>& point3D_ids) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(obs_manager_);
  THROW_CHECK(options.Check());

  LocalBundleAdjustmentReport report;

  // Find images that have most 3D points with given image in common.
  const std::vector<image_t> local_bundle = FindLocalBundle(options, image_id);

  // Do the bundle adjustment only if there is any connected images.
  BundleAdjustmentConfig ba_config;
  std::unordered_set<image_t> image_ids;
  if (local_bundle.size() > 0) {
    ba_config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);

    // Insert the images of all local frames.
    const Image& image = reconstruction_->Image(image_id);
    std::set<frame_t> frame_ids;
    frame_ids.insert(image.FrameId());
    for (const data_t& data_id : image.FramePtr()->ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
    for (const image_t local_image_id : local_bundle) {
      const Image& local_image = reconstruction_->Image(local_image_id);
      frame_ids.insert(local_image.FrameId());
      for (const data_t& data_id : local_image.FramePtr()->ImageIds()) {
        ba_config.AddImage(data_id.id);
      }
    }

    // Fix the existing images, if option specified.
    if (options.fix_existing_frames) {
      for (const frame_t frame_id : frame_ids) {
        if (existing_frame_ids_.count(frame_id)) {
          ba_config.SetConstantRigFromWorldPose(frame_id);
        }
      }
    }

    // Fix rig poses, if not all frames within the local bundle.
    std::unordered_map<rig_t, size_t> num_frames_per_rig;
    num_frames_per_rig.reserve(frame_ids.size());
    for (const frame_t frame_id : frame_ids) {
      const Frame& frame = reconstruction_->Frame(frame_id);
      num_frames_per_rig[frame.RigId()] += 1;
    }
    for (const auto& [rig_id, num_frames] : num_frames_per_rig) {
      if (options.constant_rigs.count(rig_id) ||
          num_frames < reg_stats_.num_reg_frames_per_rig.at(rig_id)) {
        const Rig& rig = reconstruction_->Rig(rig_id);
        for (const auto& [sensor_id, _] : rig.Sensors()) {
          ba_config.SetConstantSensorFromRigPose(sensor_id);
        }
      }
    }

    // Fix camera intrinsics, if not all registered images within local bundle.
    std::unordered_map<camera_t, size_t> num_images_per_camera;
    num_images_per_camera.reserve(ba_config.NumImages());
    for (const image_t image_id : ba_config.Images()) {
      const Image& image = reconstruction_->Image(image_id);
      num_images_per_camera[image.CameraId()] += 1;
    }
    for (const auto& [camera_id, num_images] : num_images_per_camera) {
      if (options.constant_cameras.count(camera_id) ||
          num_images < reg_stats_.num_reg_images_per_camera.at(camera_id)) {
        ba_config.SetConstantCamIntrinsics(camera_id);
      }
    }

    // Make sure, we refine all new and short-track 3D points, no matter if
    // they are fully contained in the local image set or not. Do not include
    // long track 3D points as they are usually already very stable and adding
    // to them to bundle adjustment and track merging/completion would slow
    // down the local bundle adjustment significantly.
    std::unordered_set<point3D_t> variable_point3D_ids;
    for (const point3D_t point3D_id : point3D_ids) {
      const Point3D& point3D = reconstruction_->Point3D(point3D_id);
      constexpr size_t kMaxTrackLength = 15;
      if (!point3D.HasError() || point3D.track.Length() <= kMaxTrackLength) {
        ba_config.AddVariablePoint(point3D_id);
        variable_point3D_ids.insert(point3D_id);
      }
    }

    // Adjust the local bundle.
    image_ids = ba_config.Images();
    std::unique_ptr<BundleAdjuster> bundle_adjuster =
        CreateDefaultBundleAdjuster(
            ba_options, std::move(ba_config), *reconstruction_);
    const ceres::Solver::Summary summary = bundle_adjuster->Solve();

    report.num_adjusted_observations = summary.num_residuals / 2;

    // Merge refined tracks with other existing points.
    report.num_merged_observations =
        triangulator_->MergeTracks(tri_options, variable_point3D_ids);
    // Complete tracks that may have failed to triangulate before refinement
    // of camera pose and calibration in bundle-adjustment. This may avoid
    // that some points are filtered and it helps for subsequent image
    // registrations.
    report.num_completed_observations =
        triangulator_->CompleteTracks(tri_options, variable_point3D_ids);
    report.num_completed_observations +=
        triangulator_->CompleteImage(tri_options, image_id);
  }

  // Filter both the modified images and all changed 3D points to make sure
  // there are no outlier points in the model. This results in duplicate work as
  // many of the provided 3D points may also be contained in the adjusted
  // images, but the filtering is not a bottleneck at this point.
  report.num_filtered_observations = obs_manager_->FilterPoints3DInImages(
      options.filter_max_reproj_error, options.filter_min_tri_angle, image_ids);
  report.num_filtered_observations +=
      obs_manager_->FilterPoints3D(options.filter_max_reproj_error,
                                   options.filter_min_tri_angle,
                                   point3D_ids);

  return report;
}

/**
 * [功能描述]：执行全局光束法平差（Bundle Adjustment），同时优化所有已注册图像的相机位姿和3D点坐标。
 * @param options：增量式重建的配置选项。
 * @param ba_options：光束法平差的配置选项，包含求解器参数等。
 * @return true表示优化成功，false表示优化失败。
 */
bool IncrementalMapper::AdjustGlobalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options) {
  // 确保重建对象和观测管理器已初始化
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(obs_manager_);

  // 复制BA选项以便自定义修改
  BundleAdjustmentOptions custom_ba_options = ba_options;

  // 对于前期注册的图像（少于10帧），使用更严格的收敛条件以获得更精确的初始重建
  const size_t kMinNumRegFramesForFastBA = 10;
  if (reconstruction_->NumRegFrames() < kMinNumRegFramesForFastBA) {
    // 收敛容差缩小10倍，迭代次数翻倍
    custom_ba_options.solver_options.function_tolerance /= 10;
    custom_ba_options.solver_options.gradient_tolerance /= 10;
    custom_ba_options.solver_options.parameter_tolerance /= 10;
    custom_ba_options.solver_options.max_num_iterations *= 2;
    custom_ba_options.solver_options.max_linear_solver_iterations = 200;
  }

  // 过滤负深度的观测点，避免光束法平差中的退化情况
  obs_manager_->FilterObservationsWithNegativeDepth();

  // 配置光束法平差：将所有已注册帧中的图像添加到BA配置中
  BundleAdjustmentConfig ba_config;
  for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
    const Frame& frame = reconstruction_->Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      ba_config.AddImage(data_id.id);
    }
  }

  // 确保至少有2张图像用于全局BA
  THROW_CHECK_GE(ba_config.NumImages(), 2) << "At least two images must be "
                                              "registered for global "
                                              "bundle-adjustment";

  // 如果选项指定固定已存在的帧，则将这些帧的位姿设为常量（不优化）
  if (options.fix_existing_frames) {
    for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
      if (existing_frame_ids_.count(frame_id)) {
        ba_config.SetConstantRigFromWorldPose(frame_id);
      }
    }
  }

  // 将指定的Rig中所有传感器的相对位姿设为常量
  for (const auto& rig_id : options.constant_rigs) {
    const Rig& rig = reconstruction_->Rig(rig_id);
    for (const auto& [sensor_id, _] : rig.Sensors()) {
      ba_config.SetConstantSensorFromRigPose(sensor_id);
    }
  }

  // 将指定的相机内参设为常量（不优化）
  for (const auto& camera_id : options.constant_cameras) {
    ba_config.SetConstantCamIntrinsics(camera_id);
  }

  // 仅当至少有3张图像注册时才使用位姿先验
  const bool use_prior_position =
      options.use_prior_position && ba_config.NumImages() > 2;

  // 根据是否使用位姿先验创建相应的BA优化器
  std::unique_ptr<BundleAdjuster> bundle_adjuster;
  if (!use_prior_position) {
    // 不使用位姿先验：通过固定两个相机位姿来固定尺度规范（Gauge）
    // 相比固定三个点，这种方式优化更稳定、收敛更快
    // TODO(jsch): 研究是否可以完全不固定规范，初步实验表明这样更快
    ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
    bundle_adjuster = CreateDefaultBundleAdjuster(
        std::move(custom_ba_options), ba_config, *reconstruction_);
  } else {
    // 使用位姿先验：配置位姿先验BA选项
    PosePriorBundleAdjustmentOptions prior_options;
    prior_options.use_robust_loss_on_prior_position =
        options.use_robust_loss_on_prior_position;  // 是否对位姿先验使用鲁棒损失函数
    prior_options.prior_position_loss_scale = options.prior_position_loss_scale;  // 位姿先验损失缩放因子
    prior_options.alignment_ransac_options.random_seed = options.random_seed;  // RANSAC随机种子
    bundle_adjuster =
        CreatePosePriorBundleAdjuster(std::move(custom_ba_options),
                                      prior_options,
                                      ba_config,
                                      database_cache_->PosePriors(),
                                      *reconstruction_);
  }

  // 执行BA优化，返回是否成功（非FAILURE即为成功）
  return bundle_adjuster->Solve().termination_type != ceres::FAILURE;
}

void IncrementalMapper::IterativeLocalRefinement(
    const int max_num_refinements,
    const double max_refinement_change,
    const Options& options,
    const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id) {
  BundleAdjustmentOptions custom_ba_options = ba_options;
  for (int i = 0; i < max_num_refinements; ++i) {
    const auto report = AdjustLocalBundle(options,
                                          custom_ba_options,
                                          tri_options,
                                          image_id,
                                          GetModifiedPoints3D());
    VLOG(1) << "=> Merged observations: " << report.num_merged_observations;
    VLOG(1) << "=> Completed observations: "
            << report.num_completed_observations;
    VLOG(1) << "=> Filtered observations: " << report.num_filtered_observations;
    const double changed =
        report.num_adjusted_observations == 0
            ? 0
            : (report.num_merged_observations +
               report.num_completed_observations +
               report.num_filtered_observations) /
                  static_cast<double>(report.num_adjusted_observations);
    VLOG(1) << StringPrintf("=> Changed observations: %.6f", changed);
    if (changed < max_refinement_change) {
      break;
    }
    // Only use robust cost function for first iteration.
    custom_ba_options.loss_function_type =
        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  }
  ClearModifiedPoints3D();
}

void IncrementalMapper::IterativeGlobalRefinement(
    const int max_num_refinements,
    const double max_refinement_change,
    const Options& options,
    const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options,
    const bool normalize_reconstruction) {
  CompleteAndMergeTracks(tri_options);
  VLOG(1) << "=> Retriangulated observations: " << Retriangulate(tri_options);
  for (int i = 0; i < max_num_refinements; ++i) {
    const size_t num_observations = reconstruction_->ComputeNumObservations();
    AdjustGlobalBundle(options, ba_options);
    if (normalize_reconstruction && !options.use_prior_position) {
      // Normalize scene for numerical stability and
      // to avoid large scale changes in the viewer.
      reconstruction_->Normalize();
    }
    size_t num_changed_observations = CompleteAndMergeTracks(tri_options);
    num_changed_observations += FilterPoints(options);
    const double changed =
        num_observations == 0
            ? 0
            : static_cast<double>(num_changed_observations) / num_observations;
    VLOG(1) << StringPrintf("=> Changed observations: %.6f", changed);
    if (changed < max_refinement_change) {
      break;
    }
  }
  ClearModifiedPoints3D();
}

size_t IncrementalMapper::FilterFrames(const Options& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(obs_manager_);
  THROW_CHECK(options.Check());

  // Do not filter frames in the early stage of the reconstruction, since the
  // calibration is often still refining a lot. Hence, the camera parameters
  // are not stable in the beginning.
  const size_t kMinNumFrames = 20;
  if (reconstruction_->NumRegFrames() < kMinNumFrames) {
    return {};
  }

  const std::vector<frame_t> frame_ids =
      obs_manager_->FilterFrames(options.min_focal_length_ratio,
                                 options.max_focal_length_ratio,
                                 options.max_extra_param);

  for (const frame_t frame_id : frame_ids) {
    if (!options.fix_existing_frames ||
        existing_frame_ids_.count(frame_id) == 0) {
      DeRegisterFrameEvent(frame_id);
      filtered_frames_.insert(frame_id);
    }
  }

  const size_t num_filtered_frames = frame_ids.size();
  VLOG(1) << "=> Filtered frames: " << num_filtered_frames;
  return num_filtered_frames;
}

size_t IncrementalMapper::FilterPoints(const Options& options) {
  THROW_CHECK_NOTNULL(obs_manager_);
  THROW_CHECK(options.Check());
  const size_t num_filtered_observations = obs_manager_->FilterAllPoints3D(
      options.filter_max_reproj_error, options.filter_min_tri_angle);
  VLOG(1) << "=> Filtered observations: " << num_filtered_observations;
  return num_filtered_observations;
}

std::shared_ptr<class Reconstruction> IncrementalMapper::Reconstruction()
    const {
  return reconstruction_;
}

class ObservationManager& IncrementalMapper::ObservationManager() const {
  THROW_CHECK_NOTNULL(obs_manager_);
  return *obs_manager_;
}

IncrementalTriangulator& IncrementalMapper::Triangulator() const {
  THROW_CHECK_NOTNULL(triangulator_);
  return *triangulator_;
}

const std::unordered_set<frame_t>& IncrementalMapper::FilteredFrames() const {
  return filtered_frames_;
}

const std::unordered_set<image_t>& IncrementalMapper::ExistingFrameIds() const {
  return existing_frame_ids_;
}

void IncrementalMapper::ResetInitializationStats() {
  reg_stats_.init_image_pairs.clear();
  reg_stats_.init_num_reg_trials.clear();
}

const std::unordered_map<rig_t, size_t>& IncrementalMapper::NumRegFramesPerRig()
    const {
  return reg_stats_.num_reg_frames_per_rig;
}

const std::unordered_map<camera_t, size_t>&
IncrementalMapper::NumRegImagesPerCamera() const {
  return reg_stats_.num_reg_images_per_camera;
}

size_t IncrementalMapper::NumTotalRegImages() const {
  return reg_stats_.num_total_reg_images;
}

size_t IncrementalMapper::NumSharedRegImages() const {
  return reg_stats_.num_shared_reg_images;
}

const std::unordered_set<point3D_t>& IncrementalMapper::GetModifiedPoints3D() {
  return triangulator_->GetModifiedPoints3D();
}

void IncrementalMapper::ClearModifiedPoints3D() {
  triangulator_->ClearModifiedPoints3D();
}

std::vector<image_t> IncrementalMapper::FindLocalBundle(
    const Options& options, const image_t image_id) const {
  return IncrementalMapperImpl::FindLocalBundle(
      options, image_id, *reconstruction_);
}

/**
 * [功能描述]：处理帧注册事件，更新与帧、图像、相机相关的注册统计信息。
 * @param frame_id：要处理的帧ID。
 * @return 无返回值。
 */
void IncrementalMapper::RegisterFrameEvent(const frame_t frame_id) {
  // 根据帧ID获取对应的Frame对象
  const Frame& frame = reconstruction_->Frame(frame_id);

  // 获取该帧所属Rig的注册帧计数引用，并递增
  size_t& num_reg_frames_for_rig =
      reg_stats_.num_reg_frames_per_rig[frame.RigId()];
  num_reg_frames_for_rig += 1;

  // 遍历该帧中包含的所有图像
  for (const data_t& data_id : frame.ImageIds()) {
    // 获取图像对象
    const Image& image = reconstruction_->Image(data_id.id);

    // 获取该图像所属相机的注册图像计数引用，并递增
    size_t& num_reg_images_for_camera =
        reg_stats_.num_reg_images_per_camera[image.CameraId()];
    num_reg_images_for_camera += 1;

    // 获取该图像的注册次数引用，并递增
    size_t& num_regs_for_image = reg_stats_.num_registrations[data_id.id];
    num_regs_for_image += 1;

    // 根据注册次数更新统计信息
    if (num_regs_for_image == 1) {
      // 首次注册：增加总注册图像数
      reg_stats_.num_total_reg_images += 1;
    } else if (num_regs_for_image > 1) {
      // 多次注册：增加共享注册图像数（表示该图像被多个重建共享使用）
      reg_stats_.num_shared_reg_images += 1;
    }
  }
}

void IncrementalMapper::DeRegisterFrameEvent(const frame_t frame_id) {
  const Frame& frame = reconstruction_->Frame(frame_id);

  size_t& num_reg_frames_for_rig =
      reg_stats_.num_reg_frames_per_rig.at(frame.RigId());
  THROW_CHECK_GT(num_reg_frames_for_rig, 0);
  num_reg_frames_for_rig -= 1;

  for (const data_t& data_id : frame.ImageIds()) {
    const Image& image = reconstruction_->Image(data_id.id);

    size_t& num_reg_images_for_camera =
        reg_stats_.num_reg_images_per_camera.at(image.CameraId());
    THROW_CHECK_GT(num_reg_images_for_camera, 0);
    num_reg_images_for_camera -= 1;

    size_t& num_regs_for_image = reg_stats_.num_registrations[data_id.id];
    num_regs_for_image -= 1;
    if (num_regs_for_image == 0) {
      reg_stats_.num_total_reg_images -= 1;
    } else if (num_regs_for_image > 0) {
      reg_stats_.num_shared_reg_images -= 1;
    }
  }
}

bool IncrementalMapper::EstimateInitialTwoViewGeometry(
    const IncrementalMapper::Options& options,
    const image_t image_id1,
    const image_t image_id2,
    Rigid3d& cam2_from_cam1) {
  return IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
      options, *database_cache_, image_id1, image_id2, cam2_from_cam1);
}

}  // namespace colmap
