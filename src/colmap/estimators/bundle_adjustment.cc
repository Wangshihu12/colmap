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

#include "colmap/estimators/bundle_adjustment.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/manifold.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <iomanip>

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

void BundleAdjustmentConfig::FixGauge(BundleAdjustmentGauge gauge) {
  fixed_gauge_ = gauge;
}

BundleAdjustmentGauge BundleAdjustmentConfig::FixedGauge() const {
  return fixed_gauge_;
}

size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentConfig::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantCamIntrinsics() const {
  return constant_cam_intrinsics_.size();
}

size_t BundleAdjustmentConfig::NumConstantSensorFromRigPoses() const {
  return constant_sensor_from_rig_poses_.size();
}

size_t BundleAdjustmentConfig::NumConstantRigFromWorldPoses() const {
  return constant_rig_from_world_poses_.size();
}

size_t BundleAdjustmentConfig::NumVariablePoints() const {
  return variable_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoints() const {
  return constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumResiduals(
    const Reconstruction& reconstruction) const {
  // Count the number of observations for all added images.
  size_t num_observations = 0;
  for (const image_t image_id : image_ids_) {
    num_observations += reconstruction.Image(image_id).NumPoints3D();
  }

  // Count the number of observations for all added 3D points that are not
  // already added as part of the images above.

  auto NumObservationsForPoint = [this,
                                  &reconstruction](const point3D_t point3D_id) {
    size_t num_observations_for_point = 0;
    const auto& point3D = reconstruction.Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      if (image_ids_.count(track_el.image_id) == 0) {
        ++num_observations_for_point;
      }
    }
    return num_observations_for_point;
  };

  for (const auto point3D_id : variable_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }
  for (const auto point3D_id : constant_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }

  return 2 * num_observations;
}

void BundleAdjustmentConfig::AddImage(const image_t image_id) {
  image_ids_.insert(image_id);
}

bool BundleAdjustmentConfig::HasImage(const image_t image_id) const {
  return image_ids_.find(image_id) != image_ids_.end();
}

void BundleAdjustmentConfig::RemoveImage(const image_t image_id) {
  image_ids_.erase(image_id);
}

void BundleAdjustmentConfig::SetConstantCamIntrinsics(
    const camera_t camera_id) {
  constant_cam_intrinsics_.insert(camera_id);
}

void BundleAdjustmentConfig::SetVariableCamIntrinsics(
    const camera_t camera_id) {
  constant_cam_intrinsics_.erase(camera_id);
}

bool BundleAdjustmentConfig::HasConstantCamIntrinsics(
    const camera_t camera_id) const {
  return constant_cam_intrinsics_.find(camera_id) !=
         constant_cam_intrinsics_.end();
}

void BundleAdjustmentConfig::SetConstantSensorFromRigPose(
    const sensor_t sensor_id) {
  constant_sensor_from_rig_poses_.insert(sensor_id);
}

void BundleAdjustmentConfig::SetVariableSensorFromRigPose(
    const sensor_t sensor_id) {
  constant_sensor_from_rig_poses_.erase(sensor_id);
}

bool BundleAdjustmentConfig::HasConstantSensorFromRigPose(
    const sensor_t sensor_id) const {
  return constant_sensor_from_rig_poses_.find(sensor_id) !=
         constant_sensor_from_rig_poses_.end();
}

void BundleAdjustmentConfig::SetConstantRigFromWorldPose(
    const frame_t frame_id) {
  constant_rig_from_world_poses_.insert(frame_id);
}

void BundleAdjustmentConfig::SetVariableRigFromWorldPose(
    const frame_t frame_id) {
  constant_rig_from_world_poses_.erase(frame_id);
}

bool BundleAdjustmentConfig::HasConstantRigFromWorldPose(
    const frame_t frame_id) const {
  return constant_rig_from_world_poses_.find(frame_id) !=
         constant_rig_from_world_poses_.end();
}

const std::unordered_set<image_t>& BundleAdjustmentConfig::Images() const {
  return image_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::VariablePoints()
    const {
  return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::ConstantPoints()
    const {
  return constant_point3D_ids_;
}

const std::unordered_set<camera_t>
BundleAdjustmentConfig::ConstantCamIntrinsics() const {
  return constant_cam_intrinsics_;
}

const std::unordered_set<sensor_t>&
BundleAdjustmentConfig::ConstantSensorFromRigPoses() const {
  return constant_sensor_from_rig_poses_;
}

const std::unordered_set<frame_t>&
BundleAdjustmentConfig::ConstantRigFromWorldPoses() const {
  return constant_rig_from_world_poses_;
}

void BundleAdjustmentConfig::AddVariablePoint(const point3D_t point3D_id) {
  THROW_CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentConfig::AddConstantPoint(const point3D_t point3D_id) {
  THROW_CHECK(!HasVariablePoint(point3D_id));
  constant_point3D_ids_.insert(point3D_id);
}

bool BundleAdjustmentConfig::HasPoint(const point3D_t point3D_id) const {
  return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool BundleAdjustmentConfig::HasVariablePoint(
    const point3D_t point3D_id) const {
  return variable_point3D_ids_.find(point3D_id) != variable_point3D_ids_.end();
}

bool BundleAdjustmentConfig::HasConstantPoint(
    const point3D_t point3D_id) const {
  return constant_point3D_ids_.find(point3D_id) != constant_point3D_ids_.end();
}

void BundleAdjustmentConfig::RemoveVariablePoint(const point3D_t point3D_id) {
  variable_point3D_ids_.erase(point3D_id);
}

void BundleAdjustmentConfig::RemoveConstantPoint(const point3D_t point3D_id) {
  constant_point3D_ids_.erase(point3D_id);
}

BundleAdjuster::BundleAdjuster(BundleAdjustmentOptions options,
                               BundleAdjustmentConfig config)
    : options_(std::move(options)), config_(std::move(config)) {
  THROW_CHECK(options_.Check());
}

const BundleAdjustmentOptions& BundleAdjuster::Options() const {
  return options_;
}

const BundleAdjustmentConfig& BundleAdjuster::Config() const { return config_; }

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

ceres::LossFunction* BundleAdjustmentOptions::CreateLossFunction() const {
  ceres::LossFunction* loss_function = nullptr;
  switch (loss_function_type) {
    case LossFunctionType::TRIVIAL:
      loss_function = new ceres::TrivialLoss();
      break;
    case LossFunctionType::SOFT_L1:
      loss_function = new ceres::SoftLOneLoss(loss_function_scale);
      break;
    case LossFunctionType::CAUCHY:
      loss_function = new ceres::CauchyLoss(loss_function_scale);
      break;
  }
  THROW_CHECK_NOTNULL(loss_function);
  return loss_function;
}

ceres::Solver::Options BundleAdjustmentOptions::CreateSolverOptions(
    const BundleAdjustmentConfig& config, const ceres::Problem& problem) const {
  ceres::Solver::Options custom_solver_options = solver_options;
  if (VLOG_IS_ON(2)) {
    custom_solver_options.minimizer_progress_to_stdout = true;
    custom_solver_options.logging_type =
        ceres::LoggingType::PER_MINIMIZER_ITERATION;
  }

  const int num_images = config.NumImages();
  const bool has_sparse =
      custom_solver_options.sparse_linear_algebra_library_type !=
      ceres::NO_SPARSE;

  int max_num_images_direct_dense_solver =
      max_num_images_direct_dense_cpu_solver;
  int max_num_images_direct_sparse_solver =
      max_num_images_direct_sparse_cpu_solver;

#ifdef COLMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (use_gpu && num_images >= min_num_images_gpu_solver) {
    cuda_solver_enabled = true;
    custom_solver_options.dense_linear_algebra_library_type = ceres::CUDA;
    max_num_images_direct_dense_solver = max_num_images_direct_dense_gpu_solver;
  }
#else
  if (use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without CUDA support. Falling back to CPU-based dense "
           "solvers.";
  }
#endif

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3)) && \
    !defined(CERES_NO_CUDSS)
  if (use_gpu && num_images >= min_num_images_gpu_solver) {
    cuda_solver_enabled = true;
    custom_solver_options.sparse_linear_algebra_library_type =
        ceres::CUDA_SPARSE;
    max_num_images_direct_sparse_solver =
        max_num_images_direct_sparse_gpu_solver;
  }
#else
  if (use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without cuDSS support. Falling back to CPU-based sparse "
           "solvers.";
  }
#endif

  if (cuda_solver_enabled) {
    const std::vector<int> gpu_indices = CSVToVector<int>(gpu_index);
    THROW_CHECK_GT(gpu_indices.size(), 0);
    SetBestCudaDevice(gpu_indices[0]);
  }
#else
  if (use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but COLMAP was "
           "compiled without CUDA support. Falling back to CPU-based "
           "solvers.";
  }
#endif  // COLMAP_CUDA_ENABLED

  if (num_images <= max_num_images_direct_dense_solver) {
    custom_solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (has_sparse && num_images <= max_num_images_direct_sparse_solver) {
    custom_solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    custom_solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    custom_solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  if (problem.NumResiduals() < min_num_residuals_for_cpu_multi_threading) {
    custom_solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    custom_solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR
  } else {
    custom_solver_options.num_threads =
        GetEffectiveNumThreads(custom_solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
    custom_solver_options.num_linear_solver_threads =
        GetEffectiveNumThreads(custom_solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  }

  std::string solver_error;
  THROW_CHECK(custom_solver_options.IsValid(&solver_error)) << solver_error;
  return custom_solver_options;
}

bool BundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GE(loss_function_scale, 0);
  CHECK_OPTION_LT(max_num_images_direct_dense_cpu_solver,
                  max_num_images_direct_sparse_cpu_solver);
  CHECK_OPTION_LT(max_num_images_direct_dense_gpu_solver,
                  max_num_images_direct_sparse_gpu_solver);
  return true;
}

namespace {

/**
 * [功能描述]：配置相机内参的参数化方式。
 *            根据优化选项和配置，决定哪些相机内参保持常量、哪些参与优化。
 *            支持对焦距、主点、畸变参数等进行独立控制。
 * @param options：光束法平差选项，控制是否优化各类内参。
 * @param config：光束法平差配置，指定哪些相机内参是常量。
 * @param camera_ids：需要参数化的相机ID集合。
 * @param reconstruction：三维重建数据引用。
 * @param problem：Ceres优化问题引用。
 */
void ParameterizeCameras(const BundleAdjustmentOptions& options,
                         const BundleAdjustmentConfig& config,
                         const std::set<camera_t>& camera_ids,
                         Reconstruction& reconstruction,
                         ceres::Problem& problem) {
  // 判断是否所有相机内参都保持常量
  // 当焦距、主点、额外参数都不优化时，整个相机内参向量为常量
  const bool constant_camera = !options.refine_focal_length &&
                               !options.refine_principal_point &&
                               !options.refine_extra_params;

  // 遍历所有需要参数化的相机
  for (const camera_t camera_id : camera_ids) {
    Camera& camera = reconstruction.Camera(camera_id);

    if (constant_camera || config.HasConstantCamIntrinsics(camera_id)) {
      // 情况1：整个相机内参向量保持常量
      // 条件：全局设置不优化任何内参 或 该相机被显式配置为常量
      problem.SetParameterBlockConstant(camera.params.data());
    } else {
      // 情况2：部分内参可变，需要使用子集流形（Subset Manifold）
      // 收集需要保持常量的参数索引
      std::vector<int> const_camera_params;

      // 如果不优化焦距，将焦距参数索引加入常量列表
      // 注意：不同相机模型的焦距参数数量可能不同（如fx, fy）
      if (!options.refine_focal_length) {
        const span<const size_t> params_idxs = camera.FocalLengthIdxs();
        const_camera_params.insert(
            const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }

      // 如果不优化主点，将主点参数索引加入常量列表
      // 主点通常为 (cx, cy)
      if (!options.refine_principal_point) {
        const span<const size_t> params_idxs = camera.PrincipalPointIdxs();
        const_camera_params.insert(
            const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }

      // 如果不优化额外参数，将额外参数索引加入常量列表
      // 额外参数包括畸变系数（如k1, k2, p1, p2等）
      if (!options.refine_extra_params) {
        const span<const size_t> params_idxs = camera.ExtraParamsIdxs();
        const_camera_params.insert(
            const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }

      // 如果存在需要保持常量的参数，设置子集流形
      // 子集流形允许参数向量中的部分元素固定，其余元素参与优化
      if (const_camera_params.size() > 0) {
        SetSubsetManifold(static_cast<int>(camera.params.size()),
                          const_camera_params,
                          &problem,
                          camera.params.data());
      }
    }
  }
}

/**
 * [功能描述]：通过固定三个3D点来消除光束法平差的规范自由度（Gauge Freedom）。
 *            光束法平差存在7个自由度（3平移+3旋转+1尺度），需要通过约束来消除。
 *            该结构体选择三个线性无关的点进行固定，从而固定重建的位置、朝向和尺度。
 */
struct FixedGaugeWithThreePoints {
  // 已固定的点数量，最多为3个
  Eigen::Index num_fixed_points = 0;
  // 存储已固定点的坐标，每列为一个3D点坐标
  // 矩阵形状：3x3，第i列存储第i个固定点的(x,y,z)坐标
  Eigen::Matrix3d fixed_points = Eigen::Matrix3d::Zero();

  /**
   * [功能描述]：尝试添加一个固定点。
   *            只有当该点与已有固定点线性无关时才会被添加，
   *            以确保三个点能够完全约束7自由度中的6个（位置和朝向）。
   * @param point：候选的3D点坐标，形状为(3,)的向量。
   * @return 如果点被成功添加返回true，否则返回false。
   *         返回false的情况：已有3个固定点，或该点与已有点线性相关。
   */
  bool MaybeAddFixedPoint(const Eigen::Vector3d& point) {
    // 已经有3个固定点，无法再添加
    if (num_fixed_points >= 3) {
      return false;
    }

    // 将候选点暂时放入矩阵的对应列
    fixed_points.col(num_fixed_points) = point;

    // 使用列主元Householder QR分解计算矩阵的秩
    // 如果秩增加了，说明新点与已有点线性无关，可以接受
    if (fixed_points.colPivHouseholderQr().rank() > num_fixed_points) {
      ++num_fixed_points;
      return true;
    } else {
      // 新点与已有点线性相关（共线或共面），拒绝该点
      // 清空该列，等待下一个候选点
      fixed_points.col(num_fixed_points).setZero();
      return false;
    }
  }
};

/**
 * [功能描述]：通过固定三个线性无关的3D点来消除光束法平差的规范自由度。
 *            该函数分两步执行：
 *            1. 首先检查问题中是否已有足够的常量点可用于固定规范；
 *            2. 若不足，则主动将一些可变点设为常量。
 * @param point3D_num_observations：3D点ID到其观测数量的映射。
 * @param reconstruction：三维重建数据引用。
 * @param problem：Ceres优化问题引用。
 */
void FixGaugeWithThreePoints(
    const std::unordered_map<point3D_t, size_t>& point3D_num_observations,
    Reconstruction& reconstruction,
    ceres::Problem& problem) {
  // 创建规范固定器，用于跟踪已选择的固定点
  FixedGaugeWithThreePoints fixed_gauge;

  // ========== 第一步：检查已有的常量点是否足够 ==========
  // 遍历所有3D点，查找已经被设为常量的点
  // 如果已有3个线性无关的常量点，则无需额外操作
  for (const auto& [point3D_id, num_observations] : point3D_num_observations) {
    const Point3D& point3D = reconstruction.Point3D(point3D_id);
    // 检查该点是否已是常量，并尝试添加为固定点
    if (problem.IsParameterBlockConstant(point3D.xyz.data()) &&
        fixed_gauge.MaybeAddFixedPoint(point3D.xyz) &&
        fixed_gauge.num_fixed_points >= 3) {
      // 已找到3个线性无关的常量点，规范已固定，直接返回
      return;
    }
  }

  // ========== 第二步：主动固定额外的点 ==========
  // 如果已有常量点不足，需要将一些可变点设为常量
  for (const auto& [point3D_id, num_observations] : point3D_num_observations) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    // 检查该点是否为可变点，并尝试添加为固定点
    if (!problem.IsParameterBlockConstant(point3D.xyz.data()) &&
        fixed_gauge.MaybeAddFixedPoint(point3D.xyz)) {
      // 该点线性无关，将其设为常量
      problem.SetParameterBlockConstant(point3D.xyz.data());
      if (fixed_gauge.num_fixed_points >= 3) {
        // 已成功固定3个点，规范已固定，直接返回
        return;
      }
    }
  }

  // 警告：未能找到足够的线性无关点来固定规范自由度
  // 这可能发生在点数过少或所有点共线/共面的情况下
  LOG(WARNING)
      << "Failed to fix Gauge due to insufficient number of fixed points: "
      << fixed_gauge.num_fixed_points;
}

/**
 * [功能描述]：通过固定两个相机的位姿来消除光束法平差的规范自由度。
 *            固定第一个相机的完整位姿（6自由度：3旋转+3平移），
 *            固定第二个相机平移向量中基线最大的那个分量（1自由度），
 *            共消除7自由度（3平移+3旋转+1尺度）。
 *
 * 注意：该实现未处理所有退化边缘情况，例如：
 *   - 选中的两个相机之间没有足够的共视观测约束；
 *   - 对于多相机组（rig），可以更智能地在组内选择相机对。
 *
 * @param options：光束法平差选项。
 * @param config：光束法平差配置。
 * @param image_ids：参与优化的图像ID集合。
 * @param point3D_num_observations：3D点观测数量映射（用于回退方案）。
 * @param reconstruction：三维重建数据引用。
 * @param problem：Ceres优化问题引用。
 */
void FixGaugeWithTwoCamsFromWorld(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    const std::set<image_t>& image_ids,
    const std::unordered_map<point3D_t, size_t>& point3D_num_observations,
    Reconstruction& reconstruction,
    ceres::Problem& problem) {
  // 如果所有帧的位姿都不需要优化（都是常量），则无需固定规范
  if (!options.refine_rig_from_world) {
    return;
  }

  // 用于存储选中的两个图像指针
  Image* image1 = nullptr;  // 第一个相机（将固定完整位姿）
  Image* image2 = nullptr;  // 第二个相机（将固定平移的一个分量）

  // ========== 第一步：在已固定的相机中搜索 ==========
  // 优先使用已经被配置为常量位姿的相机
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    // 条件：该相机是rig的参考传感器 且 其帧位姿被配置为常量
    if (image.FramePtr()->RigPtr()->IsRefSensor(
            image.CameraPtr()->SensorId()) &&
        config.HasConstantRigFromWorldPose(image.FrameId())) {
      if (image1 == nullptr) {
        // 找到第一个常量位姿的相机
        image1 = &image;
      } else if (image1 != nullptr && image1->FrameId() != image.FrameId()) {
        // 如果已有两个不同帧的常量位姿相机，规范已固定，无需额外操作
        return;
      }
    }
  }

  // 辅助lambda：检查图像是否为已参数化的参考传感器
  // 参考传感器是rig中用于定义rig坐标系的相机
  auto IsParameterizedRefSensor = [&problem](const Image& image) {
    return image.FramePtr()->RigPtr()->IsRefSensor(
               image.CameraPtr()->SensorId()) &&
           problem.HasParameterBlock(
               image.FramePtr()->RigFromWorld().translation.data());
  };

  // ========== 第二步：在可变相机中搜索 ==========
  // 如果已固定相机不足，需要从可变相机中选择
  Eigen::Index frame2_from_world_fixed_dim = 0;  // 记录第二相机要固定的平移分量索引
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    if (image1 == nullptr && IsParameterizedRefSensor(image)) {
      // 选择第一个可变相机作为image1
      image1 = &image;
    } else if (image1 != nullptr && image1->FrameId() != image.FrameId() &&
               IsParameterizedRefSensor(image)) {
      // 寻找与image1不同帧的第二个相机
      // 计算两相机之间的基线向量（相对平移）
      // baseline = T_1w * T_w2 的平移部分 = 相机2在相机1坐标系下的位置
      const Eigen::Vector3d baseline =
          (image1->FramePtr()->RigFromWorld() *
           Inverse(image.FramePtr()->RigFromWorld()))
              .translation;
      // 选择基线中绝对值最大的分量作为要固定的维度
      // 这样可以最好地约束尺度；如果基线太小，尺度约束会不稳定
      if (baseline.cwiseAbs().maxCoeff(&frame2_from_world_fixed_dim) > 1e-9) {
        image2 = &image;
        break;
      }
    }
  }

  // ========== 回退方案：使用三点固定规范 ==========
  // TODO(jsch): 可以考虑在同一帧或不同帧的相机之间回退。
  // 由于组合情况较多，这里简单地回退到三点固定方案。
  // 未来支持IMU或其他传感器时，应采用不同的规范固定策略。
  if (image1 == nullptr || image2 == nullptr) {
    LOG(WARNING) << "Failed to fix Gauge with two cameras. "
                    "Falling back to fixing Gauge with three points.";
    FixGaugeWithThreePoints(point3D_num_observations, reconstruction, problem);
    return;
  }

  // ========== 第三步：固定选中的两个相机 ==========
  // 固定第一个相机的完整位姿（旋转+平移，共6自由度）
  Rigid3d& frame1_from_world = image1->FramePtr()->RigFromWorld();
  if (!config.HasConstantRigFromWorldPose(image1->FrameId())) {
    // 将旋转四元数设为常量
    problem.SetParameterBlockConstant(
        frame1_from_world.rotation.coeffs().data());
    // 将平移向量设为常量
    problem.SetParameterBlockConstant(frame1_from_world.translation.data());
  }

  // 固定第二个相机平移向量中的一个分量（1自由度，用于约束尺度）
  Rigid3d& frame2_from_world = image2->FramePtr()->RigFromWorld();
  if (!config.HasConstantRigFromWorldPose(image2->FrameId())) {
    // 使用子集流形固定平移向量中基线最大的那个分量
    // 例如：如果x方向基线最大，则固定tx，让ty和tz可优化
    SetSubsetManifold(3,
                      {static_cast<int>(frame2_from_world_fixed_dim)},
                      &problem,
                      frame2_from_world.translation.data());
  }
}

void ParameterizeImages(const BundleAdjustmentOptions& options,
                        const BundleAdjustmentConfig& config,
                        const std::set<image_t>& image_ids,
                        Reconstruction& reconstruction,
                        ceres::Problem& problem) {
  std::unordered_set<rig_t> parameterized_rig_ids;
  std::unordered_set<sensor_t> parameterized_sensor_ids;
  std::unordered_set<frame_t> parameterized_frame_ids;
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    parameterized_rig_ids.insert(image.FramePtr()->RigId());

    // Parameterize sensor_from_rig.
    const sensor_t sensor_id = image.CameraPtr()->SensorId();
    const bool not_parameterized_before =
        parameterized_sensor_ids.insert(sensor_id).second;
    if (not_parameterized_before && !image.HasTrivialFrame()) {
      Rigid3d& sensor_from_rig =
          image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
      // CostFunction assumes unit quaternions.
      sensor_from_rig.rotation.normalize();
      if (problem.HasParameterBlock(sensor_from_rig.rotation.coeffs().data())) {
        SetQuaternionManifold(&problem,
                              sensor_from_rig.rotation.coeffs().data());
        if (!options.refine_sensor_from_rig ||
            config.HasConstantSensorFromRigPose(sensor_id)) {
          problem.SetParameterBlockConstant(
              sensor_from_rig.rotation.coeffs().data());
          problem.SetParameterBlockConstant(sensor_from_rig.translation.data());
        }
      }
    }

    // Parameterize rig_from_world.
    if (parameterized_frame_ids.insert(image.FrameId()).second) {
      Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();
      // CostFunction assumes unit quaternions.
      rig_from_world.rotation.normalize();
      if (problem.HasParameterBlock(rig_from_world.rotation.coeffs().data())) {
        SetQuaternionManifold(&problem,
                              rig_from_world.rotation.coeffs().data());
        if (!options.refine_rig_from_world ||
            config.HasConstantRigFromWorldPose(image.FrameId())) {
          problem.SetParameterBlockConstant(
              rig_from_world.rotation.coeffs().data());
          problem.SetParameterBlockConstant(rig_from_world.translation.data());
        }
      }
    }
  }

  // Set the rig poses as constant, if the reference sensor is not part of the
  // problem. Otherwise, the relative pose between the sensors is not well
  // constrained. Notice that this does not handle degenerate configurations and
  // assumes the observations in the problem constrain the relative poses
  // sufficiently.
  for (const rig_t rig_id : parameterized_rig_ids) {
    Rig& rig = reconstruction.Rig(rig_id);
    if (parameterized_sensor_ids.count(rig.RefSensorId()) != 0) {
      continue;
    }
    for (auto& [_, sensor_from_rig] : rig.Sensors()) {
      if (sensor_from_rig.has_value() &&
          problem.HasParameterBlock(sensor_from_rig->translation.data())) {
        problem.SetParameterBlockConstant(
            sensor_from_rig->rotation.coeffs().data());
        problem.SetParameterBlockConstant(sensor_from_rig->translation.data());
      }
    }
  }
}

void ParameterizePoints(
    const BundleAdjustmentConfig& config,
    const std::unordered_map<point3D_t, size_t>& point3D_num_observations,
    Reconstruction& reconstruction,
    ceres::Problem& problem) {
  for (const auto& [point3D_id, num_observations] : point3D_num_observations) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (point3D.track.Length() > num_observations) {
      problem.SetParameterBlockConstant(point3D.xyz.data());
    }
  }

  for (const point3D_t point3D_id : config.ConstantPoints()) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    problem.SetParameterBlockConstant(point3D.xyz.data());
  }
}

/**
 * [功能描述]：默认的光束法平差（Bundle Adjustment）优化器类。
 *            该类负责构建和求解非线性最小二乘优化问题，用于同时优化
 *            相机位姿、相机内参和3D点坐标，最小化重投影误差。
 *            继承自BundleAdjuster基类。
 */
class DefaultBundleAdjuster : public BundleAdjuster {
 public:
  /**
   * [功能描述]：构造函数，初始化光束法平差问题。
   *            设置优化问题、添加图像和3D点观测、配置参数化方式、处理规范约束。
   * @param options：光束法平差选项，控制优化行为（如损失函数、是否优化位姿等）。
   * @param config：光束法平差配置，指定参与优化的图像、点和约束条件。
   * @param reconstruction：三维重建数据，包含相机、图像、3D点等信息（引用传递，优化后会被修改）。
   */
  DefaultBundleAdjuster(BundleAdjustmentOptions options,
                        BundleAdjustmentConfig config,
                        Reconstruction& reconstruction)
      : BundleAdjuster(std::move(options), std::move(config)),
        // 根据配置选项创建损失函数（如Huber、Cauchy等鲁棒损失函数）
        loss_function_(std::unique_ptr<ceres::LossFunction>(
            options_.CreateLossFunction())) {
    // 配置Ceres优化问题选项
    ceres::Problem::Options problem_options;
    // 设置损失函数的所有权为外部管理，避免Problem析构时重复释放
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    // 创建Ceres优化问题实例
    problem_ = std::make_shared<ceres::Problem>(problem_options);

    // ========== 第一步：构建优化问题 ==========
    // 警告：AddPointsToProblem假设AddImageToProblem已被先调用，
    // 不要改变以下指令的顺序！
    // 首先添加所有配置中的图像到优化问题
    for (const image_t image_id : config_.Images()) {
      AddImageToProblem(image_id, reconstruction);
    }
    // 添加可变3D点（这些点的坐标会被优化）
    for (const auto point3D_id : config_.VariablePoints()) {
      AddPointToProblem(point3D_id, reconstruction);
    }
    // 添加常量3D点（这些点的坐标保持固定，但仍参与约束）
    for (const auto point3D_id : config_.ConstantPoints()) {
      AddPointToProblem(point3D_id, reconstruction);
    }

    // ========== 第二步：配置参数化方式 ==========
    // 对相机内参进行参数化（设置哪些内参是常量/可变的）
    ParameterizeCameras(options_,
                        config_,
                        parameterized_camera_ids_,
                        reconstruction,
                        *problem_);
    // 对图像位姿进行参数化（设置旋转使用四元数的流形约束等）
    ParameterizeImages(
        options_, config_, parameterized_image_ids_, reconstruction, *problem_);
    // 对3D点进行参数化（设置常量点约束）
    ParameterizePoints(
        config_, point3D_num_observations_, reconstruction, *problem_);

    // ========== 第三步：处理规范自由度（Gauge）约束 ==========
    // 光束法平差存在7个自由度（3平移+3旋转+1尺度），需要固定以避免退化
    switch (config_.FixedGauge()) {
      case BundleAdjustmentGauge::UNSPECIFIED:
        // 不指定规范约束，由用户自行处理
        break;
      case BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD:
        // 通过固定两个相机的位姿来固定规范自由度
        FixGaugeWithTwoCamsFromWorld(options_,
                                     config_,
                                     parameterized_image_ids_,
                                     point3D_num_observations_,
                                     reconstruction,
                                     *problem_);
        break;
      case BundleAdjustmentGauge::THREE_POINTS:
        // 通过固定三个3D点来固定规范自由度
        FixGaugeWithThreePoints(
            point3D_num_observations_, reconstruction, *problem_);
        break;
      default:
        LOG(FATAL) << "Unknown BundleAdjustmentGauge";
    }
  }

  /**
   * [功能描述]：执行光束法平差优化求解。
   * @return 返回Ceres求解器的汇总信息，包含迭代次数、收敛状态、最终代价等。
   */
  ceres::Solver::Summary Solve() override {
    ceres::Solver::Summary summary;
    // 如果没有残差项，直接返回空结果
    if (problem_->NumResiduals() == 0) {
      return summary;
    }

    // 根据配置和问题规模创建求解器选项
    const ceres::Solver::Options solver_options =
        options_.CreateSolverOptions(config_, *problem_);

    // 调用Ceres求解器进行优化
    ceres::Solve(solver_options, problem_.get(), &summary);

    // 如果开启了打印摘要或详细日志，输出优化报告
    if (options_.print_summary || VLOG_IS_ON(1)) {
      PrintSolverSummary(summary, "Bundle adjustment report");
    }

    return summary;
  }

  /**
   * [功能描述]：获取Ceres优化问题的共享指针。
   * @return 返回指向ceres::Problem的共享指针引用，允许外部访问和修改问题。
   */
  std::shared_ptr<ceres::Problem>& Problem() override { return problem_; }

  /**
   * [功能描述]：将指定图像添加到优化问题中。
   *            根据图像是否具有简单帧（Trivial Frame）选择不同的处理方式。
   * @param image_id：要添加的图像ID。
   * @param reconstruction：三维重建数据引用。
   */
  void AddImageToProblem(const image_t image_id,
                         Reconstruction& reconstruction) {
    // 获取图像引用
    Image& image = reconstruction.Image(image_id);

    // 根据帧类型选择处理方式
    if (image.HasTrivialFrame()) {
      // 简单帧：相机直接与世界坐标系关联（无rig结构）
      AddImageWithTrivialFrame(image, reconstruction);
    } else {
      // 非简单帧：相机是相机组（rig）的一部分
      AddImageWithNonTrivialFrame(image, reconstruction);
    }
  }

  /**
   * [功能描述]：添加具有简单帧的图像到优化问题。
   *            简单帧表示相机直接与世界坐标系关联，没有相机组（rig）结构。
   *            为图像中每个有3D点对应的2D观测添加重投影误差残差项。
   * @param image：图像对象引用。
   * @param reconstruction：三维重建数据引用。
   */
  void AddImageWithTrivialFrame(Image& image, Reconstruction& reconstruction) {
    // 获取相机对象引用
    Camera& camera = *image.CameraPtr();

    // 断言检查：确保图像确实是简单帧
    THROW_CHECK(image.HasTrivialFrame());
    // 获取相机从世界坐标系到相机坐标系的变换（cam_from_world = T_cw）
    Rigid3d& cam_from_world = image.FramePtr()->RigFromWorld();

    // 判断相机位姿是否保持常量（不参与优化）
    // 条件：不需要优化rig位姿 或 该帧被配置为常量位姿
    const bool constant_cam_from_world =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());

    // ========== 为每个观测添加重投影误差残差项 ==========
    size_t num_observations = 0;
    for (const Point2D& point2D : image.Points2D()) {
      // 跳过没有对应3D点的2D特征点
      if (!point2D.HasPoint3D()) {
        continue;
      }

      // 统计有效观测数量
      num_observations += 1;
      // 更新该3D点的观测计数
      point3D_num_observations_[point2D.point3D_id] += 1;

      // 获取对应的3D点
      Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      // 检查：3D点的轨迹长度必须大于1（至少被两张图像观测到）
      THROW_CHECK_GT(point3D.track.Length(), 1);

      if (constant_cam_from_world) {
        // 位姿固定情况：使用常量位姿的代价函数
        // 只优化3D点坐标和相机内参
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, cam_from_world),
            loss_function_.get(),
            point3D.xyz.data(),      // 3D点坐标（待优化）
            camera.params.data());   // 相机内参（待优化/常量）
      } else {
        // 位姿可变情况：同时优化位姿、3D点和内参
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorCostFunctor>(camera.model_id,
                                                             point2D.xy),
            loss_function_.get(),
            cam_from_world.rotation.coeffs().data(),   // 旋转四元数（待优化）
            cam_from_world.translation.data(),         // 平移向量（待优化）
            point3D.xyz.data(),                        // 3D点坐标（待优化）
            camera.params.data());                     // 相机内参（待优化/常量）
      }
    }

    // 如果该图像有有效观测，将其相机和图像ID加入已参数化集合
    if (num_observations > 0) {
      parameterized_camera_ids_.insert(image.CameraId());
      parameterized_image_ids_.insert(image.ImageId());
    }
  }

  /**
   * [功能描述]：添加具有非简单帧的图像到优化问题。
   *            非简单帧表示相机是相机组（rig）的一部分，
   *            涉及相机到rig的变换和rig到世界的变换两层变换。
   * @param image：图像对象引用。
   * @param reconstruction：三维重建数据引用。
   */
  void AddImageWithNonTrivialFrame(Image& image,
                                   Reconstruction& reconstruction) {
    // 获取相机和传感器ID
    Camera& camera = *image.CameraPtr();
    const sensor_t sensor_id = camera.SensorId();

    // 断言检查：确保图像不是简单帧
    THROW_CHECK(!image.HasTrivialFrame());
    // 获取相机到rig的变换（cam_from_rig = T_cr）
    Rigid3d& cam_from_rig =
        image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
    // 获取rig到世界的变换（rig_from_world = T_rw）
    Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();

    // 判断相机到rig的变换是否保持常量
    const bool constant_sensor_from_rig =
        !options_.refine_sensor_from_rig ||
        config_.HasConstantSensorFromRigPose(sensor_id);
    // 判断rig到世界的变换是否保持常量
    const bool constant_rig_from_world =
        !options_.refine_rig_from_world ||
        config_.HasConstantRigFromWorldPose(image.FrameId());

    // ========== 为每个观测添加重投影误差残差项 ==========
    size_t num_observations = 0;
    for (const Point2D& point2D : image.Points2D()) {
      // 跳过没有对应3D点的2D特征点
      if (!point2D.HasPoint3D()) {
        continue;
      }

      num_observations += 1;
      point3D_num_observations_[point2D.point3D_id] += 1;

      Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      THROW_CHECK_GT(point3D.track.Length(), 1);

      // 注意：!constant_sensor_from_rig && constant_rig_from_world 的情况
      // 比较罕见，因此没有为其设计专门的代价函数
      if (constant_sensor_from_rig && constant_rig_from_world) {
        // 情况1：相机到rig和rig到世界的变换都固定
        // 预计算组合变换 T_cw = T_cr * T_rw
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, cam_from_rig * rig_from_world),
            loss_function_.get(),
            point3D.xyz.data(),
            camera.params.data());
      } else if (!constant_rig_from_world && constant_sensor_from_rig) {
        // 情况2：rig到世界可变，相机到rig固定
        // 优化rig位姿，相机到rig变换作为常量传入
        problem_->AddResidualBlock(
            CreateCameraCostFunction<RigReprojErrorConstantRigCostFunctor>(
                camera.model_id, point2D.xy, cam_from_rig),
            loss_function_.get(),
            rig_from_world.rotation.coeffs().data(),   // rig旋转（待优化）
            rig_from_world.translation.data(),         // rig平移（待优化）
            point3D.xyz.data(),
            camera.params.data());
      } else {
        // 情况3：相机到rig和/或rig到世界都可变
        // 同时优化两层变换
        problem_->AddResidualBlock(
            CreateCameraCostFunction<RigReprojErrorCostFunctor>(camera.model_id,
                                                                point2D.xy),
            loss_function_.get(),
            cam_from_rig.rotation.coeffs().data(),     // 相机到rig旋转（待优化）
            cam_from_rig.translation.data(),           // 相机到rig平移（待优化）
            rig_from_world.rotation.coeffs().data(),   // rig到世界旋转（待优化）
            rig_from_world.translation.data(),         // rig到世界平移（待优化）
            point3D.xyz.data(),
            camera.params.data());
      }
    }

    // 如果该图像有有效观测，将其相机和图像ID加入已参数化集合
    if (num_observations > 0) {
      parameterized_camera_ids_.insert(image.CameraId());
      parameterized_image_ids_.insert(image.ImageId());
    }
  }

  /**
   * [功能描述]：将指定的3D点添加到优化问题中。
   *            为该3D点在所有观测图像中添加重投影误差约束（对于尚未添加的图像）。
   *            这些额外图像的位姿被视为常量，不参与优化。
   * @param point3D_id：要添加的3D点ID。
   * @param reconstruction：三维重建数据引用。
   */
  void AddPointToProblem(const point3D_t point3D_id,
                         Reconstruction& reconstruction) {
    // 获取3D点引用
    Point3D& point3D = reconstruction.Point3D(point3D_id);

    // 获取该3D点已添加的观测数量的引用
    size_t& num_observations = point3D_num_observations_[point3D_id];

    // 检查3D点是否已完全包含在问题中
    // 即其所有观测都已通过 variable_image_ids、constant_image_ids、
    // constant_x_image_ids 添加
    if (num_observations == point3D.track.Length()) {
      return;
    }

    // 遍历该3D点的所有观测轨迹元素
    for (const auto& track_el : point3D.track.Elements()) {
      // 跳过已在FillImages中添加过的观测
      if (config_.HasImage(track_el.image_id)) {
        continue;
      }

      num_observations += 1;

      // 获取观测该点的图像、相机和2D点
      Image& image = reconstruction.Image(track_el.image_id);
      Camera& camera = *image.CameraPtr();
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);

      if (image.HasTrivialFrame()) {
        // 简单帧情况：直接使用相机到世界的变换
        Rigid3d& cam_from_world = image.FramePtr()->RigFromWorld();

        // 添加常量位姿的重投影误差（这些图像的位姿不参与优化）
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, cam_from_world),
            loss_function_.get(),
            point3D.xyz.data(),
            camera.params.data());
      } else {
        // 非简单帧情况：需要组合相机到rig和rig到世界的变换
        Rigid3d& cam_from_rig = image.FramePtr()->RigPtr()->SensorFromRig(
            image.CameraPtr()->SensorId());
        Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();

        // 使用组合变换 T_cw = T_cr * T_rw 作为常量
        problem_->AddResidualBlock(
            CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor>(
                camera.model_id, point2D.xy, cam_from_rig * rig_from_world),
            loss_function_.get(),
            point3D.xyz.data(),
            camera.params.data());
      }

      // 如果该相机是新添加的，将其内参设置为常量
      // 原因：对应图像未显式包含在配置中，不应优化其内参
      if (parameterized_camera_ids_.insert(image.CameraId()).second) {
        config_.SetConstantCamIntrinsics(image.CameraId());
      }
    }
  }

 private:
  // Ceres优化问题对象，存储所有残差项和参数块
  std::shared_ptr<ceres::Problem> problem_;
  // 鲁棒损失函数，用于减少外点对优化的影响
  std::unique_ptr<ceres::LossFunction> loss_function_;

  // 已添加到问题中的相机ID集合，用于跟踪哪些相机已被参数化
  std::set<camera_t> parameterized_camera_ids_;
  // 已添加到问题中的图像ID集合，用于跟踪哪些图像已被参数化
  std::set<image_t> parameterized_image_ids_;
  // 3D点ID到其观测数量的映射，用于跟踪每个3D点被添加了多少观测
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;
};

/**
 * [功能描述]：带位姿先验的光束法平差优化器。
 *            利用GPS/GNSS等外部位置先验信息约束重建，使重建结果具有真实尺度和地理坐标。
 *            继承自BundleAdjuster基类。
 */
class PosePriorBundleAdjuster : public BundleAdjuster {
 public:
  /**
   * [功能描述]：构造函数，初始化带位姿先验的BA优化器。
   * @param options：基本BA配置选项。
   * @param prior_options：位姿先验相关的配置选项。
   * @param config：BA配置，指定哪些图像/相机参与优化。
   * @param pose_priors：图像ID到位姿先验的映射表（通常来自GPS/GNSS数据）。
   * @param reconstruction：待优化的重建对象引用。
   */
  PosePriorBundleAdjuster(BundleAdjustmentOptions options,
                          PosePriorBundleAdjustmentOptions prior_options,
                          BundleAdjustmentConfig config,
                          std::unordered_map<image_t, PosePrior> pose_priors,
                          Reconstruction& reconstruction)
      : BundleAdjuster(std::move(options), std::move(config)),
        prior_options_(prior_options),
        pose_priors_(std::move(pose_priors)),
        reconstruction_(reconstruction) {
    // 尝试将重建对齐到位姿先验坐标系
    const bool use_prior_position = AlignReconstruction();

    // 根据是否有足够的有效位姿先验来决定如何固定BA问题的7自由度
    if (use_prior_position) {
      // 有位姿先验：对重建进行归一化以避免数值不稳定
      // 注意：不变换先验本身，先验会在添加到ceres::Problem时进行变换
      normalized_from_metric_ = reconstruction_.Normalize(/*fixed_scale=*/true);
    } else {
      // 无位姿先验：通过固定三个3D点来固定尺度规范
      config_.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
    }

    // 创建默认BA优化器用于处理基本的重投影误差
    default_bundle_adjuster_ = std::make_unique<DefaultBundleAdjuster>(
        options_, config_, reconstruction);

    // 如果使用位姿先验，将先验约束添加到优化问题中
    if (use_prior_position) {
      // 如果配置了鲁棒损失函数，使用Cauchy损失减少离群值影响
      if (prior_options_.use_robust_loss_on_prior_position) {
        prior_loss_function_ = std::make_unique<ceres::CauchyLoss>(
            prior_options_.prior_position_loss_scale);
      }

      // 遍历所有参与优化的图像，添加对应的位姿先验约束
      for (const image_t image_id : config_.Images()) {
        const auto pose_prior_it = pose_priors_.find(image_id);
        if (pose_prior_it != pose_priors_.end()) {
          AddPosePriorToProblem(
              image_id, pose_prior_it->second, reconstruction);
        }
      }
    }
  }

  /**
   * [功能描述]：执行BA优化求解。
   * @return ceres求解器的摘要信息，包含收敛状态、迭代次数等。
   */
  ceres::Solver::Summary Solve() override {
    ceres::Solver::Summary summary;
    std::shared_ptr<ceres::Problem> problem =
        default_bundle_adjuster_->Problem();

    // 如果没有残差项，直接返回空摘要
    if (problem->NumResiduals() == 0) {
      return summary;
    }

    // 创建求解器选项并执行优化
    const ceres::Solver::Options solver_options =
        options_.CreateSolverOptions(config_, *problem);

    ceres::Solve(solver_options, problem.get(), &summary);

    // 将重建从归一化坐标系变换回真实尺度坐标系
    reconstruction_.Transform(Inverse(normalized_from_metric_));

    // 根据配置输出优化摘要
    if (options_.print_summary || VLOG_IS_ON(1)) {
      PrintSolverSummary(summary, "Pose Prior Bundle adjustment report");
    }

    return summary;
  }

  /**
   * [功能描述]：获取ceres优化问题对象的引用。
   * @return 指向ceres::Problem的共享指针引用。
   */
  std::shared_ptr<ceres::Problem>& Problem() override {
    return default_bundle_adjuster_->Problem();
  }

  /**
   * [功能描述]：将单个图像的位姿先验约束添加到优化问题中。
   * @param image_id：图像ID。
   * @param prior：该图像的位姿先验信息（包含位置和协方差）。
   * @param reconstruction：重建对象引用。
   */
  void AddPosePriorToProblem(image_t image_id,
                             const PosePrior& prior,
                             Reconstruction& reconstruction) {
    // 验证先验数据的有效性
    if (!prior.IsValid() || !prior.IsCovarianceValid()) {
      LOG(ERROR) << "Could not add prior for image #" << image_id;
      return;
    }

    Image& image = reconstruction.Image(image_id);
    // 仅对简单帧（非多相机Rig）的图像施加先验约束
    if (!image.HasTrivialFrame()) {
      // TODO(jsch): 仅在参考传感器上施加位姿先验。
      // 如果只有非参考传感器图像有对应的位姿先验，当前实现会失败。
      // 未来将用专门的GNSS/GPS传感器建模来替代。
      return;
    }

    THROW_CHECK(image.HasPose());
    // 获取相机位姿的引用（从世界坐标系到相机坐标系的刚体变换）
    Rigid3d& cam_from_world = image.FramePtr()->RigFromWorld();

    std::shared_ptr<ceres::Problem>& problem =
        default_bundle_adjuster_->Problem();

    // 获取平移参数块指针
    double* cam_from_world_translation = cam_from_world.translation.data();
    // 如果该参数块不在优化问题中，跳过
    if (!problem->HasParameterBlock(cam_from_world_translation)) {
      return;
    }

    // 获取旋转参数块指针（四元数已在AddImageToProblem()中归一化）
    double* cam_from_world_rotation = cam_from_world.rotation.coeffs().data();

    // 添加位姿先验残差块：使用协方差加权的绝对位置先验代价函数
    problem->AddResidualBlock(
        CovarianceWeightedCostFunctor<AbsolutePosePositionPriorCostFunctor>::
            Create(prior.position_covariance,
                   normalized_from_metric_ * prior.position),  // 将先验位置变换到归一化坐标系
        prior_loss_function_.get(),  // 可选的鲁棒损失函数
        cam_from_world_rotation,
        cam_from_world_translation);
  }

  /**
   * [功能描述]：将重建对齐到位姿先验坐标系（通常是GPS/地理坐标系）。
   *            使用RANSAC鲁棒估计相似变换（Sim3）。
   * @return true表示对齐成功，false表示对齐失败（无足够有效先验）。
   */
  bool AlignReconstruction() {
    RANSACOptions ransac_options = prior_options_.alignment_ransac_options;

    // 如果未指定最大误差阈值，根据先验协方差自动计算
    if (ransac_options.max_error <= 0) {
      double max_stddev_sum = 0;
      size_t num_valid_covs = 0;

      // 遍历所有位姿先验，计算协方差最大特征值的平方根（最大标准差）
      for (const auto& [_, pose_prior] : pose_priors_) {
        if (pose_prior.IsCovarianceValid()) {
          // 计算协方差矩阵的最大特征值，取平方根得到最大标准差
          const double max_stddev =
              std::sqrt(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(
                            pose_prior.position_covariance)
                            .eigenvalues()
                            .maxCoeff());
          max_stddev_sum += max_stddev;
          ++num_valid_covs;
        }
      }

      // 如果没有有效协方差，无法进行对齐
      if (num_valid_covs == 0) {
        LOG(WARNING) << "No pose priors with valid covariance found.";
        return false;
      }

      // 使用3σ置信区间作为最大误差阈值（假设无离群值）
      ransac_options.max_error = 3 * max_stddev_sum / num_valid_covs;
    }

    VLOG(2) << "Robustly aligning reconstruction with max_error="
            << ransac_options.max_error;

    // 使用RANSAC鲁棒估计从原始坐标系到真实尺度坐标系的Sim3变换
    Sim3d metric_from_orig;
    const bool success = AlignReconstructionToPosePriors(
        reconstruction_, pose_priors_, ransac_options, &metric_from_orig);

    // 如果对齐成功，应用变换
    if (success) {
      reconstruction_.Transform(metric_from_orig);
    } else {
      LOG(WARNING) << "Alignment w.r.t. prior positions failed";
    }

    // 详细日志：输出对齐误差统计
    if (VLOG_IS_ON(2) && success) {
      std::vector<double> verr2_wrt_prior;
      verr2_wrt_prior.reserve(reconstruction_.NumRegImages());

      // 计算每张图像相对于先验位置的误差平方
      for (const image_t image_id : reconstruction_.RegImageIds()) {
        const auto pose_prior_it = pose_priors_.find(image_id);
        if (pose_prior_it != pose_priors_.end() &&
            pose_prior_it->second.IsValid()) {
          const auto& image = reconstruction_.Image(image_id);
          verr2_wrt_prior.push_back(
              (image.ProjectionCenter() - pose_prior_it->second.position)
                  .squaredNorm());
        }
      }

      // 输出RMSE和中位数误差
      VLOG(2) << "Alignment error w.r.t. prior positions:\n"
              << "  - rmse:   " << std::sqrt(Mean(verr2_wrt_prior)) << '\n'
              << "  - median: " << std::sqrt(Median(verr2_wrt_prior)) << '\n';
    }

    return success;
  }

 private:
  // 位姿先验BA配置选项
  const PosePriorBundleAdjustmentOptions prior_options_;
  // 图像ID到位姿先验的映射表
  const std::unordered_map<image_t, PosePrior> pose_priors_;
  // 待优化的重建对象引用
  Reconstruction& reconstruction_;

  // 默认BA优化器（处理基本的重投影误差）
  std::unique_ptr<DefaultBundleAdjuster> default_bundle_adjuster_;
  // 位姿先验的鲁棒损失函数（可选，用于抑制离群值）
  std::unique_ptr<ceres::LossFunction> prior_loss_function_;

  // 从真实尺度坐标系到归一化坐标系的Sim3变换
  Sim3d normalized_from_metric_;
};

}  // namespace

std::unique_ptr<BundleAdjuster> CreateDefaultBundleAdjuster(
    BundleAdjustmentOptions options,
    BundleAdjustmentConfig config,
    Reconstruction& reconstruction) {
  return std::make_unique<DefaultBundleAdjuster>(
      std::move(options), std::move(config), reconstruction);
}

std::unique_ptr<BundleAdjuster> CreatePosePriorBundleAdjuster(
    BundleAdjustmentOptions options,
    PosePriorBundleAdjustmentOptions prior_options,
    BundleAdjustmentConfig config,
    std::unordered_map<image_t, PosePrior> pose_priors,
    Reconstruction& reconstruction) {
  return std::make_unique<PosePriorBundleAdjuster>(std::move(options),
                                                   prior_options,
                                                   std::move(config),
                                                   std::move(pose_priors),
                                                   reconstruction);
}

void PrintSolverSummary(const ceres::Solver::Summary& summary,
                        const std::string& header) {
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << summary.FullReport();
  }

  std::ostringstream log;
  log << header << '\n';
  log << std::right << std::setw(16) << "Residuals : ";
  log << std::left << summary.num_residuals_reduced << '\n';

  log << std::right << std::setw(16) << "Parameters : ";
  log << std::left << summary.num_effective_parameters_reduced << '\n';

  log << std::right << std::setw(16) << "Iterations : ";
  log << std::left
      << summary.num_successful_steps + summary.num_unsuccessful_steps << '\n';

  log << std::right << std::setw(16) << "Time : ";
  log << std::left << summary.total_time_in_seconds << " [s]\n";

  log << std::right << std::setw(16) << "Initial cost : ";
  log << std::right << std::setprecision(6)
      << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
      << " [px]\n";

  log << std::right << std::setw(16) << "Final cost : ";
  log << std::right << std::setprecision(6)
      << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
      << " [px]\n";

  log << std::right << std::setw(16) << "Termination : ";

  std::string termination = "";

  switch (summary.termination_type) {
    case ceres::CONVERGENCE:
      termination = "Convergence";
      break;
    case ceres::NO_CONVERGENCE:
      termination = "No convergence";
      break;
    case ceres::FAILURE:
      termination = "Failure";
      break;
    case ceres::USER_SUCCESS:
      termination = "User success";
      break;
    case ceres::USER_FAILURE:
      termination = "User failure";
      break;
    default:
      termination = "Unknown";
      break;
  }

  log << std::right << termination << "\n\n";
  LOG(INFO) << log.str();
}

}  // namespace colmap
