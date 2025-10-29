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

#include "colmap/controllers/incremental_pipeline.h"

#include "colmap/estimators/alignment.h"
#include "colmap/scene/database.h"
#include "colmap/util/file.h"
#include "colmap/util/timer.h"

namespace colmap {
namespace {

void IterativeGlobalRefinement(const IncrementalPipelineOptions& options,
                               const IncrementalMapper::Options& mapper_options,
                               IncrementalMapper& mapper) {
  LOG(INFO) << "Retriangulation and Global bundle adjustment";
  mapper.IterativeGlobalRefinement(options.ba_global_max_refinements,
                                   options.ba_global_max_refinement_change,
                                   mapper_options,
                                   options.GlobalBundleAdjustment(),
                                   options.Triangulation());
  mapper.FilterFrames(mapper_options);
}

void ExtractColors(const std::string& image_path,
                   const image_t image_id,
                   Reconstruction& reconstruction) {
  if (!reconstruction.ExtractColorsForImage(image_id, image_path)) {
    LOG(WARNING) << StringPrintf("Could not read image %s at path %s.",
                                 reconstruction.Image(image_id).Name().c_str(),
                                 image_path.c_str());
  }
}

void WriteSnapshot(const Reconstruction& reconstruction,
                   const std::string& snapshot_path) {
  LOG(INFO) << "Creating snapshot";
  // Get the current timestamp in milliseconds.
  const size_t timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  // Write reconstruction to unique path with current timestamp.
  const std::string path =
      JoinPaths(snapshot_path, StringPrintf("%010zu", timestamp));
  CreateDirIfNotExists(path);
  VLOG(1) << "=> Writing to " << path;
  reconstruction.Write(path);
}

}  // namespace

IncrementalMapper::Options IncrementalPipelineOptions::Mapper() const {
  IncrementalMapper::Options options = mapper;
  options.abs_pose_refine_focal_length = ba_refine_focal_length;
  options.abs_pose_refine_extra_params = ba_refine_extra_params;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  options.local_ba_num_images = ba_local_num_images;
  options.fix_existing_frames = fix_existing_frames;
  options.constant_rigs = constant_rigs;
  options.constant_cameras = constant_cameras;
  options.use_prior_position = use_prior_position;
  options.use_robust_loss_on_prior_position = use_robust_loss_on_prior_position;
  options.prior_position_loss_scale = prior_position_loss_scale;
  options.random_seed = random_seed;
  return options;
}

IncrementalTriangulator::Options IncrementalPipelineOptions::Triangulation()
    const {
  IncrementalTriangulator::Options options = triangulation;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.random_seed = random_seed;
  return options;
}

BundleAdjustmentOptions IncrementalPipelineOptions::LocalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = ba_local_function_tolerance;
  options.solver_options.gradient_tolerance = 10.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_local_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.logging_type = ceres::LoggingType::SILENT;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = false;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.refine_sensor_from_rig = ba_refine_sensor_from_rig;
  options.min_num_residuals_for_cpu_multi_threading =
      ba_min_num_residuals_for_cpu_multi_threading;
  options.loss_function_scale = 1.0;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  options.use_gpu = ba_use_gpu;
  options.gpu_index = ba_gpu_index;
  return options;
}

BundleAdjustmentOptions IncrementalPipelineOptions::GlobalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = ba_global_function_tolerance;
  options.solver_options.gradient_tolerance = 1.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_global_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.logging_type = ceres::LoggingType::SILENT;
  if (VLOG_IS_ON(2)) {
    options.solver_options.minimizer_progress_to_stdout = true;
    options.solver_options.logging_type =
        ceres::LoggingType::PER_MINIMIZER_ITERATION;
  }
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = false;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.refine_sensor_from_rig = ba_refine_sensor_from_rig;
  options.min_num_residuals_for_cpu_multi_threading =
      ba_min_num_residuals_for_cpu_multi_threading;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  options.use_gpu = ba_use_gpu;
  options.gpu_index = ba_gpu_index;
  return options;
}

bool IncrementalPipelineOptions::Check() const {
  CHECK_OPTION_GT(min_num_matches, 0);
  CHECK_OPTION_GT(max_num_models, 0);
  CHECK_OPTION_GT(max_model_overlap, 0);
  CHECK_OPTION_GE(min_model_size, 0);
  CHECK_OPTION_GT(init_num_trials, 0);
  CHECK_OPTION_GT(min_focal_length_ratio, 0);
  CHECK_OPTION_GT(max_focal_length_ratio, 0);
  CHECK_OPTION_GE(max_extra_param, 0);
  CHECK_OPTION_GE(ba_local_num_images, 2);
  CHECK_OPTION_GE(ba_local_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_global_frames_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_frames_freq, 0);
  CHECK_OPTION_GT(ba_global_points_freq, 0);
  CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_local_max_refinements, 0);
  CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
  CHECK_OPTION_GT(ba_global_max_refinements, 0);
  CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
  CHECK_OPTION_GE(snapshot_frames_freq, 0);
  CHECK_OPTION_GT(prior_position_loss_scale, 0.);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_GE(random_seed, -1);
  CHECK_OPTION(Mapper().Check());
  CHECK_OPTION(Triangulation().Check());
  return true;
}

IncrementalPipeline::IncrementalPipeline(
    std::shared_ptr<const IncrementalPipelineOptions> options,
    const std::string& image_path,
    const std::string& database_path,
    std::shared_ptr<class ReconstructionManager> reconstruction_manager)
    : options_(std::move(options)),
      image_path_(image_path),
      database_path_(database_path),
      reconstruction_manager_(std::move(reconstruction_manager)) {
  THROW_CHECK(options_->Check());
  RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
  RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
  RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

/**
 * [功能描述]：增量式重建管道的主运行函数，执行完整的Structure-from-Motion重建流程
 * @return [返回值说明]：无返回值，通过异常或日志输出处理错误
 */
void IncrementalPipeline::Run() {
  // 创建计时器，用于记录整个重建过程的耗时
  Timer run_timer;
  run_timer.Start();
  
  // 加载数据库，如果加载失败则直接返回
  if (!LoadDatabase()) {
    return;
  }

  // 检查是否在开始重建前已有子重建，即用户是否导入了现有的重建
  const bool continue_reconstruction = reconstruction_manager_->Size() > 0;
  // 确保只能从一个单一的重建继续，不能从多个重建继续
  THROW_CHECK_LE(reconstruction_manager_->Size(), 1)
      << "Can only continue from a single reconstruction, "
         "but multiple are given.";

  // 获取数据库中图像的总数量
  const size_t num_images = database_cache_->NumImages();

  // 创建增量式映射器并配置其选项
  IncrementalMapper::Options mapper_options = options_->Mapper();
  IncrementalMapper mapper(database_cache_);
  
  // 执行主要的重建过程
  Reconstruct(mapper,
              mapper_options,
              /*continue_reconstruction=*/continue_reconstruction);

  // 定义初始化约束松弛的次数
  const size_t kNumInitRelaxations = 2;
  
  // 进行多次初始化约束松弛，以提高图像注册的成功率
  for (size_t i = 0; i < kNumInitRelaxations; ++i) {
    // 如果所有图像都已注册或用户请求停止，则跳出循环
    if (mapper.NumTotalRegImages() == num_images || CheckIfStopped()) {
      break;
    }

    // 第一次松弛：降低最小内点数量要求
    LOG(INFO) << "=> Relaxing the initialization constraints.";
    mapper_options.init_min_num_inliers /= 2;  // 将最小内点数量减半
    mapper.ResetInitializationStats();  // 重置初始化统计信息
    // 重新执行重建，不继续现有重建
    Reconstruct(mapper, mapper_options, /*continue_reconstruction=*/false);

    // 再次检查是否所有图像都已注册或用户请求停止
    if (mapper.NumTotalRegImages() == num_images || CheckIfStopped()) {
      break;
    }

    // 第二次松弛：降低最小三角化角度要求
    LOG(INFO) << "=> Relaxing the initialization constraints.";
    mapper_options.init_min_tri_angle /= 2;  // 将最小三角化角度减半
    mapper.ResetInitializationStats();  // 重置初始化统计信息
    // 再次重新执行重建
    Reconstruct(mapper, mapper_options, /*continue_reconstruction=*/false);
  }

  // 打印整个重建过程的耗时（以分钟为单位）
  run_timer.PrintMinutes();
}

bool IncrementalPipeline::LoadDatabase() {
  LOG(INFO) << "Loading database";

  // Make sure images of the given reconstruction are also included when
  // manually specifying images for the reconstruction procedure.
  std::unordered_set<std::string> image_names = {options_->image_names.begin(),
                                                 options_->image_names.end()};
  if (reconstruction_manager_->Size() == 1 && !options_->image_names.empty()) {
    const auto& reconstruction = reconstruction_manager_->Get(0);
    for (const image_t image_id : reconstruction->RegImageIds()) {
      const auto& image = reconstruction->Image(image_id);
      image_names.insert(image.Name());
    }
  }

  Timer timer;
  timer.Start();
  database_cache_ = DatabaseCache::Create(
      *Database::Open(database_path_),
      /*min_num_matches=*/static_cast<size_t>(options_->min_num_matches),
      /*ignore_watermarks=*/options_->ignore_watermarks,
      /*image_names=*/image_names);
  timer.PrintMinutes();

  if (database_cache_->NumImages() == 0) {
    LOG(WARNING) << "No images with matches found in the database";
    return false;
  }

  // If prior positions are to be used and setup from the database, convert
  // geographic coords. to cartesian ones
  if (options_->use_prior_position) {
    return database_cache_->SetupPosePriors();
  }

  return true;
}

/**
 * [功能描述]：初始化三维重建过程，选择并注册初始图像对。
 *             该函数是增量式SfM的起点，负责找到或验证一对合适的初始图像，
 *             估计它们之间的相对位姿，三角化初始3D点，并执行全局优化。
 * 
 * @param mapper：增量映射器对象的引用，提供图像对选择、注册和优化等功能。
 * @param mapper_options：映射器的配置选项，控制初始化过程中的各种参数。
 * @param reconstruction：当前重建实例的引用，用于存储初始化后的数据。
 * @return 返回初始化状态（Status），包括SUCCESS（成功）、NO_INITIAL_PAIR（无初始图像对）
 *         或BAD_INITIAL_PAIR（初始图像对质量差）。
 */
IncrementalPipeline::Status IncrementalPipeline::InitializeReconstruction(
    IncrementalMapper& mapper,
    const IncrementalMapper::Options& mapper_options,
    Reconstruction& reconstruction) {
  // 获取初始图像对的ID（可能是用户指定的，也可能是默认的无效值）
  image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
  image_t image_id2 = static_cast<image_t>(options_->init_image_id2);

  // cam2_from_cam1：表示从相机1坐标系到相机2坐标系的刚体变换（旋转+平移）
  Rigid3d cam2_from_cam1;
  
  // 判断用户是否提供了初始图像对
  if (!options_->IsInitialPairProvided()) {
    // 情况1：用户未指定初始图像对，需要自动查找
    LOG(INFO) << "Finding good initial image pair";
    // 自动搜索最佳的初始图像对
    // 选择标准：特征匹配数量多、基线合适、重投影误差小等
    const bool find_init_success = mapper.FindInitialImagePair(
        mapper_options, image_id1, image_id2, cam2_from_cam1);
    if (!find_init_success) {
      // 如果找不到合适的初始图像对，返回失败状态
      LOG(INFO) << "=> No good initial image pair found.";
      return Status::NO_INITIAL_PAIR;
    }
  } else {
    // 情况2：用户指定了初始图像对，需要验证其有效性
    // 首先检查指定的图像是否存在于重建数据中
    if (!reconstruction.ExistsImage(image_id1) ||
        !reconstruction.ExistsImage(image_id2)) {
      LOG(INFO) << StringPrintf(
          "=> Initial image pair #%d and #%d does not exist.",
          image_id1,
          image_id2);
      return Status::NO_INITIAL_PAIR;
    }
    // 估计用户指定的两张图像之间的几何关系（本质矩阵或基础矩阵）
    const bool provided_init_success = mapper.EstimateInitialTwoViewGeometry(
        mapper_options, image_id1, image_id2, cam2_from_cam1);
    if (!provided_init_success) {
      // 如果用户指定的图像对质量不佳（匹配数量少、几何关系不稳定等），返回失败状态
      LOG(INFO) << "=> Provided pair is unsuitable for initialization.";
      return Status::BAD_INITIAL_PAIR;
    }
  }

  // 注册初始图像对：将两张图像及其相对位姿添加到重建中
  LOG(INFO) << StringPrintf(
      "Registering initial image pair #%d and #%d", image_id1, image_id2);
  mapper.RegisterInitialImagePair(
      mapper_options, image_id1, image_id2, cam2_from_cam1);

  // 三角化初始图像对观测到的3D点
  // 配置三角化选项，设置最小三角化角度以确保点的深度精度
  IncrementalTriangulator::Options tri_options;
  tri_options.min_angle = mapper_options.init_min_tri_angle;
  // 遍历两张初始图像
  for (const image_t image_id : {image_id1, image_id2}) {
    const Image& image = reconstruction.Image(image_id);
    // 对图像所属帧的所有图像进行三角化
    // （在多相机或视频序列中，一个帧可能包含多张图像）
    for (const data_t& data_id : image.FramePtr()->ImageIds()) {
      mapper.TriangulateImage(tri_options, data_id.id);
    }
  }

  // 执行全局光束法平差优化
  // 优化初始图像对的相机位姿和三角化得到的3D点，提高初始重建精度
  LOG(INFO) << "Global bundle adjustment";
  mapper.AdjustGlobalBundle(mapper_options, options_->GlobalBundleAdjustment());
  
  // 归一化重建结果：将重建坐标系规范化（通常是设置合适的尺度和原点）
  reconstruction.Normalize();
  
  // 过滤低质量的3D点（重投影误差大、观测数量少等）
  mapper.FilterPoints(mapper_options);
  
  // 过滤无效的帧（注册失败或点数量不足等）
  mapper.FilterFrames(mapper_options);

  // 验证初始化结果：检查初始图像对是否成功注册
  // 如果没有注册任何帧或没有三角化任何3D点，说明初始化失败
  if (reconstruction.NumRegFrames() == 0 || reconstruction.NumPoints3D() == 0) {
    return Status::BAD_INITIAL_PAIR;
  }

  // 验证三角化的3D点数量是否足够
  // 如果点数量少于后续图像注册所需的最小内点数量，初始化失败
  // 因为没有足够的3D点来通过PnP算法注册新图像
  if (static_cast<int>(reconstruction.NumPoints3D()) <
      mapper_options.abs_pose_min_num_inliers) {
    return Status::BAD_INITIAL_PAIR;
  }

  // 如果启用了颜色提取，为初始图像对的3D点提取颜色信息
  if (options_->extract_colors) {
    for (const image_t image_id : {image_id1, image_id2}) {
      const Image& image = reconstruction.Image(image_id);
      for (const data_t& data_id : image.FramePtr()->ImageIds()) {
        ExtractColors(image_path_, data_id.id, reconstruction);
      }
    }
  }
  
  // 初始化成功，返回成功状态
  return Status::SUCCESS;
}

bool IncrementalPipeline::CheckRunGlobalRefinement(
    const Reconstruction& reconstruction,
    const size_t ba_prev_num_reg_frames,
    const size_t ba_prev_num_points) {
  return reconstruction.NumRegFrames() >=
             options_->ba_global_frames_ratio * ba_prev_num_reg_frames ||
         reconstruction.NumRegFrames() >=
             options_->ba_global_frames_freq + ba_prev_num_reg_frames ||
         reconstruction.NumPoints3D() >=
             options_->ba_global_points_ratio * ba_prev_num_points ||
         reconstruction.NumPoints3D() >=
             options_->ba_global_points_freq + ba_prev_num_points;
}

/**
 * [功能描述]：执行子模型的增量式重建过程。
 *             该函数首先注册初始图像对，然后迭代地注册新图像、三角化3D点、
 *             执行局部和全局光束法平差优化，直到无法注册更多图像或满足停止条件。
 * 
 * @param mapper：增量映射器对象的引用，提供图像注册、三角化等核心功能。
 * @param mapper_options：映射器的配置选项，控制注册和优化的各种参数。
 * @param reconstruction：当前重建实例的共享指针，存储相机、图像和3D点等数据。
 * @return 返回重建状态（Status），包括SUCCESS（成功）、INTERRUPTED（中断）、
 *         NO_INITIAL_PAIR（无初始图像对）或BAD_INITIAL_PAIR（初始图像对质量差）。
 */
IncrementalPipeline::Status IncrementalPipeline::ReconstructSubModel(
    IncrementalMapper& mapper,
    const IncrementalMapper::Options& mapper_options,
    const std::shared_ptr<Reconstruction>& reconstruction) {
  // 开始新的重建过程
  mapper.BeginReconstruction(reconstruction);

  ////////////////////////////////////////////////////////////////////////////
  // 注册初始图像对
  ////////////////////////////////////////////////////////////////////////////

  // 如果当前重建中没有已注册的帧（即重建为空），则需要初始化
  if (reconstruction->NumRegFrames() == 0) {
    // 初始化重建：选择并注册第一对图像
    const Status init_status = IncrementalPipeline::InitializeReconstruction(
        mapper, mapper_options, *reconstruction);
    // 如果初始化失败（无合适图像对或图像对质量差），直接返回失败状态
    if (init_status != Status::SUCCESS) {
      return init_status;
    }
  }
  // 触发初始图像对注册完成的回调
  Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

  ////////////////////////////////////////////////////////////////////////////
  // 增量式映射主循环
  ////////////////////////////////////////////////////////////////////////////

  // 记录上一次保存快照时的已注册帧数
  size_t snapshot_prev_num_reg_frames = reconstruction->NumRegFrames();
  // 记录上一次执行全局BA时的已注册帧数
  size_t ba_prev_num_reg_frames = reconstruction->NumRegFrames();
  // 记录上一次执行全局BA时的3D点数量
  size_t ba_prev_num_points = reconstruction->NumPoints3D();

  // 当前轮是否成功注册了新图像
  bool reg_next_success = true;
  // 上一轮是否成功注册了新图像
  bool prev_reg_next_success = true;
  
  // 主循环：只要当前轮或上一轮成功注册了图像，就继续尝试
  // 这样即使一轮失败，也会再尝试一轮（在全局BA之后）
  do {
    // 检查是否接收到停止信号
    if (CheckIfStopped()) {
      break;
    }

    // 更新注册成功标志
    prev_reg_next_success = reg_next_success;
    reg_next_success = false;

    // 查找下一批待注册的候选图像（按优先级排序）
    const std::vector<image_t> next_images =
        mapper.FindNextImages(mapper_options);

    // 如果没有候选图像，退出主循环
    if (next_images.empty()) {
      break;
    }

    // 尝试注册候选图像列表中的图像
    image_t next_image_id;
    for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
      next_image_id = next_images[reg_trial];

      // 记录当前尝试注册的图像信息
      LOG(INFO) << StringPrintf("Registering image #%d (num_reg_frames=%d)",
                                next_image_id,
                                reconstruction->NumRegFrames());
      LOG(INFO) << StringPrintf(
          "=> Image sees %d / %d points",
          mapper.ObservationManager().NumVisiblePoints3D(next_image_id),
          mapper.ObservationManager().NumObservations(next_image_id));

      // 尝试注册该图像（估计其相机位姿）
      reg_next_success =
          mapper.RegisterNextImage(mapper_options, next_image_id);

      // 如果注册成功，跳出尝试循环
      if (reg_next_success) {
        break;
      } else {
        LOG(INFO) << "=> Could not register, trying another image.";

        // 如果初始图像对在一段时间内无法继续扩展，则放弃当前重建
        // 这可能意味着初始图像对选择不佳，应该尝试不同的初始图像对
        const size_t kMinNumInitialRegTrials = 30;
        if (reg_trial >= kMinNumInitialRegTrials &&
            reconstruction->NumRegFrames() <
                static_cast<size_t>(options_->min_model_size)) {
          break;
        }
      }
    }

    // 如果成功注册了新图像，执行后续处理步骤
    if (reg_next_success) {
      const Image& image = reconstruction->Image(next_image_id);
      
      // 步骤1：对新注册图像所属帧的所有图像进行三角化
      // 三角化：根据多视图观测计算新的3D点
      for (const data_t& data_id : image.FramePtr()->ImageIds()) {
        mapper.TriangulateImage(options_->Triangulation(), data_id.id);
      }
      
      // 步骤2：执行迭代式局部光束法平差优化
      // 优化新注册图像及其邻近图像的相机位姿和3D点
      mapper.IterativeLocalRefinement(options_->ba_local_max_refinements,
                                      options_->ba_local_max_refinement_change,
                                      mapper_options,
                                      options_->LocalBundleAdjustment(),
                                      options_->Triangulation(),
                                      next_image_id);

      // 步骤3：检查是否需要执行全局光束法平差优化
      // 全局BA会优化所有相机位姿和3D点，开销较大但精度更高
      if (CheckRunGlobalRefinement(
              *reconstruction, ba_prev_num_reg_frames, ba_prev_num_points)) {
        IterativeGlobalRefinement(*options_, mapper_options, mapper);
        // 更新全局BA后的统计信息
        ba_prev_num_points = reconstruction->NumPoints3D();
        ba_prev_num_reg_frames = reconstruction->NumRegFrames();
      }

      // 步骤4：如果启用了颜色提取，为3D点提取颜色信息
      if (options_->extract_colors) {
        for (const data_t& data_id : image.FramePtr()->ImageIds()) {
          ExtractColors(image_path_, data_id.id, *reconstruction);
        }
      }

      // 步骤5：如果启用了快照功能，定期保存重建结果
      // 当注册的帧数达到快照频率阈值时保存
      if (options_->snapshot_frames_freq > 0 &&
          reconstruction->NumRegFrames() >=
              options_->snapshot_frames_freq + snapshot_prev_num_reg_frames) {
        snapshot_prev_num_reg_frames = reconstruction->NumRegFrames();
        WriteSnapshot(*reconstruction, options_->snapshot_path);
      }

      // 触发新图像注册完成的回调
      Callback(NEXT_IMAGE_REG_CALLBACK);
    }

    // 检查模型重叠度：如果当前模型与其他模型的共享图像数量超过阈值，停止扩展
    // 这用于多模型重建场景，避免模型之间过度重叠
    const size_t max_model_overlap =
        static_cast<size_t>(options_->max_model_overlap);
    if (mapper.NumSharedRegImages() >= max_model_overlap) {
      break;
    }

    // 如果当前轮未能注册任何图像，但上一轮成功了，则尝试最后一次全局BA
    // 全局BA可能会改善重建质量，使得之前无法注册的图像变得可以注册
    // 如果这之后仍然无法注册，则退出主循环
    if (!reg_next_success && prev_reg_next_success) {
      IterativeGlobalRefinement(*options_, mapper_options, mapper);
    }
  } while (reg_next_success || prev_reg_next_success);

  // 如果在循环中检测到停止信号，返回中断状态
  if (CheckIfStopped()) {
    return Status::INTERRUPTED;
  }

  // 最终全局BA：如果重建不为空，且上一次增量BA不是全局BA，则执行最终的全局优化
  // 这确保最终重建结果达到最优精度
  if (reconstruction->NumRegFrames() > 0 &&
      reconstruction->NumRegFrames() != ba_prev_num_reg_frames &&
      reconstruction->NumPoints3D() != ba_prev_num_points) {
    IterativeGlobalRefinement(*options_, mapper_options, mapper);
  }
  
  // 返回成功状态
  return Status::SUCCESS;
}

/**
 * [功能描述]：执行增量式三维重建的主流程。
 *             该函数会尝试多次初始化重建，每次尝试都会构建一个子模型，
 *             根据重建状态决定是否保留、丢弃或继续尝试新的重建。
 * 
 * @param mapper：增量映射器对象的引用，负责执行实际的重建操作。
 * @param mapper_options：映射器的配置选项，包含重建过程中的各种参数设置。
 * @param continue_reconstruction：是否继续之前的重建（true表示继续，false表示从头开始）。
 * @return 无返回值。
 */
void IncrementalPipeline::Reconstruct(
    IncrementalMapper& mapper,
    const IncrementalMapper::Options& mapper_options,
    bool continue_reconstruction) {
  // 循环尝试多次重建初始化，最多尝试 init_num_trials 次
  for (int num_trials = 0; num_trials < options_->init_num_trials;
       ++num_trials) {
    // 检查是否接收到停止信号
    if (CheckIfStopped()) {
      break;
    }
    
    // 确定当前重建的索引
    size_t reconstruction_idx;
    if (!continue_reconstruction || num_trials > 0) {
      // 如果不是继续重建或不是第一次尝试，则创建新的重建实例
      reconstruction_idx = reconstruction_manager_->Add();
    } else {
      // 如果是继续重建且是第一次尝试，则使用索引0（已存在的重建）
      reconstruction_idx = 0;
    }
    // 获取当前重建实例的共享指针
    std::shared_ptr<Reconstruction> reconstruction =
        reconstruction_manager_->Get(reconstruction_idx);

    // 执行子模型重建，返回重建状态
    const Status status =
        ReconstructSubModel(mapper, mapper_options, reconstruction);
    
    // 根据重建状态进行不同的处理
    switch (status) {
      // 情况1：重建过程被中断（例如用户手动停止）
      case Status::INTERRUPTED: {
        // 更新所有3D点的重投影误差
        reconstruction->UpdatePoint3DErrors();
        LOG(INFO) << "Keeping reconstruction due to interrupt";
        // 结束重建但不丢弃结果（保留已完成的部分）
        mapper.EndReconstruction(/*discard=*/false);
        // 将重建结果对齐到原始相机设备（Rig）的尺度
        AlignReconstructionToOrigRigScales(database_cache_->Rigs(),
                                           reconstruction.get());
        return;
      }

      // 情况2：无法找到合适的初始图像对
      case Status::NO_INITIAL_PAIR: {
        LOG(INFO) << "Discarding reconstruction due to no initial pair";
        // 结束重建并丢弃结果
        mapper.EndReconstruction(/*discard=*/true);
        // 从重建管理器中删除该重建实例
        reconstruction_manager_->Delete(reconstruction_idx);
        // 如果找不到初始图像对，直接退出尝试循环
        // 因为除非放宽初始化阈值，否则下一次尝试也不会找到合适的图像对
        return;
      }

      // 情况3：找到了初始图像对但质量不佳
      case Status::BAD_INITIAL_PAIR: {
        LOG(INFO) << "Discarding reconstruction due to bad initial pair";
        // 结束重建并丢弃结果
        mapper.EndReconstruction(/*discard=*/true);
        // 从重建管理器中删除该重建实例
        reconstruction_manager_->Delete(reconstruction_idx);
        // 丢弃当前的初始图像对，在下一次尝试中会从剩余的图像对中选择
        // 继续循环进行下一次尝试
        break;
      }

      // 情况4：重建成功完成
      case Status::SUCCESS: {
        // 记录已注册的图像总数
        // 这个值在后续可能因模型尺寸不足而丢弃模型之前就需要记录下来
        // 用于判断是否所有图像都已被注册
        const size_t total_num_reg_images = mapper.NumTotalRegImages();

        // 计算最小模型尺寸阈值
        // 如果图像总数较少，则不强制要求最小模型尺寸，以便重建小型图像集合
        // 默认为数据库中图像数量的80%和配置的最小模型尺寸中的较小值
        const size_t min_model_size = std::min<size_t>(
            0.8 * database_cache_->NumImages(), options_->min_model_size);
        
        // 判断是否应该丢弃当前重建结果
        if ((options_->multiple_models && reconstruction_manager_->Size() > 1 &&
             reconstruction->NumRegFrames() < min_model_size) ||
            reconstruction->NumRegFrames() == 0) {
          // 以下情况会丢弃重建：
          // 1. 启用多模型模式 && 不是第一个重建 && 注册的帧数小于最小模型尺寸
          // 2. 或者注册的帧数为0（重建失败）
          LOG(INFO) << "Discarding reconstruction due to insufficient size";
          mapper.EndReconstruction(/*discard=*/true);
          reconstruction_manager_->Delete(reconstruction_idx);
        } else {
          // 保留当前重建结果
          // 更新所有3D点的重投影误差
          reconstruction->UpdatePoint3DErrors();
          LOG(INFO) << "Keeping successful reconstruction";
          // 结束重建但保留结果
          mapper.EndReconstruction(/*discard=*/false);
          // 将重建结果对齐到原始相机设备的尺度
          AlignReconstructionToOrigRigScales(database_cache_->Rigs(),
                                             reconstruction.get());
        }

        // 触发最后一张图像注册完成的回调
        Callback(LAST_IMAGE_REG_CALLBACK);

        // 判断是否应该结束整个重建流程
        if (!options_->multiple_models ||
            reconstruction_manager_->Size() >=
                static_cast<size_t>(options_->max_num_models) ||
            total_num_reg_images >= database_cache_->NumImages() - 1) {
          // 以下情况会结束重建流程：
          // 1. 未启用多模型模式
          // 2. 或者已达到最大模型数量限制
          // 3. 或者几乎所有图像都已被注册（至少N-1张）
          return;
        }

        // 如果需要构建多个模型且未达到退出条件，则继续下一次尝试
        break;
      }

      // 默认情况：未知的重建状态（不应该发生）
      default:
        LOG(FATAL_THROW) << "Unknown reconstruction status.";
    }
  }
}

void IncrementalPipeline::TriangulateReconstruction(
    const std::shared_ptr<Reconstruction>& reconstruction) {
  THROW_CHECK(LoadDatabase());
  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction);

  LOG(INFO) << "Iterative triangulation";
  size_t image_idx = 0;
  for (const image_t image_id : reconstruction->RegImageIds()) {
    const auto& image = reconstruction->Image(image_id);

    LOG(INFO) << StringPrintf(
        "Triangulating image #%d (%d)", image_id, image_idx++);
    const size_t num_existing_points3D = image.NumPoints3D();
    LOG(INFO) << "=> Image sees " << num_existing_points3D << " / "
              << mapper.ObservationManager().NumObservations(image_id)
              << " points";

    mapper.TriangulateImage(options_->Triangulation(), image_id);
    VLOG(1) << "=> Triangulated "
            << (image.NumPoints3D() - num_existing_points3D) << " points";
  }

  LOG(INFO) << "Retriangulation and Global bundle adjustment";
  mapper.IterativeGlobalRefinement(options_->ba_global_max_refinements,
                                   options_->ba_global_max_refinement_change,
                                   options_->Mapper(),
                                   options_->GlobalBundleAdjustment(),
                                   options_->Triangulation(),
                                   /*normalize_reconstruction=*/false);
  mapper.EndReconstruction(/*discard=*/false);

  reconstruction->UpdatePoint3DErrors();

  LOG(INFO) << "Extracting colors";
  reconstruction->ExtractColorsForAllImages(image_path_);
}

}  // namespace colmap
