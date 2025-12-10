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

#include "colmap/mvs/fusion.h"

#include "colmap/math/math.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <Eigen/Geometry>

namespace colmap {
namespace mvs {
namespace internal {

// Use the sparse model to find most connected image that has not yet been
// fused. This is used as a heuristic to ensure that the workspace cache reuses
// already cached images as efficient as possible.
int FindNextImage(const std::vector<std::vector<int>>& overlapping_images,
                  const std::vector<char>& used_images,
                  const std::vector<char>& fused_images,
                  const int prev_image_idx) {
  THROW_CHECK_EQ(used_images.size(), fused_images.size());

  for (const auto image_idx : overlapping_images.at(prev_image_idx)) {
    if (used_images.at(image_idx) && !fused_images.at(image_idx)) {
      return image_idx;
    }
  }

  // If none of the overlapping images are not yet fused, simply return the
  // first image that has not yet been fused.
  for (size_t image_idx = 0; image_idx < fused_images.size(); ++image_idx) {
    if (used_images[image_idx] && !fused_images[image_idx]) {
      return image_idx;
    }
  }

  return -1;
}

}  // namespace internal

void StereoFusionOptions::Print() const {
#define PrintOption(option) LOG(INFO) << #option ": " << (option) << '\n';
  PrintHeading2("StereoFusion::Options");
  PrintOption(mask_path);
  PrintOption(max_image_size);
  PrintOption(min_num_pixels);
  PrintOption(max_num_pixels);
  PrintOption(max_traversal_depth);
  PrintOption(max_reproj_error);
  PrintOption(max_depth_error);
  PrintOption(max_normal_error);
  PrintOption(check_num_images);
  PrintOption(use_cache);
  PrintOption(cache_size);
  const auto& bbox_min = bounding_box.first.transpose().eval();
  const auto& bbox_max = bounding_box.second.transpose().eval();
  PrintOption(bbox_min);
  PrintOption(bbox_max);
#undef PrintOption
}

bool StereoFusionOptions::Check() const {
  CHECK_OPTION_GE(min_num_pixels, 0);
  CHECK_OPTION_LE(min_num_pixels, max_num_pixels);
  CHECK_OPTION_GT(max_traversal_depth, 0);
  CHECK_OPTION_GE(max_reproj_error, 0);
  CHECK_OPTION_GE(max_depth_error, 0);
  CHECK_OPTION_GE(max_normal_error, 0);
  CHECK_OPTION_GT(check_num_images, 0);
  CHECK_OPTION_GT(cache_size, 0);
  return true;
}

StereoFusion::StereoFusion(const StereoFusionOptions& options,
                           const std::string& workspace_path,
                           const std::string& workspace_format,
                           const std::string& pmvs_option_name,
                           const std::string& input_type)
    : options_(options),
      workspace_path_(workspace_path),
      workspace_format_(workspace_format),
      pmvs_option_name_(pmvs_option_name),
      input_type_(input_type),
      max_squared_reproj_error_(options_.max_reproj_error *
                                options_.max_reproj_error),
      min_cos_normal_error_(std::cos(DegToRad(options_.max_normal_error))) {
  THROW_CHECK(options_.Check());
}

const std::vector<PlyPoint>& StereoFusion::GetFusedPoints() const {
  return fused_points_;
}

const std::vector<std::vector<int>>& StereoFusion::GetFusedPointsVisibility()
    const {
  return fused_points_visibility_;
}

/**
 * [功能描述]：执行立体融合的主函数，将多视图深度图融合成稠密3D点云。
 *             该函数遍历所有图像的深度图，通过多视图一致性检查将像素深度
 *             融合成统一的3D点，同时记录每个点的可见性信息。
 */
void StereoFusion::Run() {
  // 初始化计时器，用于统计整个融合过程的耗时
  Timer run_timer;
  run_timer.Start();

  // 清空之前的融合结果
  fused_points_.clear();                 // 清空融合点云容器
  fused_points_visibility_.clear();      // 清空点云可见性容器

  // 打印当前融合选项配置
  options_.Print();

  LOG(INFO) << "Reading workspace...";

  // ============== 第一阶段：配置并加载工作空间 ==============
  Workspace::Options workspace_options;

  // 处理PMVS格式的工作空间，设置对应的立体匹配文件夹
  auto workspace_format_lower_case = workspace_format_;
  StringToLower(&workspace_format_lower_case);
  if (workspace_format_lower_case == "pmvs") {
    // PMVS格式使用特定的立体匹配文件夹命名格式
    workspace_options.stereo_folder =
        StringPrintf("stereo-%s", pmvs_option_name_.c_str());
  }

  // 设置工作空间选项参数
  workspace_options.num_threads = options_.num_threads;        // 线程数
  workspace_options.max_image_size = options_.max_image_size;  // 最大图像尺寸
  workspace_options.image_as_rgb = true;                       // 以RGB格式读取图像
  workspace_options.cache_size = options_.cache_size;          // 缓存大小
  workspace_options.workspace_path = workspace_path_;          // 工作空间路径
  workspace_options.workspace_format = workspace_format_;      // 工作空间格式
  workspace_options.input_type = input_type_;                  // 输入类型（geometric/photometric）

  // 从fusion.cfg配置文件读取待融合的图像名称列表
  const auto image_names = ReadTextFileLines(JoinPaths(
      workspace_path_, workspace_options.stereo_folder, "fusion.cfg"));
  
  // 根据是否使用缓存模式创建不同类型的工作空间
  int num_threads = 1;
  if (options_.use_cache) {
    // 缓存模式：按需加载数据，适用于大规模数据集（内存受限情况）
    workspace_ = std::make_unique<CachedWorkspace>(workspace_options);
  } else {
    // 非缓存模式：一次性加载所有数据到内存，支持多线程并行处理
    workspace_ = std::make_unique<Workspace>(workspace_options);
    workspace_->Load(image_names);
    num_threads = GetEffectiveNumThreads(options_.num_threads);
  }

  // 检查是否收到停止信号
  if (CheckIfStopped()) {
    run_timer.PrintMinutes();
    return;
  }

  LOG(INFO) << "Reading configuration...";

  // ============== 第二阶段：初始化数据结构 ==============
  // 获取场景模型（包含相机参数和图像信息）
  const auto& model = workspace_->GetModel();

  // 获取每张图像的最大重叠图像列表（用于确定融合时的参考视图）
  const double kMinTriangulationAngle = 0;  // 最小三角化角度阈值
  if (model.GetMaxOverlappingImagesFromPMVS().empty()) {
    // 如果没有PMVS提供的重叠信息，则根据相机位置计算
    overlapping_images_ = model.GetMaxOverlappingImages(
        options_.check_num_images, kMinTriangulationAngle);
  } else {
    // 使用PMVS提供的重叠图像信息
    overlapping_images_ = model.GetMaxOverlappingImagesFromPMVS();
  }

  // 为每个线程分配独立的融合点存储空间（避免线程竞争）
  task_fused_points_.resize(num_threads);              // 各线程的融合点容器
  task_fused_points_visibility_.resize(num_threads);   // 各线程的可见性容器

  // 初始化图像级别的数据结构
  used_images_.resize(model.images.size(), false);      // 标记图像是否有效（有完整输入数据）
  fused_images_.resize(model.images.size(), false);     // 标记图像是否已完成融合
  fused_pixel_masks_.resize(model.images.size());       // 每个图像的像素融合掩码
  depth_map_sizes_.resize(model.images.size());         // 深度图尺寸
  bitmap_scales_.resize(model.images.size());           // 图像与深度图的缩放比例
  P_.resize(model.images.size());                       // 投影矩阵 P = K[R|t]
  inv_P_.resize(model.images.size());                   // 逆投影矩阵（用于反投影）
  inv_R_.resize(model.images.size());                   // 旋转矩阵的逆（转置）

  // ============== 第三阶段：预计算每张图像的相机参数 ==============
  for (const auto& image_name : image_names) {
    const int image_idx = model.GetImageIdx(image_name);

    // 检查图像是否具有完整的输入数据（位图、深度图、法向量图）
    if (!workspace_->HasBitmap(image_idx) ||
        !workspace_->HasDepthMap(image_idx) ||
        !workspace_->HasNormalMap(image_idx)) {
      LOG(WARNING) << StringPrintf(
          "Ignoring image %s, because input does not exist.",
          image_name.c_str());
      continue;
    }

    const auto& image = model.images.at(image_idx);
    const auto& depth_map = workspace_->GetDepthMap(image_idx);

    // 标记该图像为有效图像
    used_images_.at(image_idx) = true;

    // 初始化该图像的像素融合掩码（记录哪些像素已被融合）
    InitFusedPixelMask(image_idx, depth_map.GetWidth(), depth_map.GetHeight());

    // 存储深度图尺寸
    depth_map_sizes_.at(image_idx) =
        std::make_pair(depth_map.GetWidth(), depth_map.GetHeight());

    // 计算深度图与原始图像的缩放比例（深度图可能被降采样）
    bitmap_scales_.at(image_idx) = std::make_pair(
        static_cast<float>(depth_map.GetWidth()) / image.GetWidth(),
        static_cast<float>(depth_map.GetHeight()) / image.GetHeight());

    // 获取并调整相机内参矩阵K（根据缩放比例调整焦距和主点）
    // K矩阵形状: [3x3]，包含 fx, fy, cx, cy
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
            image.GetK());
    K(0, 0) *= bitmap_scales_.at(image_idx).first;   // 调整 fx
    K(0, 2) *= bitmap_scales_.at(image_idx).first;   // 调整 cx
    K(1, 1) *= bitmap_scales_.at(image_idx).second;  // 调整 fy
    K(1, 2) *= bitmap_scales_.at(image_idx).second;  // 调整 cy

    // 计算投影矩阵 P = K[R|t]，用于将3D点投影到图像平面
    ComposeProjectionMatrix(
        K.data(), image.GetR(), image.GetT(), P_.at(image_idx).data());
    // 计算逆投影矩阵，用于将2D像素反投影到3D空间
    ComposeInverseProjectionMatrix(
        K.data(), image.GetR(), image.GetT(), inv_P_.at(image_idx).data());
    // 存储旋转矩阵的逆（转置），用于法向量变换
    inv_R_.at(image_idx) =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
            image.GetR())
            .transpose();
  }

  // ============== 第四阶段：多线程并行融合 ==============
  LOG(INFO) << StringPrintf("Starting fusion with %d threads", num_threads);
  ThreadPool thread_pool(num_threads);

  // 使用行步长为10来避免相邻行并行处理时产生重复工作
  // 因为相邻像素很可能融合到同一个3D点
  const int kRowStride = 10;
  
  // 定义处理图像行的Lambda函数
  auto ProcessImageRows = [&, this](const int row_start,
                                    const int height,
                                    const int width,
                                    const int image_idx,
                                    const Mat<char>& fused_pixel_mask) {
    // 计算当前任务处理的行范围
    const int row_end = std::min(height, row_start + kRowStride);
    // 遍历指定行范围内的所有像素
    for (int row = row_start; row < row_end; ++row) {
      for (int col = 0; col < width; ++col) {
        // 如果该像素已被融合，则跳过
        if (fused_pixel_mask.Get(row, col) > 0) {
          continue;
        }
        // 获取当前线程ID，用于将结果存入对应线程的容器
        const int thread_id = thread_pool.GetThreadIndex();
        // 执行单个像素的融合操作
        Fuse(thread_id, image_idx, row, col);
      }
    }
  };

  // 遍历所有图像进行融合
  size_t num_fused_images = 0;      // 已融合图像计数
  size_t total_fused_points = 0;    // 总融合点数
  // 循环遍历图像，FindNextImage根据重叠关系选择下一张待处理图像
  for (int image_idx = 0; image_idx >= 0;
       image_idx = internal::FindNextImage(
           overlapping_images_, used_images_, fused_images_, image_idx)) {
    // 检查是否收到停止信号
    if (CheckIfStopped()) {
      break;
    }

    // 为当前图像计时
    Timer timer;
    timer.Start();

    LOG(INFO) << StringPrintf("Fusing image [%d/%d] with index %d",
                              num_fused_images + 1,
                              model.images.size(),
                              image_idx)
              << std::flush;

    // 获取当前图像的深度图尺寸和融合掩码
    const int width = depth_map_sizes_.at(image_idx).first;
    const int height = depth_map_sizes_.at(image_idx).second;
    const auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);

    // 将图像按行分块，提交到线程池并行处理
    for (int row_start = 0; row_start < height; row_start += kRowStride) {
      thread_pool.AddTask(ProcessImageRows,
                          row_start,
                          height,
                          width,
                          image_idx,
                          fused_pixel_mask);
    }
    // 等待当前图像的所有任务完成
    thread_pool.Wait();

    // 更新融合状态
    num_fused_images += 1;
    fused_images_.at(image_idx) = true;

    // 统计当前总融合点数（累加所有线程的结果）
    total_fused_points = 0;
    for (const auto& task_fused_points : task_fused_points_) {
      total_fused_points += task_fused_points.size();
    }
    LOG(INFO) << StringPrintf(
        " in %.3fs (%d points)", timer.ElapsedSeconds(), total_fused_points);
  }

  // ============== 第五阶段：合并各线程的融合结果 ==============
  // 预分配内存以提高效率
  fused_points_.reserve(total_fused_points);
  fused_points_visibility_.reserve(total_fused_points);
  
  // 将各线程的融合点和可见性信息合并到主容器
  for (size_t thread_id = 0; thread_id < task_fused_points_.size();
       ++thread_id) {
    // 合并融合点
    fused_points_.insert(fused_points_.end(),
                         task_fused_points_[thread_id].begin(),
                         task_fused_points_[thread_id].end());
    task_fused_points_[thread_id].clear();  // 释放线程局部存储

    // 合并可见性信息
    fused_points_visibility_.insert(
        fused_points_visibility_.end(),
        task_fused_points_visibility_[thread_id].begin(),
        task_fused_points_visibility_[thread_id].end());
    task_fused_points_visibility_[thread_id].clear();  // 释放线程局部存储
  }

  // 检查融合结果是否为空
  if (fused_points_.empty()) {
    LOG(WARNING)
        << "Could not fuse any points. This is likely caused by "
           "incorrect settings - filtering must be enabled for the last "
           "call to patch match stereo.";
  }

  // 输出最终统计信息
  LOG(INFO) << "Number of fused points: " << fused_points_.size();
  run_timer.PrintMinutes();
}

void StereoFusion::InitFusedPixelMask(int image_idx,
                                      size_t width,
                                      size_t height) {
  Bitmap mask;
  Mat<char>& fused_pixel_mask = fused_pixel_masks_.at(image_idx);
  const std::string mask_image_name =
      workspace_->GetModel().GetImageName(image_idx);
  std::string mask_path =
      JoinPaths(options_.mask_path, mask_image_name + ".png");
  if (!ExistsFile(mask_path) && HasFileExtension(mask_image_name, ".png")) {
    mask_path = JoinPaths(options_.mask_path, mask_image_name);
  }
  fused_pixel_mask = Mat<char>(width, height, 1);
  if (!options_.mask_path.empty() && ExistsFile(mask_path) &&
      mask.Read(mask_path, false)) {
    BitmapColor<uint8_t> color;
    mask.Rescale(static_cast<int>(width),
                 static_cast<int>(height),
                 Bitmap::RescaleFilter::kBox);
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        mask.GetPixel(col, row, &color);
        fused_pixel_mask.Set(row, col, color.r == 0 ? 1 : 0);
      }
    }
  } else {
    fused_pixel_mask.Fill(0);
  }
}

/**
 * [功能描述]：对单个像素执行多视图融合操作。
 *             从参考像素出发，通过投影到重叠视图找到对应像素，
 *             进行深度、重投影和法向量一致性检查后融合成单个3D点。
 * @param thread_id：当前执行线程的ID，用于将结果存入对应线程的容器。
 * @param image_idx：参考图像的索引。
 * @param row：参考像素的行坐标。
 * @param col：参考像素的列坐标。
 */
void StereoFusion::Fuse(const int thread_id,
                        const int image_idx,
                        const int row,
                        const int col) {
  // 融合队列：存储待处理的像素（使用类似BFS的遍历方式）
  std::vector<FusionData> fusion_queue;
  // 将初始参考像素加入队列，遍历深度为0
  fusion_queue.emplace_back(image_idx, row, col, 0);

  // 参考点信息：用于后续视图的一致性检查
  Eigen::Vector4f fused_ref_point = Eigen::Vector4f::Zero();   // 参考点的齐次坐标 [x, y, z, 1]
  Eigen::Vector3f fused_ref_normal = Eigen::Vector3f::Zero();  // 参考点的法向量

  // 累积容器：存储所有通过一致性检查的像素对应的3D点信息
  std::vector<float> fused_point_x;    // 3D点X坐标集合
  std::vector<float> fused_point_y;    // 3D点Y坐标集合
  std::vector<float> fused_point_z;    // 3D点Z坐标集合
  std::vector<float> fused_point_nx;   // 法向量X分量集合
  std::vector<float> fused_point_ny;   // 法向量Y分量集合
  std::vector<float> fused_point_nz;   // 法向量Z分量集合
  std::vector<uint8_t> fused_point_r;  // 颜色R通道集合
  std::vector<uint8_t> fused_point_g;  // 颜色G通道集合
  std::vector<uint8_t> fused_point_b;  // 颜色B通道集合
  std::unordered_set<int> fused_point_visibility;  // 可见该点的图像索引集合

  // ============== 主循环：遍历融合队列中的所有候选像素 ==============
  while (!fusion_queue.empty()) {
    // 取出队列尾部元素（LIFO方式，类似DFS）
    const auto data = fusion_queue.back();
    const int image_idx = data.image_idx;          // 当前图像索引
    const int row = data.row;                      // 当前像素行
    const int col = data.col;                      // 当前像素列
    const int traversal_depth = data.traversal_depth;  // 当前遍历深度

    fusion_queue.pop_back();

    // ---------- 检查1：像素是否已被融合 ----------
    auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);
    if (fused_pixel_mask.Get(row, col) > 0) {
      continue;  // 已融合，跳过
    }

    // 获取当前像素的深度值
    const auto& depth_map = workspace_->GetDepthMap(image_idx);
    const float depth = depth_map.Get(row, col);

    // ---------- 检查2：深度值有效性 ----------
    if (depth <= 0.0f) {
      continue;  // 无效深度（被过滤或无深度信息），跳过
    }

    // ---------- 检查3：与参考点的一致性（仅对非参考像素执行） ----------
    // 遍历深度>0表示当前不是初始参考像素，需要进行一致性检查
    if (traversal_depth > 0) {
      // 将参考点投影到当前视图，得到投影坐标 [u*z, v*z, z]
      const Eigen::Vector3f proj = P_.at(image_idx) * fused_ref_point;

      // 计算深度误差：|投影深度 - 实际深度| / 实际深度
      const float depth_error = std::abs((proj(2) - depth) / depth);
      if (depth_error > options_.max_depth_error) {
        continue;  // 深度误差过大，跳过
      }

      // 计算重投影误差：参考点投影位置与当前像素位置的差异
      const float col_diff = proj(0) / proj(2) - col;  // 列方向差异
      const float row_diff = proj(1) / proj(2) - row;  // 行方向差异
      const float squared_reproj_error =
          col_diff * col_diff + row_diff * row_diff;   // 平方重投影误差
      if (squared_reproj_error > max_squared_reproj_error_) {
        continue;  // 重投影误差过大，跳过
      }
    }

    // ---------- 计算法向量（从相机坐标系转换到世界坐标系） ----------
    const auto& normal_map = workspace_->GetNormalMap(image_idx);
    // 使用旋转矩阵的逆（转置）将法向量从相机坐标系变换到世界坐标系
    const Eigen::Vector3f normal =
        inv_R_.at(image_idx) * Eigen::Vector3f(normal_map.Get(row, col, 0),
                                               normal_map.Get(row, col, 1),
                                               normal_map.Get(row, col, 2));

    // ---------- 检查4：法向量一致性（仅对非参考像素执行） ----------
    if (traversal_depth > 0) {
      // 计算当前法向量与参考法向量的夹角余弦值
      const float cos_normal_error = fused_ref_normal.dot(normal);
      if (cos_normal_error < min_cos_normal_error_) {
        continue;  // 法向量差异过大，跳过
      }
    }

    // ---------- 计算像素对应的3D点坐标 ----------
    // 使用逆投影矩阵将2D像素反投影到3D空间
    // 输入：[col*depth, row*depth, depth, 1]（齐次像素坐标乘以深度）
    const Eigen::Vector3f xyz =
        inv_P_.at(image_idx) *
        Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

    // ---------- 读取像素颜色 ----------
    BitmapColor<uint8_t> color;
    const auto& bitmap_scale = bitmap_scales_.at(image_idx);
    // 使用最近邻插值获取颜色（考虑深度图与原图的缩放比例）
    workspace_->GetBitmap(image_idx).InterpolateNearestNeighbor(
        col / bitmap_scale.first, row / bitmap_scale.second, &color);

    // 标记当前像素为已访问/已融合
    fused_pixel_mask.Set(row, col, 1);

    // ---------- 检查5：边界框过滤 ----------
    // 检查3D点是否在用户指定的边界框内
    if (xyz(0) < options_.bounding_box.first(0) ||
        xyz(1) < options_.bounding_box.first(1) ||
        xyz(2) < options_.bounding_box.first(2) ||
        xyz(0) > options_.bounding_box.second(0) ||
        xyz(1) > options_.bounding_box.second(1) ||
        xyz(2) > options_.bounding_box.second(2)) {
      continue;  // 超出边界框，跳过
    }

    // ---------- 累积当前像素的信息到融合点 ----------
    fused_point_x.push_back(xyz(0));
    fused_point_y.push_back(xyz(1));
    fused_point_z.push_back(xyz(2));
    fused_point_nx.push_back(normal(0));
    fused_point_ny.push_back(normal(1));
    fused_point_nz.push_back(normal(2));
    fused_point_r.push_back(color.r);
    fused_point_g.push_back(color.g);
    fused_point_b.push_back(color.b);
    fused_point_visibility.insert(image_idx);  // 记录该图像可见此点

    // ---------- 设置参考点（仅对初始像素执行） ----------
    if (traversal_depth == 0) {
      // 第一个像素作为参考点，后续像素都与其进行一致性比较
      fused_ref_point = Eigen::Vector4f(xyz(0), xyz(1), xyz(2), 1.0f);
      fused_ref_normal = normal;
    }

    // ---------- 终止条件1：达到最大像素数 ----------
    if (fused_point_x.size() >= static_cast<size_t>(options_.max_num_pixels)) {
      break;  // 已收集足够多的像素，停止遍历
    }

    // ---------- 终止条件2：达到最大遍历深度 ----------
    if (traversal_depth >= options_.max_traversal_depth - 1) {
      continue;  // 达到最大深度，不再向下一层扩展
    }

    // ---------- 扩展：将当前点投影到重叠视图，添加新的候选像素 ----------
    for (const auto next_image_idx : overlapping_images_.at(image_idx)) {
      // 跳过无效图像或已完成融合的图像
      if (!used_images_.at(next_image_idx) ||
          fused_images_.at(next_image_idx)) {
        continue;
      }

      // 将当前3D点投影到下一个视图
      const Eigen::Vector3f next_proj =
          P_.at(next_image_idx) * xyz.homogeneous();
      // 计算投影像素坐标（四舍五入取整）
      const int next_col =
          static_cast<int>(std::round(next_proj(0) / next_proj(2)));
      const int next_row =
          static_cast<int>(std::round(next_proj(1) / next_proj(2)));

      // 检查投影坐标是否在图像范围内
      const auto& depth_map_size = depth_map_sizes_.at(next_image_idx);
      if (next_col < 0 || next_row < 0 || next_col >= depth_map_size.first ||
          next_row >= depth_map_size.second) {
        continue;  // 超出图像范围，跳过
      }
      // 将新的候选像素加入融合队列，遍历深度+1
      fusion_queue.emplace_back(
          next_image_idx, next_row, next_col, traversal_depth + 1);
    }
  }

  // ============== 融合结果生成：计算最终的3D点 ==============
  const size_t num_pixels = fused_point_x.size();
  // 只有当融合的像素数达到最小阈值时才生成有效点
  if (num_pixels >= static_cast<size_t>(options_.min_num_pixels)) {
    PlyPoint fused_point;  // 最终融合点

    // ---------- 计算融合法向量（取各分量的中值） ----------
    Eigen::Vector3f fused_normal;
    fused_normal.x() = Median(fused_point_nx);
    fused_normal.y() = Median(fused_point_ny);
    fused_normal.z() = Median(fused_point_nz);
    const float fused_normal_norm = fused_normal.norm();
    // 法向量模长过小则丢弃该点（避免归一化时除零）
    if (fused_normal_norm < std::numeric_limits<float>::epsilon()) {
      return;
    }

    // ---------- 计算融合3D坐标（取各坐标分量的中值） ----------
    fused_point.x = Median(fused_point_x);
    fused_point.y = Median(fused_point_y);
    fused_point.z = Median(fused_point_z);

    // ---------- 归一化法向量 ----------
    fused_point.nx = fused_normal.x() / fused_normal_norm;
    fused_point.ny = fused_normal.y() / fused_normal_norm;
    fused_point.nz = fused_normal.z() / fused_normal_norm;

    // ---------- 计算融合颜色（取各通道的中值） ----------
    fused_point.r =
        TruncateCast<float, uint8_t>(std::round(Median(fused_point_r)));
    fused_point.g =
        TruncateCast<float, uint8_t>(std::round(Median(fused_point_g)));
    fused_point.b =
        TruncateCast<float, uint8_t>(std::round(Median(fused_point_b)));

    // ---------- 将融合点存入当前线程的结果容器 ----------
    task_fused_points_[thread_id].push_back(fused_point);
    // 同时存储该点的可见性信息（哪些图像可以观测到该点）
    task_fused_points_visibility_[thread_id].emplace_back(
        fused_point_visibility.begin(), fused_point_visibility.end());
  }
}

void WritePointsVisibility(
    const std::string& path,
    const std::vector<std::vector<int>>& points_visibility) {
  std::fstream file(path, std::ios::out | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  WriteBinaryLittleEndian<uint64_t>(&file, points_visibility.size());

  for (const auto& visibility : points_visibility) {
    WriteBinaryLittleEndian<uint32_t>(&file, visibility.size());
    for (const auto& image_idx : visibility) {
      WriteBinaryLittleEndian<uint32_t>(&file, image_idx);
    }
  }
}

}  // namespace mvs
}  // namespace colmap
