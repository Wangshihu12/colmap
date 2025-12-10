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

#include "colmap/exe/mvs.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/meshing.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/file.h"

namespace colmap {

/**
 * [功能描述]：运行Delaunay网格化器，将稀疏或稠密点云转换为三角网格模型。
 *             该函数支持两种输入类型：稀疏重建结果和稠密重建结果。
 * @param argc：命令行参数数量。
 * @param argv：命令行参数数组。
 * @return：成功返回 EXIT_SUCCESS，失败返回 EXIT_FAILURE。
 */
int RunDelaunayMesher(int argc, char** argv) {
// 编译时检查：如果未启用CGAL库，则输出错误信息并返回失败
#if !defined(COLMAP_CGAL_ENABLED)
  LOG(ERROR) << "Delaunay meshing requires CGAL, which is not "
                "available on your system.";
  return EXIT_FAILURE;
#else   // COLMAP_CGAL_ENABLED
  // 定义输入输出路径和输入类型变量
  std::string input_path;              // 输入路径（稠密工作空间或稀疏重建目录）
  std::string input_type = "dense";    // 输入类型，默认为"dense"（稠密）
  std::string output_path;             // 输出网格文件路径

  // 配置命令行选项解析器
  OptionManager options;
  // 添加必需参数：输入路径
  options.AddRequiredOption(
      "input_path",
      &input_path,
      "Path to either the dense workspace folder or the sparse reconstruction");
  // 添加可选参数：输入类型，可选值为 "dense" 或 "sparse"
  options.AddDefaultOption("input_type", &input_type, "{dense, sparse}");
  // 添加必需参数：输出路径
  options.AddRequiredOption("output_path", &output_path);
  // 添加Delaunay网格化相关的配置选项
  options.AddDelaunayMeshingOptions();
  // 解析命令行参数
  options.Parse(argc, argv);

  // 将输入类型转换为小写，以支持大小写不敏感的输入
  StringToLower(&input_type);
  
  // 根据输入类型选择对应的网格化方法
  if (input_type == "sparse") {
    // 稀疏点云Delaunay网格化：从稀疏重建的3D点生成网格
    mvs::SparseDelaunayMeshing(
        *options.delaunay_meshing, input_path, output_path);
  } else if (input_type == "dense") {
    // 稠密点云Delaunay网格化：从稠密重建的融合点云生成网格
    mvs::DenseDelaunayMeshing(
        *options.delaunay_meshing, input_path, output_path);
  } else {
    // 输入类型无效，输出错误信息
    LOG(ERROR) << "Invalid input type - "
                  "supported values are 'sparse' and 'dense'.";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
#endif  // COLMAP_CGAL_ENABLED
}

/**
 * [功能描述]：运行PatchMatch立体匹配算法，用于估计稠密深度图和法向量图。
 *             PatchMatch是一种基于GPU加速的多视图立体匹配方法，
 *             通过迭代优化每个像素的深度和法向量来生成稠密重建结果。
 * @param argc：命令行参数数量。
 * @param argv：命令行参数数组。
 * @return：成功返回 EXIT_SUCCESS，失败返回 EXIT_FAILURE。
 */
int RunPatchMatchStereo(int argc, char** argv) {
// 编译时检查：PatchMatch算法需要CUDA支持，若未启用则报错退出
#if !defined(COLMAP_CUDA_ENABLED)
  LOG(ERROR) << "Dense stereo reconstruction requires CUDA, which is not "
                "available on your system.";
  return EXIT_FAILURE;
#else   // COLMAP_CUDA_ENABLED
  // 定义工作空间相关变量
  std::string workspace_path;                      // 工作空间路径（包含去畸变图像的文件夹）
  std::string workspace_format = "COLMAP";         // 工作空间格式，默认为"COLMAP"
  std::string pmvs_option_name = "option-all";     // PMVS格式的选项文件名，默认为"option-all"
  std::string config_path;                         // 配置文件路径（可选）

  // 配置命令行选项解析器
  OptionManager options;
  // 添加必需参数：工作空间路径
  options.AddRequiredOption(
      "workspace_path",
      &workspace_path,
      "Path to the folder containing the undistorted images");
  // 添加可选参数：工作空间格式，支持 COLMAP 或 PMVS 格式
  options.AddDefaultOption(
      "workspace_format", &workspace_format, "{COLMAP, PMVS}");
  // 添加可选参数：PMVS选项文件名（仅在PMVS格式下使用）
  options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
  // 添加可选参数：自定义配置文件路径
  options.AddDefaultOption("config_path", &config_path);
  // 添加PatchMatch立体匹配相关的配置选项（如窗口大小、迭代次数等）
  options.AddPatchMatchStereoOptions();
  // 解析命令行参数
  options.Parse(argc, argv);

  // 将工作空间格式转换为小写，支持大小写不敏感输入
  StringToLower(&workspace_format);
  // 验证工作空间格式是否有效
  if (workspace_format != "colmap" && workspace_format != "pmvs") {
    LOG(ERROR) << "Invalid `workspace_format` - supported values are "
                  "'COLMAP' or 'PMVS'.";
    return EXIT_FAILURE;
  }

  // 创建PatchMatch控制器，负责管理整个立体匹配流程
  // 参数包括：匹配选项、工作空间路径、格式、PMVS选项名、配置路径
  mvs::PatchMatchController controller(*options.patch_match_stereo,
                                       workspace_path,
                                       workspace_format,
                                       pmvs_option_name,
                                       config_path);

  // 执行PatchMatch立体匹配，生成深度图和法向量图
  controller.Run();

  return EXIT_SUCCESS;
#endif  // COLMAP_CUDA_ENABLED
}

int RunPoissonMesher(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddPoissonMeshingOptions();
  options.Parse(argc, argv);

  THROW_CHECK(
      mvs::PoissonMeshing(*options.poisson_meshing, input_path, output_path));

  return EXIT_SUCCESS;
}

/**
 * [功能描述]：运行立体融合器，将多视图深度图融合成稠密点云。
 *             该函数读取PatchMatch生成的深度图和法向量图，
 *             通过多视图一致性检查融合成统一的3D点云。
 * @param argc：命令行参数数量。
 * @param argv：命令行参数数组。
 * @return：成功返回 EXIT_SUCCESS，失败返回 EXIT_FAILURE。
 */
int RunStereoFuser(int argc, char** argv) {
  // 定义工作空间和输入输出相关变量
  std::string workspace_path;                      // 工作空间路径
  std::string input_type = "geometric";            // 输入类型，默认为"geometric"（几何一致性深度图）
  std::string workspace_format = "COLMAP";         // 工作空间格式，默认为"COLMAP"
  std::string pmvs_option_name = "option-all";     // PMVS选项文件名
  std::string output_type = "PLY";                 // 输出格式，默认为"PLY"
  std::string output_path;                         // 输出文件路径
  std::string bbox_path;                           // 边界框文件路径（可选，用于限制融合范围）

  // 配置命令行选项解析器
  OptionManager options;
  // 添加必需参数：工作空间路径
  options.AddRequiredOption("workspace_path", &workspace_path);
  // 添加可选参数：工作空间格式（COLMAP或PMVS）
  options.AddDefaultOption(
      "workspace_format", &workspace_format, "{COLMAP, PMVS}");
  // 添加可选参数：PMVS选项文件名
  options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
  // 添加可选参数：输入类型（photometric光度一致性 或 geometric几何一致性）
  options.AddDefaultOption(
      "input_type", &input_type, "{photometric, geometric}");
  // 添加可选参数：输出格式（BIN二进制、TXT文本、PLY点云）
  options.AddDefaultOption("output_type", &output_type, "{BIN, TXT, PLY}");
  // 添加必需参数：输出路径
  options.AddRequiredOption("output_path", &output_path);
  // 添加可选参数：边界框文件路径
  options.AddDefaultOption("bbox_path", &bbox_path);
  // 添加立体融合相关的配置选项（如最小视图数、深度误差阈值等）
  options.AddStereoFusionOptions();
  // 解析命令行参数
  options.Parse(argc, argv);

  // 验证工作空间格式
  StringToLower(&workspace_format);
  if (workspace_format != "colmap" && workspace_format != "pmvs") {
    LOG(ERROR) << "Invalid `workspace_format` - supported values are "
                  "'COLMAP' or 'PMVS'.";
    return EXIT_FAILURE;
  }

  // 验证输入类型
  StringToLower(&input_type);
  if (input_type != "photometric" && input_type != "geometric") {
    LOG(ERROR) << "Invalid input type - supported values are "
                  "'photometric' and 'geometric'.";
    return EXIT_FAILURE;
  }

  // 如果提供了边界框文件，读取边界框参数用于限制融合范围
  if (!bbox_path.empty()) {
    std::ifstream file(bbox_path);
    if (file.is_open()) {
      // 获取边界框的最小和最大坐标引用
      auto& min_bound = options.stereo_fusion->bounding_box.first;   // 边界框最小点 (x_min, y_min, z_min)
      auto& max_bound = options.stereo_fusion->bounding_box.second;  // 边界框最大点 (x_max, y_max, z_max)
      // 从文件读取边界框坐标值
      file >> min_bound(0) >> min_bound(1) >> min_bound(2);
      file >> max_bound(0) >> max_bound(1) >> max_bound(2);
    } else {
      // 边界框文件无法打开，输出警告但继续执行（不进行边界检查）
      LOG(WARNING) << "Invalid bounds path: \"" << bbox_path
                   << "\" - continuing without bounds check";
    }
  }

  // 创建立体融合器对象，传入融合选项和工作空间配置
  mvs::StereoFusion fuser(*options.stereo_fusion,
                          workspace_path,
                          workspace_format,
                          pmvs_option_name,
                          input_type);

  // 执行深度图融合，生成稠密点云
  fuser.Run();

  // 创建重建对象，用于存储和输出点云数据
  Reconstruction reconstruction;

  // 如果是COLMAP格式，从稀疏重建目录读取相机和图像信息
  if (workspace_format == "colmap") {
    reconstruction.Read(JoinPaths(workspace_path, "sparse"));
  }

  // 将融合器生成的稠密点云导入到重建对象中，覆盖原有的稀疏点云
  reconstruction.ImportPLY(fuser.GetFusedPoints());

  LOG(INFO) << "Writing output: " << output_path;

  // 根据指定的输出格式写入融合结果
  StringToLower(&output_type);
  if (output_type == "bin") {
    // 以二进制格式写入完整重建数据（包含相机、图像、点云）
    reconstruction.WriteBinary(output_path);
  } else if (output_type == "txt") {
    // 以文本格式写入完整重建数据
    reconstruction.WriteText(output_path);
  } else if (output_type == "ply") {
    // 以PLY格式写入稠密点云
    WriteBinaryPlyPoints(output_path, fuser.GetFusedPoints());
    // 同时写入点云可见性信息（记录每个点被哪些视图观测到）
    mvs::WritePointsVisibility(output_path + ".vis",
                               fuser.GetFusedPointsVisibility());
  } else {
    LOG(ERROR) << "Invalid `output_type`";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

}  // namespace colmap
