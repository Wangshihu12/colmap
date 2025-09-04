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

#include "colmap/exe/database.h"
#include "colmap/exe/feature.h"
#include "colmap/exe/gui.h"
#include "colmap/exe/image.h"
#include "colmap/exe/model.h"
#include "colmap/exe/mvs.h"
#include "colmap/exe/sfm.h"
#include "colmap/exe/vocab_tree.h"
#include "colmap/util/version.h"

namespace {

typedef std::function<int(int, char**)> command_func_t;

int ShowHelp(
    const std::vector<std::pair<std::string, command_func_t>>& commands) {
  std::cout << colmap::GetVersionInfo()
            << " -- Structure-from-Motion and Multi-View Stereo\n("
            << colmap::GetBuildInfo() << ")\n\n";

  std::cout << "Usage:\n";
  std::cout << "  colmap [command] [options]\n";

  std::cout << "Documentation:\n";
  std::cout << "  https://colmap.github.io/\n";

  std::cout << "Example usage:\n";
  std::cout << "  colmap help [ -h, --help ]\n";
  std::cout << "  colmap gui\n";
  std::cout << "  colmap gui -h [ --help ]\n";
  std::cout << "  colmap automatic_reconstructor -h [ --help ]\n";
  std::cout << "  colmap automatic_reconstructor --image_path IMAGES "
               "--workspace_path WORKSPACE\n";
  std::cout << "  colmap feature_extractor --image_path IMAGES --database_path "
               "DATABASE\n";
  std::cout << "  colmap exhaustive_matcher --database_path DATABASE\n";
  std::cout << "  colmap mapper --image_path IMAGES --database_path DATABASE "
               "--output_path MODEL\n";
  std::cout << "  ...\n";

  std::cout << "Available commands:\n";
  std::cout << "  help\n";
  for (const auto& command : commands) {
    std::cout << "  " << command.first << '\n';
  }
  std::cout << '\n';

  return EXIT_SUCCESS;
}

}  // namespace

/**
 * [功能描述]：COLMAP主程序入口函数，负责解析命令行参数并调用相应的功能模块
 * @param argc [参数说明]：命令行参数数量，包括程序名
 * @param argv [参数说明]：命令行参数数组，argv[0]为程序名，argv[1]为子命令
 * @return [返回值说明]：程序执行状态，EXIT_SUCCESS表示成功，EXIT_FAILURE表示失败
 */
int main(int argc, char** argv) {
  // 初始化Google日志系统，用于输出调试和错误信息
  colmap::InitializeGlog(argv);

  // 定义命令列表：存储命令名称和对应的处理函数
  // command_func_t是函数指针类型，指向接受(int, char**)参数并返回int的函数
  std::vector<std::pair<std::string, command_func_t>> commands;
  
  // 注册所有可用的COLMAP命令及其对应的处理函数
  // GUI相关命令
  commands.emplace_back("gui", &colmap::RunGraphicalUserInterface);  // 启动图形用户界面
  
  // 重建相关命令
  commands.emplace_back("automatic_reconstructor",
                        &colmap::RunAutomaticReconstructor);  // 自动重建管道
  commands.emplace_back("bundle_adjuster", &colmap::RunBundleAdjuster);  // 捆绑调整器
  
  // 数据库操作命令
  commands.emplace_back("color_extractor", &colmap::RunColorExtractor);  // 颜色提取器
  commands.emplace_back("database_cleaner", &colmap::RunDatabaseCleaner);  // 数据库清理器
  commands.emplace_back("database_creator", &colmap::RunDatabaseCreator);  // 数据库创建器
  commands.emplace_back("database_merger", &colmap::RunDatabaseMerger);  // 数据库合并器
  
  // 网格和匹配相关命令
  commands.emplace_back("delaunay_mesher", &colmap::RunDelaunayMesher);  // Delaunay网格生成器
  commands.emplace_back("exhaustive_matcher", &colmap::RunExhaustiveMatcher);  // 穷举匹配器
  
  // 特征处理命令
  commands.emplace_back("feature_extractor", &colmap::RunFeatureExtractor);  // 特征提取器
  commands.emplace_back("feature_importer", &colmap::RunFeatureImporter);  // 特征导入器
  commands.emplace_back("geometric_verifier", &colmap::RunGeometricVerifier);  // 几何验证器
  
  // 映射和层次化处理命令
  commands.emplace_back("hierarchical_mapper", &colmap::RunHierarchicalMapper);  // 层次化映射器
  commands.emplace_back("image_deleter", &colmap::RunImageDeleter);  // 图像删除器
  commands.emplace_back("image_filterer", &colmap::RunImageFilterer);  // 图像过滤器
  commands.emplace_back("image_rectifier", &colmap::RunImageRectifier);  // 图像校正器
  commands.emplace_back("image_registrator", &colmap::RunImageRegistrator);  // 图像配准器
  commands.emplace_back("image_undistorter", &colmap::RunImageUndistorter);  // 图像去畸变器
  commands.emplace_back("image_undistorter_standalone",
                        &colmap::RunImageUndistorterStandalone);  // 独立图像去畸变器
  
  // 映射和匹配命令
  commands.emplace_back("mapper", &colmap::RunMapper);  // 映射器
  commands.emplace_back("matches_importer", &colmap::RunMatchesImporter);  // 匹配导入器
  
  // 模型处理命令
  commands.emplace_back("model_aligner", &colmap::RunModelAligner);  // 模型对齐器
  commands.emplace_back("model_analyzer", &colmap::RunModelAnalyzer);  // 模型分析器
  commands.emplace_back("model_comparer", &colmap::RunModelComparer);  // 模型比较器
  commands.emplace_back("model_converter", &colmap::RunModelConverter);  // 模型转换器
  commands.emplace_back("model_cropper", &colmap::RunModelCropper);  // 模型裁剪器
  commands.emplace_back("model_merger", &colmap::RunModelMerger);  // 模型合并器
  commands.emplace_back("model_orientation_aligner",
                        &colmap::RunModelOrientationAligner);  // 模型方向对齐器
  commands.emplace_back("model_splitter", &colmap::RunModelSplitter);  // 模型分割器
  commands.emplace_back("model_transformer", &colmap::RunModelTransformer);  // 模型变换器
  
  // 立体视觉和点云处理命令
  commands.emplace_back("patch_match_stereo", &colmap::RunPatchMatchStereo);  // 补丁匹配立体视觉
  commands.emplace_back("point_filtering", &colmap::RunPointFiltering);  // 点云过滤
  commands.emplace_back("point_triangulator", &colmap::RunPointTriangulator);  // 点三角化器
  
  // 姿态和网格生成命令
  commands.emplace_back("pose_prior_mapper", &colmap::RunPosePriorMapper);  // 姿态先验映射器
  commands.emplace_back("poisson_mesher", &colmap::RunPoissonMesher);  // Poisson网格生成器
  
  // 项目和配置命令
  commands.emplace_back("project_generator", &colmap::RunProjectGenerator);  // 项目生成器
  commands.emplace_back("rig_configurator", &colmap::RunRigConfigurator);  // 设备配置器
  commands.emplace_back("rig_bundle_adjuster", &colmap::RunRigBundleAdjuster);  // 设备捆绑调整器
  
  // 匹配策略命令
  commands.emplace_back("sequential_matcher", &colmap::RunSequentialMatcher);  // 序列匹配器
  commands.emplace_back("spatial_matcher", &colmap::RunSpatialMatcher);  // 空间匹配器
  commands.emplace_back("stereo_fusion", &colmap::RunStereoFuser);  // 立体融合器
  commands.emplace_back("transitive_matcher", &colmap::RunTransitiveMatcher);  // 传递匹配器
  
  // 词汇树相关命令
  commands.emplace_back("vocab_tree_builder", &colmap::RunVocabTreeBuilder);  // 词汇树构建器
  commands.emplace_back("vocab_tree_matcher", &colmap::RunVocabTreeMatcher);  // 词汇树匹配器
  commands.emplace_back("vocab_tree_retriever", &colmap::RunVocabTreeRetriever);  // 词汇树检索器

  // 检查命令行参数数量
  if (argc == 1) {
    // 如果没有提供子命令，显示帮助信息
    return ShowHelp(commands);
  }

  // 获取第一个参数作为子命令
  const std::string command = argv[1];
  
  // 检查是否为帮助命令
  if (command == "help" || command == "-h" || command == "--help") {
    // 显示帮助信息
    return ShowHelp(commands);
  } else {
    // 查找匹配的命令处理函数
    command_func_t matched_command_func = nullptr;
    for (const auto& command_func : commands) {
      if (command == command_func.first) {
        // 找到匹配的命令，保存对应的处理函数
        matched_command_func = command_func.second;
        break;
      }
    }
    
    if (matched_command_func == nullptr) {
      // 未找到匹配的命令，输出错误信息
      LOG(ERROR) << colmap::StringPrintf(
          "Command `%s` not recognized. To list the "
          "available commands, run `colmap help`.",
          command.c_str());
      return EXIT_FAILURE;
    } else {
      // 找到匹配的命令，准备调用对应的处理函数
      // 调整参数：移除程序名，保留子命令和后续参数
      int command_argc = argc - 1;  // 参数数量减1（去掉程序名）
      char** command_argv = &argv[1];  // 参数数组从argv[1]开始
      command_argv[0] = argv[0];  // 将程序名设置为argv[0]，保持一致性
      
      // 调用对应的命令处理函数
      return matched_command_func(command_argc, command_argv);
    }
  }

  // 正常情况下不会执行到这里，但为了完整性保留
  return ShowHelp(commands);
}
