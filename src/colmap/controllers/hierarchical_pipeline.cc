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

#include "colmap/controllers/hierarchical_pipeline.h"

#include "colmap/estimators/alignment.h"
#include "colmap/scene/database.h"
#include "colmap/scene/scene_clustering.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

namespace colmap {
namespace {

void MergeClusters(const SceneClustering::Cluster& cluster,
                   std::unordered_map<const SceneClustering::Cluster*,
                                      std::shared_ptr<ReconstructionManager>>*
                       reconstruction_managers) {
  // Extract all reconstructions from all child clusters.
  std::vector<std::shared_ptr<Reconstruction>> reconstructions;
  for (const auto& child_cluster : cluster.child_clusters) {
    if (!child_cluster.child_clusters.empty()) {
      MergeClusters(child_cluster, reconstruction_managers);
    }

    auto& reconstruction_manager = reconstruction_managers->at(&child_cluster);
    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
      reconstructions.push_back(reconstruction_manager->Get(i));
    }
  }

  // Try to merge all child cluster reconstruction.
  while (reconstructions.size() > 1) {
    bool merge_success = false;
    for (size_t i = 0; i < reconstructions.size(); ++i) {
      const int num_reg_images_i = reconstructions[i]->NumRegImages();
      for (size_t j = 0; j < i; ++j) {
        const double kMaxReprojError = 8.0;
        const int num_reg_images_j = reconstructions[j]->NumRegImages();
        if (MergeAndFilterReconstructions(
                kMaxReprojError, *reconstructions[j], *reconstructions[i])) {
          LOG(INFO) << StringPrintf(
              "=> Merged clusters with %d and %d images into %d images",
              num_reg_images_i,
              num_reg_images_j,
              reconstructions[i]->NumRegImages());
          reconstructions.erase(reconstructions.begin() + j);
          merge_success = true;
          break;
        }
      }

      if (merge_success) {
        break;
      }
    }

    if (!merge_success) {
      break;
    }
  }

  // Insert a new reconstruction manager for merged cluster.
  auto& reconstruction_manager = (*reconstruction_managers)[&cluster];
  reconstruction_manager = std::make_shared<ReconstructionManager>();
  for (const auto& reconstruction : reconstructions) {
    reconstruction_manager->Get(reconstruction_manager->Add()) = reconstruction;
  }

  // Delete all merged child cluster reconstruction managers.
  for (const auto& child_cluster : cluster.child_clusters) {
    reconstruction_managers->erase(&child_cluster);
  }
}

}  // namespace

bool HierarchicalPipeline::Options::Check() const {
  CHECK_OPTION_GT(init_num_trials, -1);
  CHECK_OPTION_GE(num_workers, -1);
  clustering_options.Check();
  THROW_CHECK_EQ(clustering_options.branching, 2);
  incremental_options.Check();
  return true;
}

HierarchicalPipeline::HierarchicalPipeline(
    const Options& options,
    std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(options),
      reconstruction_manager_(std::move(reconstruction_manager)) {
  THROW_CHECK(options_.Check());
  if (options_.incremental_options.ba_refine_sensor_from_rig) {
    LOG(WARNING)
        << "The hierarchical reconstruction pipeline currently does not work "
           "robustly when refining the rig extrinsics, because overlapping "
           "frames in different child clusters are optimized independently and "
           "can thus diverge significantly. The merging of clusters oftentimes "
           "fails in these cases.";
  }
}

/**
 * [功能描述]：执行分层式（Hierarchical）三维重建流程。
 * 该函数将整个场景划分为多个子聚类，分别对每个子聚类进行增量式重建，
 * 最后将各子聚类的重建结果合并为一个完整的三维重建。
 * 主要分为三个阶段：1) 场景图聚类  2) 聚类重建  3) 聚类合并
 */
void HierarchicalPipeline::Run() {
  // 打印阶段标题：场景分割
  PrintHeading1("Partitioning scene");
  // 创建计时器并启动，用于统计整个流程的运行耗时
  Timer run_timer;
  run_timer.Start();

  //////////////////////////////////////////////////////////////////////////////
  // 第一阶段：场景图聚类（Cluster Scene Graph）
  //////////////////////////////////////////////////////////////////////////////

  // 打开数据库，读取图像和特征匹配等信息
  auto database = Database::Open(options_.database_path);

  // 从数据库中读取所有图像信息
  LOG(INFO) << "Reading images...";
  const auto images = database->ReadAllImages();
  // 构建图像ID到图像名称的映射表，便于后续根据ID快速查找图像名
  std::unordered_map<image_t, std::string> image_id_to_name;
  image_id_to_name.reserve(images.size());
  for (const auto& image : images) {
    image_id_to_name.emplace(image.ImageId(), image.Name());
  }

  // 根据聚类选项和数据库中的场景图信息，创建场景聚类对象
  // 内部会基于图像之间的匹配关系进行层次化聚类
  SceneClustering scene_clustering =
      SceneClustering::Create(options_.clustering_options, *database);

  // 获取所有叶子节点聚类（最底层的子聚类，每个包含一组图像）
  auto leaf_clusters = scene_clustering.GetLeafClusters();

  // 统计并打印所有聚类中的图像总数及每个聚类的图像数量
  size_t total_num_images = 0;
  for (size_t i = 0; i < leaf_clusters.size(); ++i) {
    total_num_images += leaf_clusters[i]->image_ids.size();
    LOG(INFO) << StringPrintf("  Cluster %d with %d images",
                              i + 1,
                              leaf_clusters[i]->image_ids.size());
  }

  LOG(INFO) << StringPrintf("Clusters have %d images", total_num_images);

  //////////////////////////////////////////////////////////////////////////////
  // 第二阶段：聚类重建（Reconstruct Clusters）
  //////////////////////////////////////////////////////////////////////////////

  PrintHeading1("Reconstructing clusters");

  // 确定工作线程数和每个工作线程的子线程数
  const int kMaxNumThreads = -1;  // -1 表示使用系统所有可用线程
  const int num_eff_threads = GetEffectiveNumThreads(kMaxNumThreads);
  const int kDefaultNumWorkers = 8;  // 默认最大工作线程数
  // 计算实际工作线程数：若用户未指定（<1），则取聚类数、默认值和可用线程数的最小值
  const int num_eff_workers =
      options_.num_workers < 1
          ? std::min(static_cast<int>(leaf_clusters.size()),
                     std::min(kDefaultNumWorkers, num_eff_threads))
          : options_.num_workers;
  // 计算每个工作线程分配到的子线程数，至少为1
  const int num_threads_per_worker =
      std::max(1, num_eff_threads / num_eff_workers);

  // 定义Lambda函数：对单个聚类执行增量式三维重建
  auto ReconstructCluster =
      [this, &image_id_to_name, num_threads_per_worker](
          const SceneClustering::Cluster& cluster,
          std::shared_ptr<ReconstructionManager> reconstruction_manager) {
        // 如果聚类中没有图像，直接返回
        if (cluster.image_ids.empty()) {
          return;
        }

        // 拷贝增量重建选项并进行针对聚类的定制化设置
        auto incremental_options = std::make_shared<IncrementalPipelineOptions>(
            options_.incremental_options);
        // 设置最大模型重叠数为3，控制聚类间共享图像的数量
        incremental_options->max_model_overlap = 3;
        // 设置初始化尝试次数
        incremental_options->init_num_trials = options_.init_num_trials;
        // 如果线程数未手动设置，使用计算出的每个工作线程的子线程数
        if (incremental_options->num_threads < 0) {
          incremental_options->num_threads = num_threads_per_worker;
        }

        // 将聚类中的图像ID转换为图像名称，添加到增量重建选项中
        for (const auto image_id : cluster.image_ids) {
          incremental_options->image_names.push_back(
              image_id_to_name.at(image_id));
        }

        // 创建增量式重建管线并执行重建
        IncrementalPipeline mapper(std::move(incremental_options),
                                   options_.image_path,
                                   options_.database_path,
                                   std::move(reconstruction_manager));
        mapper.Run();
      };

  // 将聚类按图像数量从大到小排序，优先重建较大的聚类以更好地利用计算资源
  std::sort(leaf_clusters.begin(),
            leaf_clusters.end(),
            [](const SceneClustering::Cluster* cluster1,
               const SceneClustering::Cluster* cluster2) {
              return cluster1->image_ids.size() > cluster2->image_ids.size();
            });

  // 为每个聚类创建独立的重建管理器，避免多线程竞态条件
  std::unordered_map<const SceneClustering::Cluster*,
                     std::shared_ptr<ReconstructionManager>>
      reconstruction_managers;
  reconstruction_managers.reserve(leaf_clusters.size());

  // 创建线程池，将每个聚类的重建任务提交到线程池中并行执行
  ThreadPool thread_pool(num_eff_workers);
  for (const auto& cluster : leaf_clusters) {
    // 为每个聚类创建一个新的重建管理器
    reconstruction_managers[cluster] =
        std::make_shared<ReconstructionManager>();
    // 将重建任务添加到线程池
    thread_pool.AddTask(
        ReconstructCluster, *cluster, reconstruction_managers[cluster]);
  }
  // 等待所有聚类的重建任务完成
  thread_pool.Wait();

  //////////////////////////////////////////////////////////////////////////////
  // 第三阶段：聚类合并（Merge Clusters）
  //////////////////////////////////////////////////////////////////////////////

  // 如果存在多个叶子聚类，则需要将它们的重建结果合并
  if (leaf_clusters.size() > 1) {
    PrintHeading1("Merging clusters");

    // 从根聚类节点开始，递归地合并所有子聚类的重建结果
    MergeClusters(*scene_clustering.GetRootCluster(), &reconstruction_managers);
  }

  // 验证合并后只剩下一个重建管理器（即所有聚类已合并为一个）
  THROW_CHECK_EQ(reconstruction_managers.size(), 1);
  // 验证合并后的重建结果中至少有一张已注册的图像
  THROW_CHECK_GT(
      reconstruction_managers.begin()->second->Get(0)->NumRegImages(), 0);
  // 将合并后的重建结果赋值给成员变量中的重建管理器
  *reconstruction_manager_ = *reconstruction_managers.begin()->second;

  // 遍历所有重建结果，更新每个三维点的重投影误差
  for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
    auto reconstruction = reconstruction_manager_->Get(i);
    reconstruction->UpdatePoint3DErrors();
  }

  // 打印整个分层重建流程的总耗时（以分钟为单位）
  run_timer.PrintMinutes();
}

}  // namespace colmap
