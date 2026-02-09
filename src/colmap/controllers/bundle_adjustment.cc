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

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <ceres/ceres.h>

namespace colmap {
namespace {

// Callback functor called after each bundle adjustment iteration.
class BundleAdjustmentIterationCallback : public ceres::IterationCallback {
 public:
  explicit BundleAdjustmentIterationCallback(BaseController* controller)
      : controller_(controller) {}

  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    THROW_CHECK_NOTNULL(controller_);
    if (controller_->CheckIfStopped()) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    } else {
      return ceres::SOLVER_CONTINUE;
    }
  }

 private:
  BaseController* controller_;
};

}  // namespace

BundleAdjustmentController::BundleAdjustmentController(
    const OptionManager& options,
    std::shared_ptr<Reconstruction> reconstruction)
    : options_(options), reconstruction_(std::move(reconstruction)) {}

/**
 * [功能描述]：执行全局光束法平差（Bundle Adjustment）优化。
 * 该函数对所有已注册的图像进行全局BA优化，以最小化重投影误差，
 * 从而精化相机位姿和三维点坐标。
 */
void BundleAdjustmentController::Run() {
  // 检查重建对象是否有效（非空指针）
  THROW_CHECK_NOTNULL(reconstruction_);

  // 打印标题，标识当前阶段为全局光束法平差
  PrintHeading1("Global bundle adjustment");
  // 创建计时器并启动，用于统计BA运行耗时
  Timer run_timer;
  run_timer.Start();

  // 检查是否至少有一个已注册的帧，否则无法进行BA优化
  if (reconstruction_->NumRegFrames() == 0) {
    LOG(ERROR) << "Need at least one registered frame.";
    return;
  }

  // 过滤掉深度为负的观测点，避免BA优化中出现退化情况
  // 负深度表示三维点位于相机后方，属于无效观测
  ObservationManager(*reconstruction_).FilterObservationsWithNegativeDepth();

  // 从全局选项中拷贝一份BA优化参数
  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;

  // 创建迭代回调对象，用于在每次BA迭代后检查是否需要提前终止优化
  BundleAdjustmentIterationCallback iteration_callback(this);
  // 将回调注册到Ceres求解器的回调列表中
  ba_options.solver_options.callbacks.push_back(&iteration_callback);

  // 配置BA优化的图像集合
  BundleAdjustmentConfig ba_config;
  // 遍历所有已注册的图像ID，将其添加到BA配置中参与优化
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    ba_config.AddImage(image_id);
  }
  // 固定规范自由度（Gauge Freedom）：使用两个相机的世界坐标位姿来固定
  // 相比固定三个三维点的方式，固定两个相机能带来更稳定的优化，收敛步数更少
  // TODO(jsch): 研究是否可以完全不固定规范自由度，初步实验表明这样更快
  ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  // 创建默认的BA求解器，传入优化参数、配置和重建对象
  std::unique_ptr<BundleAdjuster> bundle_adjuster = CreateDefaultBundleAdjuster(
      std::move(ba_options), std::move(ba_config), *reconstruction_);
  // 执行BA求解
  bundle_adjuster->Solve();
  // 更新所有三维点的重投影误差
  reconstruction_->UpdatePoint3DErrors();

  // 打印BA优化总耗时（以分钟为单位）
  run_timer.PrintMinutes();
}

}  // namespace colmap
