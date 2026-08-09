<div align="center">

# ScaleGuard-4K

**面向复杂退化的跨尺度一致性高分辨率恢复智能体**

[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/liuqjjin/scaleguard-4k/actions/workflows/ci.yml/badge.svg)](https://github.com/liuqjjin/scaleguard-4k/actions/workflows/ci.yml)

ScaleGuard-4K 是一套面向真实复杂退化的图像恢复与逐级超分系统。它先分析输入中的模糊、噪声、压缩、雨、雾和低照度，再处理当前配置支持的退化，最后以 4× 为单位生成更高分辨率候选。系统会在每一级检查画质和跨尺度一致性，选择继续放大、保留当前结果或退回上一尺度。

</div>

<p align="center">
  <img src="assets/figures/scaleguard-teaser.webp" width="100%" alt="ScaleGuard-4K 方法概览">
</p>

## 项目简介

真实图像很少只含一种退化。运动与散焦会抹去边缘，传感器噪声和 JPEG 压缩会干扰纹理，雨、雾和低照度还会改变对比度与颜色。如果直接进行生成式超分，这些缺陷会进入纹理生成过程；连续放大时，前一级产生的偏差又会成为下一级的输入。

ScaleGuard-4K 将整个任务拆成两个阶段。第一阶段在原始分辨率上分析退化、规划恢复顺序并调用专家工具；第二阶段逐次生成 4× 候选，并在每次生成后决定是否接受。恢复 Agent 负责“先处理什么”，尺度控制器负责“放大到哪里为止”。

系统支持 1×、2×、4×、8× 和 16× 输出。16× 不是一次黑盒推理，而是同一会话中的两次 4× 生成，中间保留一次明确的质量判断与回退机会。

<p align="center">
  <img src="assets/figures/complex-degradation-gallery.webp" width="100%" alt="复杂退化图像的恢复与尺度选择示意">
</p>

## 系统架构

<p align="center">
  <img src="assets/figures/system-overview.webp" width="100%" alt="ScaleGuard-4K 系统架构">
</p>

| 环节 | 默认组件 | 作用 |
| --- | --- | --- |
| 退化诊断 | DepictQA（CLIP ViT-L/14、Vicuna-7B、退化适配器） | 七类退化的五级评估 |
| 内容描述 | Qwen2.5-VL-7B | 为图文偏好评价生成语义描述 |
| 任务排序 | Qwen3.7 Flash | 输出退化恢复任务的严格排列 |
| 专家恢复 | SwinIR、MPRNet、Restormer、DehazeFormer、FBCNN | 去噪、去模糊、去雨、去雾和压缩伪影去除 |
| 生成式超分 | Stable Diffusion 3 Medium、SR Transformer LoRA、VAE Encoder LoRA | 单步 4× 潜空间恢复 |
| 多尺度提示 | Qwen2.5-VL-3B、提示 LoRA | 从语义锚点和当前局部区域生成文字条件 |
| 在线尺度判断 | MUSIQ、RGB NRMSE、Gradient MAE、可选 Measurement NRMSE | 接受、停止或回退候选 |

### 退化感知

感知模块从两个角度理解输入图像：专用退化评估器逐项判断运动模糊、散焦、雨、雾、低照度、噪声和 JPEG 伪影的程度；本地视觉语言模型生成图像内容描述，为后续的感知质量比较提供语义条件。默认运行配置会诊断低照度，但不自动执行增亮，以避免在未标定时引入额外颜色漂移。

退化结果会被整理成结构化任务集合。远程规划器只接收任务名称、退化标签和历史经验，以 JSON 形式返回任务顺序，不接收图像像素。返回结果需要包含完整且无重复的任务排列，格式或顺序不合法时会在有限预算内重试。

### 恢复规划与专家执行

规划器根据复合退化决定恢复顺序。例如，一张同时包含噪声、雾和 JPEG 伪影的图像，可以先去噪，再去雾，最后处理压缩伪影。动作空间由已注册工具组成，规划器不能生成任意命令，也不能把生成式超分插入恢复步骤。

每项恢复任务可以对应多个专家模型。当前工具池使用 SwinIR、MPRNet、Restormer、DehazeFormer 和 FBCNN，覆盖去噪、运动与散焦去模糊、去雨、去雾和压缩伪影去除；2× 目标使用受控的 SwinIR 补充步骤。执行器运行符合当前任务和图像尺寸的候选工具，再结合图文偏好分数与无参考质量分选择结果。当前配置使用 HPSv2 衡量结果与图像语义描述的匹配程度，并以 MUSIQ 补充感知质量评价。若最佳候选仍低于反思阈值，Agent 会回退到可继续搜索的先前节点，重新排列尚未完成的任务。

因此，恢复过程不是固定滤镜链，而是一棵可回退的执行树：节点保存当前图像，边表示恢复任务与所用工具，质量反馈决定继续沿当前路径执行还是回到先前状态。

### 逐尺度生成

非生成恢复及可选的 2× 补充步骤完成后，系统进入唯一的生成式超分阶段。生成器以 Stable Diffusion 3 为骨干，加载 SR Transformer LoRA 与经过适配的 VAE encoder；视觉语言模型从会话的语义锚点中提取多尺度提示，补充高倍率下逐渐变弱的语义线索。

每次调用只生成一个 4× 候选。输入图像先以 Lanczos 插值到目标像素尺寸，再编码到潜空间；带 SR LoRA 的 Transformer 完成一步条件流预测，VAE Encoder LoRA 保留低分辨率输入中的结构信息，最后由 tiled VAE decoder 还原为完整图像。大图以重叠 tile 处理，潜变量通过高斯权重流式融合，减小 tile 边界并限制中间张量的常驻规模。

多尺度提示适配器在发布前经过偏好对齐，训练信号同时考虑 critic VLM 的内容评分、无关视角短语和重复 n-gram。ScaleGuard-4K 在推理时加载该适配器，不在运行过程中训练或更新模型。

生成会话只在开始时加载模型。第一张恢复结果作为语义锚点保留到会话结束，当前可信图像则随每次接受操作更新。下一步生成前，worker 会核对输入哈希是否等于当前可信状态；候选只有收到 `accept` 后才能成为下一尺度输入，`rollback` 不会提交待定候选，但候选文件仍会保留为运行证据。

### 支持的尺度路径

| 目标倍率 | 2× 补充步骤 | 4× 生成次数 | 实际路径 |
| ---: | ---: | ---: | --- |
| 1× | 0 | 0 | 原始分辨率恢复 |
| 2× | 1 | 0 | 恢复 → 2× |
| 4× | 0 | 1 | 恢复 → 4× |
| 8× | 1 | 1 | 恢复 → 2× → 8× |
| 16× | 0 | 2 | 恢复 → 4× → 16× |

2× 补充步骤由倍率策略追加，始终位于原始分辨率恢复之后。生成式超分一次只前进 4×，最多执行两次，避免把任意倍率递归交给语言模型决定。

## 跨尺度控制

<p align="center">
  <img src="assets/figures/trusted-scale-controller.webp" width="100%" alt="ScaleGuard-4K 跨尺度控制器">
</p>

设上一可信图像为 $I_k$，新生成的 4× 候选为 $C_{k+1}$。控制器分别检查画质增益、跨尺度一致性和观测一致性，不把三项压缩成一个加权总分。

### 同尺寸画质增益

先将 $I_k$ 用确定性双三次插值放大到候选尺寸，得到基线 $B_{k+1}$：

$$
\Delta Q = Q(C_{k+1}) - Q(B_{k+1})
$$

候选与基线具有相同像素尺寸，因此差值描述的是生成结果相对普通插值带来的画质变化，而不是分辨率变化本身。

### 低通跨尺度一致性

将候选低通滤波并缩回上一可信尺寸，记为 $D(L(C_{k+1}))$。控制器分别计算 RGB 归一化重建误差和梯度域平均绝对误差：

$$
E_{rgb}=\frac{\lVert D(L(C_{k+1}))-I_k\rVert_2}
{\lVert I_k\rVert_2+\epsilon}
$$

$$
E_{edge}=\operatorname{MAE}\left(\nabla D(L(C_{k+1})),\nabla I_k\right)
$$

$E_{rgb}$ 约束低频颜色与整体结构，$E_{edge}$ 关注边缘变化。低通半径随缩放比例调整，避免直接比较高分辨率纹理和低分辨率观测。

### 观测一致性

如果实验声明了前向成像模型 $H$，系统还会把候选映射回观测空间：

$$
E_{obs}=\frac{\lVert H(C_{k+1})-Y\rVert_2}
{\lVert Y\rVert_2+\epsilon}
$$

$Y$ 是实际观测图像。项目实现了重采样、高斯 PSF、JPEG、Poisson–Gaussian 噪声和均匀雾五类前向模型。观测模型及参数由实验显式给出，不进行盲估计。

### 状态决策

| 条件 | 动作 | 结果 |
| --- | --- | --- |
| 跨尺度或观测误差超限 | `rollback` | 丢弃候选，返回上一可信状态 |
| 画质增益不足 | `stop` | 结束生成，保留上一可信状态 |
| 全部通过且尚未达到目标 | `continue` | 接受候选，继续下一尺度 |
| 全部通过且达到目标 | `stop` | 接受候选并结束生成 |
| worker 或会话失败 | `rollback` | 返回最近的可信状态 |

<p align="center">
  <img src="assets/figures/trusted-scale-state-trace.webp" width="100%" alt="ScaleGuard-4K 尺度状态变化">
</p>

尺度确定后，系统最多执行一次 AdaIN 颜色对齐，并重新计算最终图像的质量与一致性。如果颜色对齐后的结果没有通过最终检查，系统会依次尝试未经颜色处理的可信结果、上一可信尺度和恢复阶段输出。运行记录对应实际写出的图像，而不是颜色处理前的中间结果。

## 实验设计

四组实验使用相同输入快照、模型版本、生成种子和指标配置：

| 组别 | 原始分辨率恢复 | 生成式超分 | 尺度策略 | 作用 |
| --- | --- | --- | --- | --- |
| A-only | 是 | 否 | — | 观察恢复阶段 |
| B-only | 否 | 是 | 固定接受 | 观察直接生成 |
| AB-fixed | 是 | 是 | 固定接受 | 固定链路基线 |
| ScaleGuard | 是 | 是 | 动态选择 | 比较尺度控制 |

主要比较关系是 ScaleGuard 与 AB-fixed。全参考评价使用 PSNR、SSIM 和 LPIPS，无参考评价使用 MUSIQ 和 CLIPIQA；同时报告 RGB NRMSE、梯度 MAE、候选接受率、停止率和回退率。统计以输入图像为重采样单元，通过 bootstrap 给出 95% 置信区间，缺失组和失败样本不做数值填补。

当前四组配对协议覆盖单次 4× 生成。16× 支持逐样本运行，两步配对消融将单独统计第一步接受率、第二步拒绝率和跨尺度误差累积。

### 预注册的 GPU 验收门槛

| 指标 | 验收值 |
| --- | ---: |
| RGB 跨尺度误差中位数 | 相对 AB-fixed 降低 28% |
| 梯度误差中位数 | 相对 AB-fixed 降低 22% |
| 已接受 4× 输出的 PSNR 差值 | ScaleGuard − AB-fixed ≥ −0.1 dB |
| 已接受 4× 输出的 SSIM 差值 | ScaleGuard − AB-fixed ≥ −0.002 |
| 已接受 4× 输出的 LPIPS 差值 | ScaleGuard − AB-fixed ≤ +0.005 |
| 已接受 4× 输出的 MUSIQ 差值 | ScaleGuard − AB-fixed ≥ −0.5 |
| 已接受 4× 输出的 CLIPIQA 差值 | ScaleGuard − AB-fixed ≥ −0.005 |

全参考指标只比较尺寸与配准关系一致的输出。实验配置、标定方法和统计口径见 [评估协议](docs/evaluation-protocol.md)。

## 安装与运行

### CPU 编排检查

需要 Python 3.10–3.14 与 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --locked --extra dev
bash scripts/run_cpu_demo.sh
```

该命令使用 mock 后端检查 CLI、尺度状态、运行清单和产物哈希，不加载模型权重。

### 双 GPU 运行

真实模型路径面向同机双 RTX 4090。完成模型协议和凭据准备后，依次执行：

```bash
scripts/autodl/check_gpu.sh
scripts/autodl/bootstrap.sh
scripts/autodl/download_weights.sh
scripts/autodl/run_smoke.sh
scripts/autodl/run_integration.sh
```

详细环境、权重和数据要求见 [安装文档](docs/installation.md)、[部署文档](docs/autodl.md) 与 [复现指南](docs/reproduction.md)。

## 项目结构

```text
src/scaleguard/          配置、后端、尺度控制、成像模型与证据验证
third_party/overlays/    恢复 Agent 与逐尺度生成的运行时适配
configs/                 本地、双 GPU 与实验配置
scripts/autodl/          环境准备、权重下载、实跑与诊断入口
scripts/experiments/     阈值标定、四组消融与结果汇总
tests/                   单元、契约、集成与评估测试
docs/                    架构、实验协议、复现步骤与设计决策
```

## 文档

| 主题 | 位置 |
| --- | --- |
| 系统架构 | [docs/architecture.md](docs/architecture.md) |
| 配置说明 | [docs/configuration.md](docs/configuration.md) |
| 实验与标定 | [docs/evaluation-protocol.md](docs/evaluation-protocol.md) |
| 安装与复现 | [docs/installation.md](docs/installation.md)、[docs/reproduction.md](docs/reproduction.md) |
| 已知限制 | [docs/limitations.md](docs/limitations.md) |
| 设计决策 | [docs/adr](docs/adr) |

## 使用边界

跨尺度检查能够发现大范围的颜色、结构与边缘漂移，不能直接判断文字、人脸身份或语义是否正确。前向模型用于受控实验，不负责估计未知相机链路。生成式超分仍可能合成观测中不存在的细节，停止和回退只能减少这类结果继续传播的机会，不能证明生成像素的真实来源。

## 许可

本项目原创代码采用 Apache-2.0 许可。运行时模型和部分指标权重遵循各自的许可条款，其中包含非商业限制，详见 [NOTICE](NOTICE)。本仓库不分发模型权重。

## 引用与致谢

感谢以下研究工作公开论文与代码：

[4KAgent](https://github.com/taco-group/4KAgent)（[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f0075fe4e59652cf43148dcfab8d3c93-Abstract-Conference.html)）：

```bibtex
@inproceedings{zuo2025fourkagent,
  title     = {4KAgent: Agentic Any Image to 4K Super-Resolution},
  author    = {Zuo, Yushen and Zheng, Qi and Wu, Mingyang and Jiang, Xinrui and
               Li, Renjie and Wang, Jian and Zhang, Yide and Mai, Gengchen and
               Wang, Lihong V. and Zou, James and Wang, Xiaoyu and
               Yang, Ming-Hsuan and Tu, Zhengzhong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

[Chain-of-Zoom](https://github.com/bryanswkim/Chain-of-Zoom)（[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b66d8cbb01ac8212830068f3d75b4c5c-Abstract-Conference.html)）：

```bibtex
@inproceedings{kim2025chainofzoom,
  title     = {Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression
               and Preference Alignment},
  author    = {Kim, Bryan Sangwoo and Kim, Jeongsol and Ye, Jong Chul},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

相关代码与权重遵循各自的原始许可。
