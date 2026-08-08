<div align="center">

# ScaleGuard-4K

**面向复杂退化的跨尺度一致性高分辨率恢复智能体**

[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/liuqjjin/scaleguard-4k/actions/workflows/ci.yml/badge.svg)](https://github.com/liuqjjin/scaleguard-4k/actions/workflows/ci.yml)

在复杂退化图像的高倍恢复中，把放大过程建模为显式的尺度状态，并在每次尺度跃迁上判断继续、停止还是回退。

</div>

> **当前状态：`STATIC_READY`。** CPU 路径、契约与部署入口已就绪，尚无项目自产的 GPU 画质、时延或消融结果。详细边界见 [docs/results/STATUS.md](docs/results/STATUS.md)。

## 问题

真实图像上的退化往往是复合的：噪声、模糊、压缩伪影、雾与低光同时出现，且退化参数未知。恢复这类图像并放大到高分辨率时，有两个困难。

**感知质量与保真度可能不一致。** 生成式超分能合成看起来合理的纹理，无参考感知指标也会给出更高的分数。但这些细节未必来自观测，也未必与原始内容一致。当放大被递归执行时，每一步引入的偏差会被下一步继续放大，最终结果可能在感知指标上更好，在结构、文字与颜色上却已经偏离。

**固定倍率级联无法表达这个过程。** 把 16× 写成"连续两次 4×"是一条没有分支的路径：它不区分哪一步的输出值得信任，也没有位置可以在中途停下或退回。要判断"放大到哪一步为止仍然可信"，需要先把尺度变成可以被检查和撤销的状态。

ScaleGuard-4K 处理的就是这个判断问题。

## 核心贡献

**显式尺度状态与三动作裁决。** 每一次尺度跃迁产生一个候选状态，由控制器裁决 `continue`、`stop` 或 `rollback`。可信状态只在裁决通过后才前进，被拒绝的候选不进入后续步骤。裁决过程是确定性的，不依赖学习模型。

**三类独立设界的判据。** 同分辨率质量增益、低通跨尺度一致性、显式前向退化模型下的观测一致性，各自设定阈值并独立否决。三者不合成单一分数，因此一个维度的劣化不会被另一个维度的提升抵消。

**有状态的恢复智能体。** 退化感知、结构化任务规划、专家恢复、质量反馈与尺度控制组成一条带反馈的流水线。生成式超分被固定在终端位置，尺度递归由控制器管理而非由规划器自由编排。

**面向计算成像的实验设计。** 五类前向成像算子支持受控合成实验；阈值由与评估集不相交的划分标定，以输入图像聚类为重采样单元做 bootstrap；四组配对实验分别估计各模块贡献与门控策略的效果。

## 方法总览

```mermaid
flowchart TD
    I["退化输入"] --> P["退化感知"]
    P --> S["结构化恢复规划"]
    S --> E["专家恢复"]
    E --> Q{"质量反馈"}
    Q -->|"未达标"| S
    Q -->|"达标"| T["可信基础状态"]
    T --> C["终端生成式超分：一次候选"]
    C --> G{"跨尺度裁决"}
    G -->|"continue"| C
    G -->|"rollback"| K["回退到上一可信状态"]
    G -->|"stop"| F["颜色对齐与最终复评"]
    K --> F
    F --> O["输出与运行清单"]
```

输入先经过退化感知与恢复规划，由专家工具在原生分辨率上处理复合退化，质量反馈决定是否需要重新规划。得到可信基础状态后进入终端阶段：生成式超分每次只产生一个候选，控制器对该候选做跨尺度裁决，通过则提升为新的可信状态，否则回退。达到目标尺度或触发停止条件后，做一次颜色对齐并对最终写出的图像重新评分。

## 跨尺度控制

三类判据在每个候选上分别计算。

**同分辨率质量增益。** 把上一可信状态用确定性插值放大到候选的像素尺寸得到基线，只在同一分辨率下比较两者的质量分：

$$\Delta Q = Q(\text{候选}) - Q(\text{基线})$$

这样比较回答的是"生成式超分是否优于朴素插值"。不同尺寸的两张图直接比较并解释为质量提升，在方法上不成立。评估器的方向被统一为越大越好。

**低通跨尺度一致性。** 把候选低通滤波并缩回上一可信尺寸，在 RGB 与梯度两个域分别度量与上一可信状态的差异，各自设定上界。这一判据约束的是低频颜色、整体结构与边缘的漂移。它不度量文字正确性、人脸身份或语义一致性，这些需要单独的失败分析。

**观测一致性。** 当实验显式配置了前向退化算子时，把候选映射回观测空间与实际观测比较。项目提供五类算子：重采样、高斯 PSF、JPEG 压缩、Poisson–Gaussian 光子噪声与均匀雾。算子及其参数必须由实验协议声明，系统不做盲估计。每个算子导出包含完整参数的规范身份，因此一份标定结果不能跨参数复用。

三类判据并列设界而不加权求和。未经标定的指标合成单一分数时，一个维度的劣化会被另一个维度的提升掩盖，而这正是需要检测的情况。裁决顺序为：跨尺度或观测判据超限则回退，质量增益不足则停止，全部通过则继续或在达到目标时接受。

尺度策略是离散的。支持 1×、2×、4×、8×、16×，终端生成式超分最多两步。2× 与 8× 路径包含一次受控 2× bridge，作为终端超分之外的唯一放大例外。实现细节、滤波参数与阈值配置见 [docs/architecture.md](docs/architecture.md) 与 [docs/configuration.md](docs/configuration.md)。

## 智能体工作流

系统的动作空间是结构化且受约束的。规划器输出的是一个任务序列，不是自由文本指令；每个任务对应一个已注册的专家工具。

**约束放大的位置。** 规划器不能把任何放大操作排入恢复计划。生成式超分只在终端阶段由控制器发起，受控 2× bridge 只能由倍率策略追加且全局至多一次。执行完成后，实际执行路径会被独立核对，出现计划外的放大即判定违约。这条约束保证了复合退化在生成之前被处理，而不是被生成式模型顺带覆盖。

**状态管理与回退。** 恢复阶段的质量反馈可以触发重新规划。终端阶段的每个候选都记录其输入状态、产出状态、三类判据的取值与裁决理由。回退是显式动作：被拒绝的候选不会成为下一步的输入，控制器返回上一可信状态。

**阶段化驻留。** 恢复阶段与终端生成阶段由同一个生命周期实例串行管理，两者不重叠。感知服务在其所属阶段内运行，在线质量反馈在 CPU 上执行，学习型指标的完整评估放在运行结束后离线进行。这一设计让重量级模型的驻留时间可控。

**最终输出的一致性。** 颜色对齐之后会对最终写出的图像重新评分，因此运行清单中记录的分数对应实际输出的字节，而不是对齐之前的中间结果。

## 实验协议

四组配对实验共享同一份输入快照、同一随机种子与同一指标版本。

| 组 | 恢复 | 目标 | 终端步 | 裁决 | 用途 |
| --- | --- | ---: | ---: | --- | --- |
| A-only | 完整 | 1× | 0 | 固定 | 估计恢复模块在原生分辨率上的贡献 |
| B-only | 跳过 | 4× | 1 | 固定 | 估计不做恢复直接生成的代价 |
| AB-fixed | 完整 | 4× | 1 | 固定 | 门控策略的直接对照 |
| ScaleGuard | 完整 | 4× | 1 | 门控 | 门控策略 |

ScaleGuard 与 AB-fixed 是门控效果的主要比较关系，两组只在裁决策略上不同。A-only 与 B-only 用于估计各模块的贡献，它们与前两组之间还存在其他差异，因此不构成严格的单因素对照。A-only 输出在原生分辨率上，不会被放大去凑 4× 的配对，该组在 4× 目标下的全参考指标标记为不适用。缺失的组不允许用其他组的输出顶替，也不插补指标。

阈值在与评估集不相交的划分上标定：人工标注可接受的尺度步，以输入图像聚类为重采样单元做 bootstrap，得到质量下分位与误差上分位。标定结果以收据形式记录，验证时从记录的源数据重跑标定并逐字段比对。签入的运行时配置没有绑定标定收据，其中的阈值是操作性默认值，用它们运行的结果可以支撑组件复现，不足以支撑门控有效性的结论。

指标包括全参考的 PSNR、SSIM、LPIPS，无参考的 MUSIQ、CLIPIQA，控制器判据的取值，以及接受、停止与回退的比率。协议、标定流程与指标定义见 [docs/evaluation-protocol.md](docs/evaluation-protocol.md)。

## 快速开始

需要 Python 3.10–3.14 与 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --locked --extra dev
bash scripts/run_cpu_demo.sh
```

演示在 CPU 上走通完整编排：生成确定性测试图，运行命令行入口，校验产出的运行清单与产物哈希。它不加载任何模型，所有派生产物标记为 mock，用于验证契约而非产生画质结论。每次运行使用独立的临时目录，不向仓库写入内容。

真实运行需要双 GPU 环境与若干外部授权资源，流程见 [docs/installation.md](docs/installation.md) 与 [docs/autodl.md](docs/autodl.md)。

## 文档

| 主题 | 位置 |
| --- | --- |
| 架构与设计决策 | [docs/architecture.md](docs/architecture.md)、[docs/adr](docs/adr) |
| 配置字段 | [docs/configuration.md](docs/configuration.md) |
| 实验协议与标定 | [docs/evaluation-protocol.md](docs/evaluation-protocol.md) |
| 复现步骤 | [docs/reproduction.md](docs/reproduction.md) |
| 证据状态 | [docs/results/STATUS.md](docs/results/STATUS.md) |
| 已知限制 | [docs/limitations.md](docs/limitations.md) |
| 安全与隐私 | [SECURITY.md](SECURITY.md) |

## 限制

跨尺度判据基于低通重建误差与梯度差异，能够检测大范围的颜色与结构漂移，不能保证文字、人脸身份或语义的正确性。前向算子是简化模型，用于受控实验，不适用于未知采集链路的盲反演。终端生成式超分可能合成观测中不存在的细节，裁决能拒绝部分不一致的候选，但无法证明每个像素的来源。完整的限制说明见 [docs/limitations.md](docs/limitations.md)。

## 许可

本项目原创代码采用 Apache-2.0 许可。运行时依赖的模型权重与部分指标实现有各自的许可条款，其中包含非商业限制，详见 [NOTICE](NOTICE)。本仓库不分发任何模型权重。

## 引用与致谢

本项目的恢复规划与专家工具编排参考并调用了 4KAgent（[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f0075fe4e59652cf43148dcfab8d3c93-Abstract-Conference.html)，[代码](https://github.com/taco-group/4KAgent)）：

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

终端生成式超分调用了 Chain-of-Zoom（[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b66d8cbb01ac8212830068f3d75b4c5c-Abstract-Conference.html)，[代码](https://github.com/bryanswkim/Chain-of-Zoom)）：

```bibtex
@inproceedings{kim2025chainofzoom,
  title     = {Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression
               and Preference Alignment},
  author    = {Kim, Bryan Sangwoo and Kim, Jeongsol and Ye, Jong Chul},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

感谢两个项目的作者开放各自的实现。它们的代码与权重遵循原始许可，本仓库不做再分发。
