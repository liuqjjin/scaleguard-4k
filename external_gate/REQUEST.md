# External gate: AutoDL dual-4090 validation

Gate ID: `autodl-2x4090-coz-gated-v1`

## EXTERNAL GATE

需要账户或数据所有者完成：

1. 开通一台 Linux AutoDL 实例，提供两张可见的 RTX 4090（每张至少
   24000 MiB）、NVIDIA driver 560.28.03 或更新版本，以及至少 150 GiB
   可用缓存盘空间。项目 hook 会用哈希锁定 wheel 自举所需的精确 uv 版本。
2. 在 Hugging Face 账户接受项目锁文件所列 Stable Diffusion 3 gated
   模型协议。
3. 在远端交互式 shell 中以隐藏输入分别导出 `HF_TOKEN` 和
   `DASHSCOPE_API_KEY`。该 Key 必须在北京地域创建并与配置中的 endpoint
   匹配；调度器只向百炼发送文本任务标签，不发送图像。overlay 只检查
   配置指定的环境变量是否存在，
   不把 secret 值写入证据。
4. 按上游说明手动取得 DepictQA degradation delta，将它放到
   `weights/4kagent/depictqa/delta/degra_eval.pt`。上游没有发布该文件的
   digest，因此脚本只记录实测 SHA-256，不把来源认证写成已完成。
5. 提供一个已授权的 smoke 输入和一个 integration 输入；不要上传受限
   数据或私人图像。

为什么这些步骤不能由仓库自动完成：

AutoDL 购买、Hugging Face 协议接受、账户令牌和输入数据授权都属于账户或
数据所有者权限。DepictQA delta 只有上游给出的 Google Drive 对象且没有
发布 digest，需要账户持有人确认取得的对象及其使用条件。仓库脚本不会
自动接受协议，不下载该 manual 项，也不保存凭据。

最小操作步骤：

```bash
cd /path/to/scaleguard-4k
read -rsp 'HF token: ' HF_TOKEN && printf '\n'
export HF_TOKEN
read -rsp 'DashScope API key: ' DASHSCOPE_API_KEY && printf '\n'
export DASHSCOPE_API_KEY
export CUDA_VISIBLE_DEVICES=0,1
export SCALEGUARD_SMOKE_INPUT=/authorized-data/smoke.png
export SCALEGUARD_INTEGRATION_INPUT=/authorized-data/integration.png
scripts/autodl/bootstrap.sh
mkdir -p weights/4kagent/depictqa/delta
# 在已授权浏览器打开：
# https://drive.google.com/file/d/1o-PN1iXctWl62Tdb8fZs1eD1Ehv6HBMh/view
# 将合法取得的文件上传为以下精确路径：
test -s weights/4kagent/depictqa/delta/degra_eval.pt
external_gate/commands.sh
unset HF_TOKEN DASHSCOPE_API_KEY
```

不要把令牌写进 `.env`、shell 历史、命令参数、issue 或回传压缩包。
即使按上面的单 shell 流程提前导出两个凭据，gate 也会按阶段限制可见性：
GPU 检查、bootstrap、安装 hook 和源码校验看不到任何凭据；下载子进程只
收到 Hugging Face 凭据；doctor 只收到固定的非敏感“凭据存在”标记；只有
真实模型运行收到配置中 `fourkagent.api_key_env` 指定的调度器凭据。诊断
收集器先把值变成非导出 shell 变量，再通过私有文件描述符只交给脱敏器。
`external_gate/commands.sh` 内的下载阶段会调用
`scripts/weights/materialize.py`，产生并验证下载 receipt、materialization
receipt 与固定 marker。发布方没有提供该人工文件的 digest，因此项目只能
记录本地实测 SHA-256，不能认证其发布方来源。

成功验证方式：

- `gpu-check/*/gpu_check.json`（由 gate 命令显式生成）为 `passed`，并列出两个经 `nvidia-smi`
  实测的 4090、各自总显存和满足 CUDA 12.6 下限的 driver 版本；
- `weight-download/*/weights-receipt.json` 为 `passed`，包含固定 revision
  和逐文件 SHA-256；manual delta 必须标为 `recorded_manual`，不能标为
  已由上游 digest 认证；
- `bootstrap/*/runtime-receipts/validation.json` 为 `passed`，绑定最终
  commit、全部环境锁和四个隔离环境 receipt；`runtime-dependencies.yaml`
  中的 DepictQA 只作为 4KAgent 传递依赖验证；
- 同一 attempt 的 `materialization-receipt.json` 为 `passed`，与固定
  marker 内容一致，并证明权重布局未修改两个审计 checkout；
- smoke 与 integration 各自的 `runtime-environments/` 包含四份 fresh
  receipt，`runtime-preflight.json` 为 schema v2，并证明当前 distributions、
  imports 和 4KAgent 工具入口仍与 bootstrap 基线精确一致；
- smoke 与 integration 的 `execution.json` 为 `passed`，原始日志、输出哈希
  和每卡显存采样同时存在；
- `collect_diagnostics.sh` 输出压缩包及其 `.sha256`，人工检查压缩包后再回传；
- 项目的 run manifest 进一步证明所用 backend、尺度决策和是否真实执行模型。

仓库已完成的双卡路径：

- 双卡、显存、磁盘和 CUDA 可见性预检；
- 不在命令参数传令牌的固定 revision 权重下载与哈希；
- smoke/integration、显存采样、日志和证据 manifest；
- 诊断信息的白名单收集、自动脱敏和内容哈希；
- 已发布的 `RESEARCH_EVALUATED` 双卡研究结果。

复现同一路径时：

- 在实例上执行 `external_gate/commands.sh`；
- 根据真实错误修复环境或适配；
- 审阅原始日志、显存峰值和产物；
- 不要把单次脚本通过等同于整份研究复现。
