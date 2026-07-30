# 训练容器（适配固定的宿主 CUDA，不改宿主）

宿主驱动 **R470 / CUDA 11.4（不可更改）**。容器通过 **CUDA 次版本兼容性**适配：镜像内装 **cu118（CUDA 11.8）的 PyTorch**，CUDA 11.x 应用可运行在 R450+ 驱动上 → 在 R470 上正常使用 8×A40。**不触碰宿主任何 CUDA 配置。**

> 之前宿主 `torch` 是 cu130（CUDA **13**，跨大版本 → 需 R580+）才报 "driver too old"。容器换 cu118 即解决。

## 一次性准备

```bash
# 1) 构建镜像（不动宿主 CUDA；下载 pytorch/pytorch:2.4.1-cuda11.8 基础镜像）
bash train/docker/build.sh

# 2) 验证能用上 A40（期望 cuda_avail True, gpus 8）
bash train/docker/run.sh

# 3) 门控模型需先登录 HF 并接受许可（PaliGemma 是 gated 模型）
export HF_TOKEN=hf_xxx           # run.sh 会透传进容器
#    并在 huggingface.co 上接受 google/paligemma2-3b-mix-448 的许可
```

## Stage-1 训练流程（数据就绪后）

```bash
# A) 骨干选型探针（Phase B，先定 PaliGemma vs Qwen / 截断层）
bash train/docker/run.sh python -m acot_probe.run_probe --config acot_probe/config.yaml

# B) 预计算并缓存 VLM layer-L 上下文（backbone 依探针结论，默认 PaliGemma-2-3B）
bash train/docker/run.sh python -m train.backbone_kv --config train/config.yaml

# C) Stage-1 EAR 预热训练
bash train/docker/run.sh python -m train.stage1_ear --config train/config.yaml
```

## 说明

- 仓库与数据以**相同绝对路径**挂载，配置无需改动即可在容器内运行。
- **数据以只读（`:ro`）挂载**，训练绝不会干扰数据生成。
- 代码是 bind-mount（非 COPY），改代码无需重建镜像。
- HF 缓存挂到 `~/.cache/huggingface`，模型只下载一次。
- 若 `--gpus all` 不生效（nvidia runtime 未注册），`run.sh` 会回退 `--runtime=nvidia`。

## 自动化

数据生成完成检测用 `python -m train.check_data_ready`（**只读目录元数据、不打开任何 h5**）。就绪后按上面 A→B→C 进入训练。
