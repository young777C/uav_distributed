# ACoT-UAV-Track training container.
#
# Adapts to the host's FIXED CUDA/driver (R470 / CUDA 11.4, cannot be changed)
# WITHOUT touching the host: we ship a PyTorch built for CUDA 11.8 (cu118).
# CUDA "minor version compatibility" lets any CUDA 11.x app run on a R450+ driver,
# so cu118 runs on the R470 host driver. (The host's default torch was cu130 =
# CUDA 13 = a different major version -> needs R580+, hence it failed.)
#
# Base ships torch 2.6.0 + cu118 (torch >=2.5 is required by recent transformers for
# Qwen3-VL: it imports DTensor from torch.distributed.tensor). Still cu118 => runs on
# the R470 host driver via CUDA minor-version compatibility. A40 = sm_86 in cu118.

FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# ML deps for Stage-1..3 training + the backbone probe (PaliGemma2 / Qwen3-VL).
# transformers>=4.57 is required for Qwen3-VL; PaliGemma2 is supported well before it.
RUN pip install --no-cache-dir \
        "transformers>=4.57" \
        "accelerate>=0.34" \
        "h5py>=3.10" \
        "pillow>=10.0" \
        "pyyaml>=6.0" \
        "sentencepiece" \
        "safetensors" \
        "einops"

WORKDIR /workspace/uav-acot-track

# Repo + data are bind-mounted at run time (see run.sh) so code edits need no rebuild.
# Default command: report the torch/CUDA/GPU status (sanity check on R470).
CMD ["python", "-c", "import torch;print('torch',torch.__version__,'| cuda_avail',torch.cuda.is_available(),'| gpus',torch.cuda.device_count())"]
