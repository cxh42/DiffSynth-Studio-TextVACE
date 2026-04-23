# gpu6：Setup 并启动 121 帧训练

> 给 **gpu6（10.240.99.120）** 上的 Claude Code 的指导文档。
>
> gpu2 已经通过 scp 把训练必需的大文件传到 gpu6：
>
> - `data/` （230 样本，已解压到项目根下）
> - `models/train/TextVACE_14B_sft_49f/epoch-4.safetensors` （8 GB，Stage 1 的 49 帧 checkpoint）
>
> 你只需要配好环境、下载基础模型、启动训练。**全程只在 `/home/xinghao/` 下操作。**

---

## 前提

- 当前机器：gpu6（`10.240.99.120`），8 × NVIDIA H100 80GB
- 项目根：`/home/xinghao/DiffSynth-Studio-TextVACE`
- 代码已通过 git 同步
- miniconda3 已装在 `/home/xinghao/miniconda3/`

---

## Step 0：快速验证（先跑这个，确认传输来的文件都在）

```bash
cd /home/xinghao/DiffSynth-Studio-TextVACE

# 训练数据 230 样本
wc -l data/metadata.csv                                    # 期望 231
ls data/raw/original_videos/ | wc -l                       # 期望 230
ls data/processed/glyph_videos_tracked/ | wc -l            # 期望 230

# Stage 1 checkpoint（Stage 2 从这里加载）
ls -la models/train/TextVACE_14B_sft_49f/epoch-4.safetensors
# 期望大小：8039988200 bytes

# 关键代码修改都在
grep -l "_OffloadToCPU" diffsynth/models/wan_video_vace.py
grep -l "_load_state_dict_into_zero3_model" examples/wanvideo/model_training/train.py
ls scripts/train_textvace_14b_121f.sh
ls examples/wanvideo/model_training/full/ds_config_14B.json
```

如果任何一条不通过，停下来让用户处理。

---

## Step 1：建 conda 环境 + 装依赖

**关键：必须用环境内的 pip，否则会装错解释器**（`conda run` 可能会误用系统 python 3.10，但本仓库需要 3.12）。

```bash
CONDA_ENV=/home/xinghao/miniconda3/envs/DiffSynth-Studio

# 如果环境不存在：
conda create -n DiffSynth-Studio python=3.12 -y

# 装依赖（都用绝对路径 pip）
cd /home/xinghao/DiffSynth-Studio-TextVACE
${CONDA_ENV}/bin/pip install -e .
${CONDA_ENV}/bin/pip install accelerate==1.13.0
${CONDA_ENV}/bin/pip install deepspeed==0.18.9
```

验证：

```bash
${CONDA_ENV}/bin/python -c "
import torch, deepspeed, accelerate
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())
print('deepspeed', deepspeed.__version__)
print('accelerate', accelerate.__version__)
"
```

预期：
```
torch 2.7.0+cu128 | cuda: True | gpus: 8
deepspeed 0.18.9
accelerate 1.13.0
```

---

## Step 2：下载 Wan2.1-VACE-14B 基础模型

这一步从 HuggingFace 拉取约 60GB 模型文件到 HF 默认缓存：

```bash
${CONDA_ENV}/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-VACE-14B')
"
```

结束后建立软链（训练脚本在 `models/Wan-AI/Wan2.1-VACE-14B` 找模型）：

```bash
SNAPSHOT_HASH=$(ls ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-14B/snapshots/)
mkdir -p /home/xinghao/DiffSynth-Studio-TextVACE/models/Wan-AI
ln -sfn ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-14B/snapshots/${SNAPSHOT_HASH} \
       /home/xinghao/DiffSynth-Studio-TextVACE/models/Wan-AI/Wan2.1-VACE-14B

# 验证：应看到 diffusion_pytorch_model-*.safetensors × 7、T5、VAE
ls /home/xinghao/DiffSynth-Studio-TextVACE/models/Wan-AI/Wan2.1-VACE-14B/ | head
```

---

## Step 3：启动 121 帧训练

### 3.1 先确认 GPU 空闲

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

8 张卡都应接近 0 MiB。有占用就联系用户。

### 3.2 启动训练脚本（后台跑 + tee 日志）

```bash
cd /home/xinghao/DiffSynth-Studio-TextVACE
mkdir -p logs
bash scripts/train_textvace_14b_121f.sh 2>&1 | tee logs/train_14b_121f.log
```

**推荐用 run_in_background 启动**（这样你可以继续监控其他事）：

```bash
# background 运行
bash scripts/train_textvace_14b_121f.sh 2>&1 > logs/train_14b_121f.log &
```

### 3.3 启动后关键日志（按先后出现）

1. `Loaded model: {...}` × 32 — 模型加载（8 rank × 4 组件：DiT/VACE/T5/VAE）
2. **`Loaded 327 params from ./models/train/TextVACE_14B_sft_49f/epoch-4.safetensors (ZeRO-3 mode)`** — 49 帧 checkpoint 加载成功（最关键的那一行）
3. `Epoch 1/1:   0%|          | 0/288 [00:00<?, ?it/s]` — 训练循环进入
4. `Do not find activation_checkpointing config in deepspeed config, skip initializing...` — 正常提示（我们用 PyTorch checkpoint，不用 DeepSpeed activation checkpointing）
5. 约 5–6 分钟后第一条 loss：`0/288 [05:43<?, ?it/s, loss=...]`

### 3.4 预期指标

- 每 step：~343 秒（5 分 43 秒）
- 1 epoch = 288 steps ≈ **27 小时**
- 每卡显存：**~74 GB / 80 GB**（接近上限但稳定）
- GPU 利用率：100%（8 卡全满）

```bash
# 实时监控
watch -n 5 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
```

### 3.5 输出

训练完成后 checkpoint 在：

```
/home/xinghao/DiffSynth-Studio-TextVACE/models/train/TextVACE_14B_sft_121f/epoch-0.safetensors
```

这就是最终的 121 帧 TextVACE 14B 模型。

---

## 方法速查

详见 [TextVACE_Method.md](TextVACE_Method.md)。720P 121 帧能在 8×H100 80GB 上跑，核心技术：

- **DeepSpeed ZeRO-3 + CPU offload**：参数 + 优化器状态 offload 到 CPU
- **手动激活值 offload**：`_OffloadToCPU` + `_RestoreToGPU` + `use_reentrant=True`
- **VACE hints CPU offload**：8 个 hint × 1.1GB 暂存 CPU，按需 `.to(device)`
- **VACE block 重构**：return tuple 而非 `torch.stack`（省 ~17GB）
- **VAE 排除 ZeRO-3**：DeepSpeed 不管 VAE，避免兼容性问题

---

## 常见问题

### Q1: `RuntimeError: DeepSpeed is not installed` 但 pip 显示已装
`conda run` 可能误用系统 python。用 `${CONDA_ENV}/bin/python` 直接调，或确认 `${CONDA_ENV}/bin` 在 PATH 最前（训练脚本第 8 行已处理）。

### Q2: checkpoint 加载报错 `size mismatch ... shape torch.Size([0])`
ZeRO-3 下参数已分片。检查 `train.py` 是否用了 `_load_state_dict_into_zero3_model`：
```bash
grep _load_state_dict_into_zero3_model examples/wanvideo/model_training/train.py
```

### Q3: 第一个 step backward 时 NCCL 错误 / OOM
底层是 CUDA OOM。正常应 ≤ 74 GB / 80 GB。查以下代码是否在：
- `diffsynth/models/wan_video_vace.py` 的 `VaceWanAttentionBlock.forward` **返回** `(c_skip, c)`（不是 `torch.stack`）
- `diffsynth/pipelines/wan_video.py` 有 `_OffloadToCPU.apply(x).requires_grad_(True)` 和 `vace_hints = [h.cpu() for h in vace_hints]`

### Q4: 中断后恢复
每个 epoch 末保存 `epoch-N.safetensors`。手动 resume 方式：在脚本里加 `--model_checkpoint_path` 指向最后一个 epoch 的 safetensors。

---

## 预启动 checklist

- [ ] `nvidia-smi` 8 张 GPU 全部空闲
- [ ] `${CONDA_ENV}/bin/python -c "import torch; print(torch.cuda.device_count())"` → 8
- [ ] `${CONDA_ENV}/bin/python -c "import deepspeed; print(deepspeed.__version__)"` → `0.18.9`
- [ ] `ls models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-*.safetensors | wc -l` → 7
- [ ] `stat -c %s models/train/TextVACE_14B_sft_49f/epoch-4.safetensors` → `8039988200`
- [ ] `wc -l data/metadata.csv` → 231
- [ ] `grep _OffloadToCPU diffsynth/models/wan_video_vace.py | head -1` 有输出
- [ ] `grep _load_state_dict_into_zero3_model examples/wanvideo/model_training/train.py | head -1` 有输出

都 ✅ 就可以启动了。
