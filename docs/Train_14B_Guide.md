# TextVACE 14B 训练指南

> 更新日期：2026-04-15
> 硬件环境：8x NVIDIA H100 80GB HBM3, 2TB CPU RAM
> Conda 环境：DiffSynth-Studio (Python 3.12)

---

## 1. 环境准备

### 1.1 已安装依赖

```bash
conda activate DiffSynth-Studio
pip install -e .        # DiffSynth-Studio
pip install accelerate  # v1.13.0
pip install deepspeed   # v0.18.9
```

### 1.2 模型下载

14B VACE 模型已从 HuggingFace 下载：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.1-VACE-14B')"
```

**本地路径**：`~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-14B/snapshots/539c162b1387eac9dc4c20bd3f74671309e76a4c/`

为避免框架从 ModelScope 重新下载，创建了软链接：

```bash
mkdir -p models/Wan-AI
ln -s ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-14B/snapshots/539c162b1387eac9dc4c20bd3f74671309e76a4c \
      models/Wan-AI/Wan2.1-VACE-14B
```

**关键环境变量**（在训练脚本中设置）：

```bash
export DIFFSYNTH_MODEL_BASE_PATH="/home/xinghao/DiffSynth-Studio-TextVACE/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true  # 跳过远程下载，使用本地文件
```

### 1.3 训练数据

从 Google Drive 下载并解压 `data_textvace.tar.gz`（1.4GB）到 `data/` 目录。

**数据结构**：
```
data/
├── raw/
│   ├── original_videos/          (230 个原始视频, 1280x720, 24fps, 120帧)
│   ├── edited_videos/            (230 个编辑后视频)
│   ├── text_masks/               (230 个掩码视频)
│   └── edit_instructions/
├── processed/
│   ├── glyph_videos_tracked/     (230 个 glyph 视频)
│   └── parsed_records.json       (解析后记录，含 target_text)
└── metadata.csv                  (训练用，已重新生成)
```

---

## 2. 数据准备变更

### 2.1 metadata.csv 重新生成

**变更**：prompt 从完整编辑指令（如 "Change ATP to NAD"）改为只使用 `target_text`（如 "NAD"）。

glyph 视频使用 `processed/glyph_videos_tracked/` 目录。

**生成脚本**：

```python
import json, csv

with open('data/processed/parsed_records.json') as f:
    records = json.load(f)

rows = []
for r in records:
    rows.append({
        'video': f'raw/{r["edited_video"]}',
        'vace_video': f'raw/{r["original_video"]}',
        'vace_video_mask': f'raw/{r["mask_video"]}',
        'glyph_video': f'processed/glyph_videos_tracked/{r["id"]}.mp4',
        'prompt': r['target_text']  # 只用目标文字
    })

with open('data/metadata.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['video', 'vace_video', 'vace_video_mask', 'glyph_video', 'prompt'])
    writer.writeheader()
    writer.writerows(rows)
```

**metadata.csv 格式**：
```
video,vace_video,vace_video_mask,glyph_video,prompt
raw/edited_videos/0000007_00000_overlay.mp4,raw/original_videos/0000007_00000.mp4,raw/text_masks/0000007_00000.mp4,processed/glyph_videos_tracked/0000007_00000.mp4,NAD
```

---

## 3. 代码修改

### 3.1 model_configs.py — 14B 添加 glyph_channels

**文件**：`diffsynth/configs/model_configs.py` (第244行)

为 14B VACE 模型配置（hash `7a513e1f257a861512b1afd387a8ecd9`）添加 `'glyph_channels': 16`：

```python
"extra_kwargs": {
    'vace_layers': (0, 5, 10, 15, 20, 25, 30, 35),
    'vace_in_dim': 96,
    'glyph_channels': 16,  # ← 新增
    'patch_size': (1, 2, 2),
    'has_image_input': False,
    'dim': 5120,
    'num_heads': 40,
    'ffn_dim': 13824,
    'eps': 1e-06
},
```

### 3.2 train.py — 禁用文件重定向

**文件**：`examples/wanvideo/model_training/train.py` (第37行)

添加 `redirect_common_files=False` 防止框架将 T5/VAE 重定向到 ModelScope 下载：

```python
self.pipe = WanVideoPipeline.from_pretrained(
    ...,
    redirect_common_files=False  # ← 新增
)
```

### 3.3 LoadVideo — 短视频补帧

**文件**：`diffsynth/core/data/operators.py` (LoadVideo.__call__)

修改 `LoadVideo` 使其在视频帧数不够时，用最后一帧填充到目标帧数（而非减少采样帧数）：

```python
def __call__(self, data: str):
    reader = self.get_reader(data)
    raw_frame_rate = reader.get_meta_data()['fps']
    total_raw_frames = reader.count_frames()
    total_available = self.get_available_num_frames(reader)
    num_frames = self.num_frames  # 不再减少
    frames = []
    for frame_id in range(num_frames):
        if frame_id < total_available:
            raw_id = self.map_single_frame_id(frame_id, raw_frame_rate, total_raw_frames)
            frame = reader.get_data(raw_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        else:
            frames.append(frames[-1])  # 补最后一帧
    reader.close()
    return frames
```

**场景**：训练数据视频为 120 帧，训练配置 121 帧时，最后一帧会被复制一次。

### 3.4 model_loader.py — ZeRO-3 VAE 排除 + 加载修复

**文件**：`diffsynth/core/loader/model.py`

#### 3.4.1 VAE 跳过 ZeRO-3 初始化

```python
def load_model(...):
    # Skip ZeRO-3 initialization for VAE to avoid compatibility issues
    skip_zero3 = 'vae' in model_class.__name__.lower() if hasattr(model_class, '__name__') else False
    with ContextManagers(get_init_context(torch_dtype=torch_dtype, device=device, skip_zero3=skip_zero3)):
        model = model_class(**config)
```

#### 3.4.2 get_init_context 添加 skip_zero3 参数

```python
def get_init_context(torch_dtype, device, skip_zero3=False):
    if is_deepspeed_zero3_enabled() and not skip_zero3:
        # 正常 ZeRO-3 初始化（参数分片）
        init_contexts = [deepspeed.zero.Init(remote_device=device, dtype=torch_dtype), set_zero3_state()]
    elif skip_zero3:
        # VAE 等模型：正常初始化，不分片
        init_contexts = []
    else:
        init_contexts = [skip_model_initialization()]
    return init_contexts
```

#### 3.4.3 VRAM management 分支支持 ZeRO-3

在 `module_map is not None` 分支中，也使用 `_load_state_dict_into_zero3_model`：

```python
if is_deepspeed_zero3_enabled():
    from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model
    _load_state_dict_into_zero3_model(model, state_dict)
else:
    model.load_state_dict(state_dict, assign=True)
```

### 3.5 runner.py — VAE 排除 DeepSpeed 管理

**文件**：`diffsynth/diffusion/runner.py` (launch_training_task)

在 `accelerator.prepare()` 前将 VAE 从 `_modules` 中移除，之后通过 `object.__setattr__` 放回（不触发 nn.Module 的子模块注册）：

```python
# 排除 VAE
vae_module = getattr(model.pipe, 'vae', None)
if vae_module is not None:
    del model.pipe._modules['vae']

model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

# 放回 VAE（不注册为子模块）
if vae_module is not None:
    vae_module.to(accelerator.device)
    pipe = model.module.pipe if hasattr(model, 'module') else model.pipe
    object.__setattr__(pipe, 'vae', vae_module)
```

**原因**：DiffSynth-Studio 的 Wan VAE 和 DeepSpeed ZeRO-3 不兼容。将 VAE 排除后：
- DiT + VACE + T5 参数通过 ZeRO-3 分片到 8 卡
- VAE（~0.3GB）在每卡独立保存，不被 DeepSpeed 管理

---

## 4. 训练配置

### 4.1 训练脚本

**文件**：`scripts/train_textvace_14b.sh`

```bash
#!/bin/bash
# TextVACE v2 (Glyph) - 14B Model Training
# 8x H100 80GB, DeepSpeed ZeRO-3, 720P, 121 frames

set -e

export PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH="/home/xinghao/DiffSynth-Studio-TextVACE/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_DEEPSPEED_ZERO3_INIT=false

find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/ \
  --dataset_metadata_path data/metadata.csv \
  --data_file_keys "video,vace_video,vace_video_mask,glyph_video" \
  --height 720 \
  --width 1280 \
  --num_frames 121 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
  --tokenizer_path "models/Wan-AI/Wan2.1-VACE-14B/google/umt5-xxl" \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/TextVACE_14B_sft" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_video_mask,glyph_video" \
  --use_gradient_checkpointing
```

### 4.2 Accelerate 配置

**文件**：`examples/wanvideo/model_training/full/accelerate_config_14B.yaml`

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: examples/wanvideo/model_training/full/ds_config_14B.json
  zero3_init_flag: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 4.3 DeepSpeed 配置

**文件**：`examples/wanvideo/model_training/full/ds_config_14B.json`

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "none" },
    "offload_param": { "device": "none" },
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "activation_checkpointing": {
    "partition_activations": false,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": false
  }
}
```

**关键设计**：
- **ZeRO-3**：模型参数分片到 8 卡（每卡 ~11GB，而非 ZeRO-2 的 ~61GB）
- **activation_checkpointing**：使用 DeepSpeed 自己的激活检查点（兼容 ZeRO-3），`cpu_checkpointing: true` 将激活值卸载到 CPU
- **gradient_accumulation_steps: 8**：等效 batch size = 8 GPUs × 1 micro_batch × 8 accum = 64

---

## 5. 已验证的进展

### 5.1 模型加载 ✅

- 8 个 rank 全部成功加载 DiT + VACE + T5 + VAE（32 次 "Loaded model"）
- ZeRO-3 分片后每卡仅占 ~11GB（相比 ZeRO-2 的 ~61GB）
- VAE 成功排除在 ZeRO-3 之外

### 5.2 训练循环启动 ✅

- 训练循环已成功进入 Epoch 1/5, 288 steps
- DeepSpeed 激活检查点已配置（`activation_checkpointing` 在 ds_config 中）

### 5.3 待解决的问题

#### 问题 1：gradient checkpointing + ZeRO-3 兼容性

**现象**：使用 PyTorch 的 `torch.utils.checkpoint` 时，backward 阶段报错：
```
CheckpointError: Recomputed values for the following tensors have different metadata than during the forward pass.
```

**原因**：ZeRO-3 在 forward 和 recomputation 时参数的分片/聚合状态不一致。

**已尝试的解决方案**：
- 在 `ds_config_14B.json` 中配置 `activation_checkpointing`
- 框架的 `gradient_checkpoint_forward()` 函数（`diffsynth/core/gradient/gradient_checkpoint.py` 第37行）在 `deepspeed.checkpointing.is_configured()` 为 True 时会自动切换到 DeepSpeed 的 checkpoint 实现

**下一步**：
- 确认 DeepSpeed 的 activation checkpointing 是否被正确初始化（日志应不再显示 "Do not find activation_checkpointing config"）
- 如果 DeepSpeed checkpoint 仍不兼容，可尝试完全禁用 gradient checkpointing（ZeRO-3 下每卡只用 11GB，有 ~68GB 余量可能足够）

#### 问题 2：完全禁用 gradient checkpointing

框架在 `train.py` 第29-31行强制开启 gradient checkpointing：
```python
if not use_gradient_checkpointing:
    warnings.warn("...")
    use_gradient_checkpointing = True
```

如需完全禁用，需修改此处逻辑。

---

## 6. 显存分析

### ZeRO-2 vs ZeRO-3 对比

| 配置 | 每卡模型显存 | 720P+121帧可行 |
|------|-------------|---------------|
| ZeRO-2 (无offload) | ~61 GB | ❌ OOM |
| ZeRO-2 + CPU offload | ~61 GB (optimizer在CPU) | ❌ OOM |
| ZeRO-2 + FP8 | ~72 GB | ❌ OOM |
| **ZeRO-3** | **~11 GB** | **✅ 可能** |

### 720P+121帧的 Latent 大小

VAE 下采样 8x 空间 + 4x 时间：
- Latent: 1280/8 × 720/8 × (121-1)/4+1 = 160 × 90 × 31
- 每个 token 维度 5120 (14B)
- 单个 tensor: 160 × 90 × 31 × 5120 × 2 bytes ≈ **4.26 GB**

---

## 7. 关键架构理解

### 7.1 DiffSynth-Studio 的 ZeRO-3 集成

框架在 `diffsynth/core/loader/model.py` 中自行调用 `deepspeed.zero.Init()` 和 `_load_state_dict_into_zero3_model()`，不完全依赖 accelerate 的 `zero3_init_flag`。`is_deepspeed_zero3_enabled()` 来自 transformers 库。

### 7.2 VAE 和 ZeRO-3 不兼容

DiffSynth-Studio 的 Wan VAE 实现与 DeepSpeed ZeRO-3 存在兼容性问题。解决方案是将 VAE 排除在 ZeRO-3 管理之外：
1. 创建时跳过 `deepspeed.zero.Init()` context
2. 从 `_modules` 中移除，使 DeepSpeed 不注册 forward hooks
3. 通过 `object.__setattr__` 放回，让 pipeline 代码正常访问

### 7.3 gradient checkpointing 三种实现

`diffsynth/core/gradient/gradient_checkpoint.py` 中的 `gradient_checkpoint_forward()` 支持三种模式：

1. **DeepSpeed checkpoint**（`deepspeed.checkpointing.is_configured()` 为 True 时）— 兼容 ZeRO-3
2. **PyTorch checkpoint + CPU offload**（`use_gradient_checkpointing_offload=True`）— 不兼容 ZeRO-3
3. **PyTorch checkpoint**（`use_gradient_checkpointing=True`）— 不兼容 ZeRO-3

使用 ZeRO-3 时必须用方案 1，需要在 DeepSpeed config 中配置 `activation_checkpointing`。

---

## 8. 启动训练

```bash
# 清除缓存
find . -name "__pycache__" -exec rm -rf {} +

# 启动
conda activate DiffSynth-Studio
bash scripts/train_textvace_14b.sh 2>&1 | tee logs/train_14b.log

# 监控
tail -f models/train/TextVACE_14B_sft/training_log.txt
nvidia-smi -l 5
```

---

## 9. 修改文件汇总

| 文件 | 修改内容 |
|------|---------|
| `diffsynth/configs/model_configs.py` | 14B VACE 添加 `glyph_channels: 16` |
| `diffsynth/core/data/operators.py` | LoadVideo 短视频补最后一帧 |
| `diffsynth/core/loader/model.py` | VAE 跳过 ZeRO-3 init + VRAM management 分支支持 ZeRO-3 加载 |
| `diffsynth/diffusion/runner.py` | VAE 排除 DeepSpeed 管理 |
| `examples/wanvideo/model_training/train.py` | 添加 `redirect_common_files=False` |
| `examples/wanvideo/model_training/full/accelerate_config_14B.yaml` | ZeRO-3 + DeepSpeed config file |
| `examples/wanvideo/model_training/full/ds_config_14B.json` | 新建：ZeRO-3 + activation_checkpointing |
| `scripts/train_textvace_14b.sh` | 新建：14B 训练脚本 |
| `data/metadata.csv` | 重新生成：prompt 使用 target_text |
