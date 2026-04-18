# TextVACE：基于 VACE 的视频文本编辑方法

> 本文档描述对 Wan2.1-VACE-14B 的架构修改、数据设计、训练策略与工程优化，用于论文准备。

---

## 1. 任务定义

**视频文本编辑（Video Text Editing）**：给定一段包含文本的视频（如招牌、字幕、物体上的文字），将视频中的文本替换为任意目标文本（多语言、符号、公式等），同时保持：
- 文本的空间/时间一致性（字形在跨帧间稳定）
- 背景、光照、物体形态不变
- 文本的视觉属性（字体、颜色、风格）与原视频自然协调

**输入**：
1. 原视频 $V_{\text{orig}}$（含需要替换的文本）
2. 文本区域掩码视频 $M$（标注出文本位置）
3. 字形参考视频 $V_{\text{glyph}}$（仅含目标文本字形，便于模型获取精确字形）
4. 目标文本字符串 $s$（作为 prompt）

**输出**：编辑后的视频 $V_{\text{edit}}$，其中原文本位置被替换为 $s$ 所对应的文字。

---

## 2. 数据设计

### 2.1 数据规模与格式

- **样本数**：230 个训练对
- **视频规格**：1280×720@24fps，120 帧（训练时补至 121 帧）
- **文件**：`data/metadata.csv`

```csv
video,vace_video,vace_video_mask,glyph_video,prompt
raw/edited_videos/0000007_00000.mp4,raw/original_videos/0000007_00000.mp4,raw/text_masks/0000007_00000.mp4,processed/glyph_videos_tracked/0000007_00000.mp4,NAD
```

五个字段：
| 字段 | 作用 |
|------|------|
| `video` | 编辑后视频（训练目标） |
| `vace_video` | 原始视频（VACE 条件输入） |
| `vace_video_mask` | 文本区域二值掩码 |
| `glyph_video` | 字形视频（新增分支） |
| `prompt` | 目标文本字符串 |

### 2.2 与原始 VACE 的数据差异

原 Wan-VACE 使用完整编辑指令作为 prompt（如 "Change ATP to NAD"），我们改为 **仅使用目标文本**（如 "NAD"）。这样 prompt 更短、更集中于字形信息。

### 2.3 字形视频生成

字形视频 $V_{\text{glyph}}$ 通过以下流程生成：
1. 从原视频中估计文本区域的时空轨迹
2. 在黑色背景上，用标准字体渲染目标文本的字形
3. 按轨迹将字形动画对齐到视频时空位置
4. 输出与原视频同分辨率/帧数的字形视频

这种"跟踪字形"（`glyph_videos_tracked`）相比静态字形图像，提供了 **跨帧一致的几何引导**，显著降低字形的时序闪烁。

---

## 3. 模型架构修改

基础模型：**Wan2.1-VACE-14B**（`dim=5120`, `num_heads=40`, `ffn_dim=13824`, 40 DiT blocks, 8 VACE blocks 位于 `layers=(0, 5, 10, 15, 20, 25, 30, 35)`）。

### 3.1 双路径字形条件注入

我们引入两条并行的字形条件路径，可独立启用：

#### 路径 1：GlyphEncoder（v2 模式，视觉字形）

将字形视频经过 VAE 编码得到的 `glyph_latent: (B, 16, T, H, W)` 压缩为定长 token 序列：

```
glyph_latent (16ch)
    ↓ Conv3D PatchEmbed (stride=(1,2,2))
    ↓ flatten → (B, N, dim)
    ↓ CrossAttention Pooling (64 learnable query tokens)
    ↓ LayerNorm + zero-init Linear
glyph_tokens: (B, 64, 5120)
```

**设计要点**：
- **可学习 query tokens**（64 个）通过 cross-attention 从可变长度的空间特征中池化出定长表示，便于后续所有 VACE block 共享使用。
- **输出投影零初始化**（`out_proj`）：确保训练开始时 GlyphEncoder 的输出为 0，模型从预训练 VACE 行为起步，逐步学习使用字形信息，避免破坏预训练能力。

#### 路径 2：TargetTextEncoder（v3 模式，字符级文本）

对目标文本字符串进行字符级编码，提供 T5 subword 分词无法保证的字符精度：

```
text "NAD"
    ↓ character hashing (ord(ch) % 8191 + 1, PAD=0)
    ↓ token_ids: (B, 64)
    ↓ char_embed + pos_embed
    ↓ 2-layer Transformer Encoder
    ↓ zero-init Linear
text_tokens: (B, 64, 5120)
```

**设计要点**：
- **字符级模运算哈希**（vocab_size=8192）：对任意 Unicode 字符（ASCII、CJK、西里尔、数学符号）都能 tokenize，无 OOV 问题。
- **padding mask** 防止注意力污染到补零位置。

### 3.2 ConditionCrossAttention

在每个 VACE block 内注入一个轻量的 cross-attention 层，以 VACE 的隐状态为 query，以字形/文本 tokens 为 key/value：

```
residual = x
x = LayerNorm(x)
q = Linear(x)                    # VACE hidden state → query
k = Linear(condition_tokens)     # glyph/text → key
v = Linear(condition_tokens)     # glyph/text → value
out = FlashAttention(q, k, v)
return residual + zero_init_Linear(out)
```

**零初始化输出投影**：训练开始时 ConditionCrossAttention 等价于恒等映射，保持预训练 VACE 行为，**残差式逐步引入**字形条件。

### 3.3 VACE Block 前向的结构性重构

**原 Wan-VACE 实现**（存在显存瓶颈）：

```python
def forward(self, c, x, context, t_mod, freqs):
    # 依赖"累积-unbind-stack"结构跨 block 传递所有 hints
    all_c = list(torch.unbind(c))  # c 包含之前所有 block 的 c_skip
    c = all_c.pop(-1)
    c = super().forward(c, context, t_mod, freqs)
    c_skip = self.after_proj(c)
    all_c += [c_skip, c]
    c = torch.stack(all_c)         # 随 block 数增长，720P 121帧下累积至 ~17GB
    return c
```

**我们的重构**（返回 tuple，外部收集）：

```python
def forward(self, c, x, context, t_mod, freqs, condition_tokens=None):
    if self.block_id == 0:
        c = self.before_proj(c) + x
    c = super().forward(c, context, t_mod, freqs)
    # 条件 cross-attention（新增）
    if condition_tokens is not None and hasattr(self, 'condition_cross_attn'):
        c = self.condition_cross_attn(c, condition_tokens)
    c_skip = self.after_proj(c)
    return c_skip, c   # 不再 stack
```

在 `VaceWanModel.forward` 中用 list 收集 hints：

```python
hints = []
for block in self.vace_blocks:
    c_skip, c = gradient_checkpoint_forward(block, ...)
    hints.append(c_skip)
return hints
```

这一重构在 720P 121 帧下节省 **~17GB GPU 显存**，是能在 8×H100 80GB 上训练的关键之一。

### 3.4 新模块的参数初始化

当 `_has_new_modules()` 为真时（`glyph_channels > 0` 或 `use_target_text_encoder=True`），`load_state_dict` 执行特殊处理：
1. 用 `strict=False` 加载预训练 VACE 权重（忽略新模块缺失键）
2. 对新模块应用 **Xavier 初始化** + **输出投影零初始化**
3. 将所有新模块 `.to(dtype=torch.bfloat16)` 对齐模型精度

---

## 4. 训练策略

### 4.1 两阶段课程学习

| 阶段 | 分辨率 | 帧数 | Epochs | LR | 显存策略 | 每 step | 总时长 |
|------|--------|------|--------|----|----|---------|--------|
| **Stage 1** | 720P (1280×720) | 49 | 5 | 5e-5 | ZeRO-3 无 offload | ~55s | ~22h |
| **Stage 2** | 720P (1280×720) | 121 | 1 | 1e-5 | ZeRO-3 + CPU offload | ~343s | ~27h |

**策略动机**：
- **Stage 1**（短帧数，大批量）：以 49 帧快速让 VACE 学到字形替换的核心能力（外观/局部时序一致性），此时 loss 下降最快。
- **Stage 2**（长帧数微调）：从 Stage 1 的 checkpoint 加载，让模型在更长的时间窗口上适应全局时序一致性。学习率降 5× 避免破坏已学能力。
- **总开销**：约 49 小时（两天），相比直接训 5×121f（~135h）节省 63%。

### 4.2 Stage 2 加载机制

Stage 2 通过新增的 `--model_checkpoint_path` 参数加载 Stage 1 输出：

```python
if is_deepspeed_zero3_enabled():
    from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model
    _load_state_dict_into_zero3_model(model, ckpt_state)
```

在 ZeRO-3 分片模式下，`model.load_state_dict()` 因为参数 shape 为 `[0]` 而失败，必须用 transformers 提供的 `_load_state_dict_into_zero3_model` 进行分片感知加载。

### 4.3 损失函数

沿用 Wan 的 **Flow Matching SFT Loss**：
```
timestep ~ Uniform(min, max) × scheduler.timesteps
noise ~ 𝒩(0, I)
noisy_latent = scheduler.add_noise(target_latent, noise, t)
target = scheduler.training_target(target_latent, noise, t)
pred = model_fn(latent=noisy_latent, vace_context, glyph_latent, t, text_emb)
loss = MSE(pred, target) × scheduler.training_weight(t)
```

仅 `vace` 模块可训练（`--trainable_models vace`），DiT、T5、VAE 全部冻结。

---

## 5. 工程优化（关键：在 8×H100 80GB 上实现 720P 121帧训练）

### 5.1 基线问题分析

720P 121帧的 latent 尺寸：`(1, 111600, 5120)` × bf16 ≈ 1.1 GB / tensor。14B 模型直接训练每卡需要 ~200GB 显存，远超 H100 80GB。

### 5.2 DeepSpeed ZeRO-3 + CPU Offload

`ds_config_14B.json`（Stage 2 用）：

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param":     { "device": "cpu", "pin_memory": true },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0
}
```

- **ZeRO-3**：参数、梯度、优化器状态跨 8 卡分片（14B → 每卡 3.5GB 参数分片）。
- **CPU Offload**：分片后的参数和优化器状态进一步 offload 到 CPU（参数按层按需 gather 到 GPU）。
- **节省**：~40GB/卡（参数 + Adam 状态）。

### 5.3 手动激活值 CPU Offload（关键创新）

PyTorch 的 `torch.autograd.graph.save_on_cpu()` 配合 `use_reentrant=True` 的 checkpoint **不能正常工作**（`save_on_cpu` 的 hook 在 context 退出时就失效，而 `use_reentrant=True` 的 recomputation 在之后的 backward 才发生）。

我们参照 PISCO 项目的做法，自定义 autograd Function 实现**手动 CPU offload**：

```python
class _OffloadToCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.gpu_device = tensor.device
        return tensor.detach().cpu().requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to(ctx.gpu_device)


class _RestoreToGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, device):
        return tensor.to(device)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.cpu(), None
```

**在 DiT block 循环中使用**：

```python
# 一次性把 context, freqs 移到 CPU
context_cpu = _OffloadToCPU.apply(context).requires_grad_(True)
freqs_cpu   = _OffloadToCPU.apply(freqs).requires_grad_(True)

for block in dit.blocks:
    x_cpu = _OffloadToCPU.apply(x).requires_grad_(True)
    x = torch.utils.checkpoint.checkpoint(
        create_custom_forward_offload(block, gpu_device),
        x_cpu, context_cpu, t_mod, freqs_cpu,
        use_reentrant=True,
    )
    del x_cpu
```

**`create_custom_forward_offload`** 在 checkpoint 内部再把 CPU 输入还原到 GPU：

```python
def custom_forward(x_cpu, ctx_cpu, tmod, freqs_cpu):
    x_gpu   = _RestoreToGPU.apply(x_cpu, gpu_device)
    ctx_gpu = _RestoreToGPU.apply(ctx_cpu, gpu_device)
    freqs_gpu = _RestoreToGPU.apply(freqs_cpu, gpu_device)
    return block(x_gpu, ctx_gpu, tmod, freqs_gpu)
```

**关键洞察**：
- `use_reentrant=True` 只保存 input tensor 的 **Python 引用**（不会序列化或复制）。
- 通过将 input 显式移到 CPU，checkpoint 保存的引用指向 CPU tensor → **几乎不占 GPU 显存**。
- `requires_grad_(True)` 避免 `use_reentrant=True` 因检测不到 grad 输入而退化为普通 forward。

### 5.4 VACE hints CPU Offload

VACE 的 8 个 hints（每个 ~1.1GB）在整个 DiT 40 层 forward/backward 期间都保持在 GPU 上，占 ~8.8GB。参照 PISCO：

```python
# VACE forward 后立刻 offload hints
vace_hints = vace(x, vace_context, context, t_mod, freqs, ...)
if use_gradient_checkpointing_offload:
    vace_hints = [h.cpu() for h in vace_hints]
    torch.cuda.empty_cache()

# DiT block 中按需加载
for block_id, block in enumerate(dit.blocks):
    x = ...block forward...
    if block_id in vace_layers_mapping:
        hint = vace_hints[vace_layers_mapping[block_id]]
        if hint.device != x.device:
            hint = hint.to(x.device)
        x = x + hint * vace_scale
```

### 5.5 VAE 排除 ZeRO-3 管理

DiffSynth-Studio 的 Wan VAE 与 DeepSpeed ZeRO-3 不兼容（ZeRO-3 的 forward hook 在 VAE 的某些操作上会失败）。解决方案：

1. **创建时跳过** `deepspeed.zero.Init()`：`skip_zero3` 标志判断模型类名包含 "vae"。
2. **`accelerator.prepare()` 前移除**：`del model.pipe._modules['vae']`，使 DeepSpeed 不注册它。
3. **prepare 后放回**：用 `object.__setattr__(pipe, 'vae', vae_module)` 绕过 `nn.Module.__setattr__`，让 pipeline 代码能访问 `pipe.vae`，但 DeepSpeed 不把它当作子模块管理。

VAE 在每卡独立保存完整副本（~300MB），可忽略。

### 5.6 禁用 checkpoint 确定性检查

ZeRO-3 在 forward 和 recomputation 时参数 gather 的 tensor strides 可能不一致，触发 `CheckpointError`。`determinism_check='none'` 关闭该检查，并不影响梯度正确性（实际计算 graph 是等价的）。

### 5.7 显存占用分解（Stage 2，每卡）

| 项 | 大小 |
|----|------|
| 模型参数分片（bf16） | ~0GB（offload 到 CPU） |
| Optimizer 状态分片（fp32 Adam） | ~0GB（offload 到 CPU） |
| 当前 DiT block 的 gathered 参数 | ~1.1GB |
| 当前 DiT block recomputation 的激活 | ~30–40GB |
| 当前 `x` latent tensor | ~1.1GB |
| Gradient 通信 buffer | ~10GB |
| VAE | ~0.3GB |
| **实测峰值** | **~74GB / 80GB** |

### 5.8 Stage 1 快速配置

Stage 1（49帧）显存需求较小，可去掉 CPU offload 提速：

```yaml
# ds_config_14B_fast.json
"offload_optimizer": { "device": "none" },
"offload_param":     { "device": "none" }
```

使用 `--use_gradient_checkpointing`（不 offload 激活到 CPU），激活值留 GPU，避免 CPU↔GPU 传输气泡。实测每卡 ~50GB，GPU 利用率接近 100%。

---

## 6. 超参数汇总

| 项 | 值 |
|----|----|
| 基础模型 | Wan2.1-VACE-14B |
| 硬件 | 8 × NVIDIA H100 80GB + 2TB CPU RAM |
| 精度 | bfloat16 |
| Batch size per GPU | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 64 |
| Gradient clipping | 1.0 |
| Optimizer | AdamW (lr=5e-5 Stage1 / 1e-5 Stage2, weight_decay=0.01) |
| LR schedule | Constant |
| GlyphEncoder: num_tokens | 64 |
| TargetTextEncoder: vocab / max_len / layers | 8192 / 64 / 2 |
| VACE layers 注入位置 | 8 层：(0, 5, 10, 15, 20, 25, 30, 35) |
| 数据重复 | 10 (dataset_repeat) |
| Stage 1 总 steps | 5 × 288 = 1440 |
| Stage 2 总 steps | 1 × 288 = 288 |

---

## 7. 修改文件索引

| 文件 | 核心修改 |
|------|---------|
| `diffsynth/models/wan_video_vace.py` | 新增 `GlyphEncoder`, `TargetTextEncoder`, `ConditionCrossAttention`；VACE block 返回 tuple；`_OffloadToCPU`/`_RestoreToGPU`；`load_state_dict` 特殊初始化 |
| `diffsynth/pipelines/wan_video.py` | `WanVideoUnit_VACE` 新增 glyph/target_text 处理；`model_fn_wan_video` 手动 CPU offload + hints offload |
| `diffsynth/core/gradient/gradient_checkpoint.py` | `save_on_cpu + use_reentrant=True` 组合 |
| `diffsynth/diffusion/runner.py` | VAE 排除 DeepSpeed 管理 |
| `diffsynth/core/loader/model.py` | ZeRO-3 init 排除 VAE；ZeRO-3 分片状态字典加载 |
| `diffsynth/configs/model_configs.py` | 14B VACE 添加 `glyph_channels=16` |
| `examples/wanvideo/model_training/train.py` | 新增 `--model_checkpoint_path` 支持两阶段加载 |
| `examples/wanvideo/model_training/full/ds_config_14B.json` | ZeRO-3 + CPU offload（Stage 2） |
| `examples/wanvideo/model_training/full/ds_config_14B_fast.json` | ZeRO-3 无 offload（Stage 1） |
| `scripts/train_textvace_14b_two_stage.sh` | 两阶段自动训练脚本 |

---

## 8. 与 PISCO 方法的对比

我们的工程优化方案参考了 PISCO 项目（cxh42/PISCO_code-main），核心的相同点：
- ZeRO-3 + CPU offload
- 手动 `_OffloadToCPU` + `use_reentrant=True` 的 checkpoint
- 低分辨率/短帧数 → 高分辨率/长帧数的两阶段训练

我们的**方法学创新**：
- **GlyphEncoder + ConditionCrossAttention**：字形视频通过 cross-attention 注入，而非 PISCO 的条件通道拼接，保留 VACE 原有 96 通道结构不变。
- **TargetTextEncoder**：字符级哈希编码，对任意 Unicode 字符可 tokenize，处理 T5 OOV 问题。
- **双路径并行**：v2（视觉字形）和 v3（字符文本）可独立或联合启用，提供灵活的条件组合。
