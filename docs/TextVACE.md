# VideoSTE：视频场景文字编辑

> 更新日期：2026-04-14

---

## 1. 研究背景

### 1.1 任务定义

**视频场景文字编辑（Video Scene Text Editing, VideoSTE）**：给定输入视频和文字区域掩码，将视频中的指定文字替换为目标文字。要求目标文字匹配原始字体/颜色/透视，非文字区域像素级不变，帧间时序一致。

### 1.2 研究空白

| 方向 | 代表工作 | 局限 |
|------|---------|------|
| 图像场景文字编辑 | TextCtrl (NeurIPS'24), AnyText, GlyphMastero (CVPR'25) | 仅单帧，无法保证视频时序一致 |
| 视频场景文字替换 | STRIVE (ICCV'21) | 基于 GAN，非扩散模型，5 年无后续 |
| 通用视频编辑 | VACE (ICCV'25), VideoPainter (SIGGRAPH'25) | 不理解字形，无法精确控制文字渲染 |
| 文字生成/渲染 | TextDiffuser-2, GlyphControl | 图像级，不处理视频时序 |

**核心空白：不存在高质量的视频场景文字编辑方法和评测基准。**

---

## 2. 分解式数据制作 Pipeline（核心贡献 1）

### 2.1 方法

将端到端的困难问题拆解为四个可解子任务：

```
输入：带文字的原始视频 + 编辑指令

Step 1: SAM3 分割目标文字区域 → 掩码视频（Mask）
Step 2: PISCO 去除掩码区域文字 → 干净视频（Clean）
Step 3: Nano Banana Pro 编辑第一帧文字 → 编辑后首帧
Step 4: SAM3 分割编辑后首帧文字 → 编辑后文字片段
Step 5: 微调 PISCO 将文字片段插入干净视频 → 编辑后视频

输出：高质量的编辑后视频
```

### 2.2 核心技术：微调 PISCO 实现文字重插入

- PISCO 原本是视频文字**去除**模型（有文字→干净）
- 用 (干净视频, 原视频) 数据对**反向微调**，学会**插入**文字
- 去除和插入是**对称任务**，同一架构双向使用
- 微调后能将编辑文字片段高质量融合进干净视频，保持光影/透视/时序一致

### 2.3 数据产出与独创性

产出 **230 个高质量视频文字编辑数据对**（1280×720, 24fps, ~5s），覆盖英文/中文/韩文/符号。另有 240 个未见视频用于泛化测试。

**这是全世界唯一的高质量真实视频文字编辑配对数据。** 现有方法全部使用合成数据训练（SynthText 引擎渲染），质量远低于真实数据。

| 方法 | 数据类型 | 质量 |
|------|---------|------|
| SRNet / MOSTEL / TextCtrl | 合成图像对 | 低-中：无真实光影/透视 |
| AnyText / TextDiffuser | 真实图（非配对） | 无编辑前后配对 |
| **Ours** | **真实视频配对** | **高：保留真实光影/透视/时序** |

---

## 3. VideoSTE-Bench 评测基准（核心贡献 2）

### 3.1 数据集构成

| 子集 | 数量 | 用途 | 有 GT |
|------|------|------|-------|
| Test-Paired | 230 个视频对 | 自动评测（PSNR/SSIM/OCR） | 有 |
| Test-Wild | 240 个视频 | 人工评测 / VLM 评测 | 无 |

### 3.2 五维度评测体系

借鉴 VBench（多维度分解）、Physics-IQ（真实 GT 对比）、LegiT（文字可读性）的设计思想：

| 维度 | 衡量什么 | 核心指标 | 灵感来源 |
|------|---------|---------|---------|
| **Text Accuracy** | 文字是否正确 | PARSeq OCR Word/Char Accuracy | LegiT |
| **Text Legibility** | 文字是否清晰 | OCR 置信度 + MUSIQ 质量分 | LegiT |
| **Background Preservation** | 背景是否不变 | 非 mask 区域 PSNR/SSIM/LPIPS | Physics-IQ |
| **Temporal Consistency** | 帧间是否稳定 | 帧间 SSIM + CLIP 一致性 | VBench |
| **Overall Realism** | 整体是否自然 | FID + VLM 2AFC 判断 | Physics-IQ |

### 3.3 按类别分析

| 分类维度 | 类别 | 分析目的 |
|---------|------|--------|
| 文字类型 | Latin / CJK / Symbol | 不同文字系统的难度差异 |
| 文字长度 | 短(1-3) / 中(4-10) / 长(10+) | 文字复杂度的影响 |
| 运动程度 | 静态 / 轻微 / 明显 | 运动对编辑质量的影响 |

详细设计见 [benchmark_design.md](benchmark_design.md)

---

## 4. 端到端方法探索（论文中的消融实验）

### 4.1 方案 v1：Glyph 通道拼接（1.3B）

将 glyph 视频 VAE 编码拼接到 VACE 输入通道（96ch→112ch）。

**结果**：文字部分可读，背景保真不足（PSNR 17-22）。Pixel-Anchored Denoising 无明显效果。

### 4.2 方案 v2：Glyph Cross-Attention（1.3B, 已完成）

```
glyph_video → VAE → GlyphEncoder(Conv3D + pooling → 64 tokens)
                                      ↓
vace_context → Conv3D → 15×[SelfAttn → TextCrossAttn → GlyphCrossAttn → FFN]
```

- 训练：5 epochs × 2300 steps，~11 小时，RTX 5090 (1.3B)
- 最终 loss: 0.005221
- 30 个未见视频推理完成
- **结论**：效果受限于 glyph 渲染质量（OCR 定位偏差、字体匹配不准、透视变换误差）

### 4.3 方案 v3：Character Encoder（1.3B, 已完成）

```
"NAD" → CharTokenize → TargetTextEncoder(Embed + 2层Transformer) → 64 tokens
                                                                      ↓
vace_context → Conv3D → 15×[SelfAttn → TextCrossAttn → CharCrossAttn → FFN]
```

- TargetTextEncoder ~72M + ConditionCrossAttention ×15 ~142M
- 训练：3 epochs × 2300 steps，~5.4 小时，RTX 5090 (1.3B)
- **结论**：文字完全不可辨认。原因：230 样本不足以让模型学会字符→视觉渲染映射（需百万级数据）。调研确认不存在预训练的文字→视觉特征编码器。

### 4.4 方案 v2-14B：Glyph Cross-Attention（14B, 计划中）

在新机器上用 14B 模型 + 49 帧重新训练 v2 方案，验证更大模型是否能改善效果。

详见第 6 节训练指南。

### 4.5 三方案对比总结

| | 分解式 Pipeline | Glyph-VACE (v2) | CharEncoder (v3) |
|--|:---:|:---:|:---:|
| 文字可读性 | 高 | 中（部分可辨认） | 极低（乱码） |
| 背景保真 | 高 | 低 | 低 |
| 时序一致 | 高 | 中 | 中 |
| 数据需求 | 少 | 中（230 不够理想） | 极大（需百万级） |
| 核心瓶颈 | 速度慢 | glyph 质量天花板 | 数据量不足 |

---

## 5. 论文规划

### 5.1 定位

**NeurIPS 2026 主赛道**，偏 Benchmark + Method

**标题方向**：
> "VideoSTE-Bench: A Real-World Benchmark for Video Scene Text Editing via Decomposed Segmentation, Removal, and Reinsertion"

### 5.2 篇幅分配

| 部分 | 比例 | 内容 |
|------|------|------|
| 数据制作 Pipeline | 40% | 分解式方法 + PISCO 微调 + 数据质量分析 |
| 评测基准 | 40% | 五维度评测体系 + Baseline 对比 + 按类别分析 |
| 端到端方法探索 | 20% | Glyph-VACE / CharEncoder-VACE 消融 + 分析 |

### 5.3 贡献点

1. **分解式方法**：首个实用的 VideoSTE 框架（分割→去除→编辑→重插入）
2. **视频文字重插入**：反向微调 PISCO 实现文字高质量融合（核心技术）
3. **VideoSTE-Bench**：首个真实视频文字编辑 benchmark（230 对 + 5 维度评测）
4. **系统性分析**：分解式 vs 端到端的范式对比，验证数据受限场景下的最优策略

### 5.4 对比实验

| 方法 | 类型 |
|------|------|
| **Ours（分解式 Pipeline）** | 主方法 |
| TextCtrl 逐帧 | Baseline（图像 STE 逐帧） |
| VACE 微调（无 glyph） | Baseline（通用视频编辑） |
| Glyph-VACE (v2, 1.3B) | 消融（端到端 + glyph） |
| Glyph-VACE (v2, 14B) | 消融（更大模型） |
| CharEncoder-VACE (v3) | 消融（端到端 + 字符编码） |

---

## 6. 14B Glyph-VACE 训练指南（新机器）

### 6.1 环境准备

```bash
# 1. 克隆代码
git clone <repo_url> DiffSynth-Studio-TextVACE
cd DiffSynth-Studio-TextVACE

# 2. 创建 conda 环境
conda create -n DiffSynth-Studio python=3.12
conda activate DiffSynth-Studio
pip install -e .
pip install accelerate

# 3. 下载 14B VACE 模型
# 从 HuggingFace 下载 Wan-AI/Wan2.1-VACE-14B
# 需要文件：
#   diffusion_pytorch_model.safetensors (或分片 diffusion_pytorch_model-*.safetensors)
#   models_t5_umt5-xxl-enc-bf16.pth
#   Wan2.1_VAE.pth
python -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.1-VACE-14B')"

# 4. 解压数据
# 将 data.tar.gz 解压到项目根目录
tar xzf data.tar.gz
```

### 6.2 修改配置

**关键：修改 14B VACE 的 model_configs.py，添加 glyph_channels**

找到 `diffsynth/configs/model_configs.py` 中 hash 为 `7a513e1f257a861512b1afd387a8ecd9` 的 `wan_video_vace` 条目：

```python
# 修改前：
{
    "model_hash": "7a513e1f257a861512b1afd387a8ecd9",
    "model_name": "wan_video_vace",
    "model_class": "diffsynth.models.wan_video_vace.VaceWanModel",
    "extra_kwargs": {
        'vace_layers': (0, 5, 10, 15, 20, 25, 30, 35),
        'vace_in_dim': 96,
        'patch_size': (1, 2, 2),
        'has_image_input': False,
        'dim': 5120,
        'num_heads': 40,
        'ffn_dim': 13824,
        'eps': 1e-06
    },
    "state_dict_converter": "..."
},

# 修改后（添加 glyph_channels）：
{
    "model_hash": "7a513e1f257a861512b1afd387a8ecd9",
    "model_name": "wan_video_vace",
    "model_class": "diffsynth.models.wan_video_vace.VaceWanModel",
    "extra_kwargs": {
        'vace_layers': (0, 5, 10, 15, 20, 25, 30, 35),
        'vace_in_dim': 96,
        'glyph_channels': 16,             # ← 添加这行
        'patch_size': (1, 2, 2),
        'has_image_input': False,
        'dim': 5120,
        'num_heads': 40,
        'ffn_dim': 13824,
        'eps': 1e-06
    },
    "state_dict_converter": "..."
},
```

注意 14B 与 1.3B 的区别：
- dim: 5120 vs 1536
- num_heads: 40 vs 12
- vace_layers: 8 层 vs 15 层
- GlyphEncoder 和 ConditionCrossAttention 会自动用 dim=5120 初始化

### 6.3 数据准备

数据已在 `data/` 目录中。确认结构：

```
data/
├── raw/
│   ├── original_videos/     (230 个原始视频)
│   ├── edited_videos/       (230 个编辑后视频)
│   ├── text_masks/          (230 个掩码视频)
│   └── edit_instructions/   (编辑指令)
├── processed/
│   ├── glyph_videos/        (214 个 OCR 渲染的 glyph 视频)
│   ├── font_info/           (字体信息)
│   └── parsed_records.json  (解析后的记录)
├── metadata.csv             (v2 glyph 方案训练用)
└── metadata_v3.csv          (v3 字符编码方案训练用)
```

v2 训练使用 `metadata.csv`，其中每行包含：
```
video,vace_video,vace_video_mask,glyph_video,prompt
raw/edited_videos/xxx.mp4,raw/original_videos/xxx.mp4,raw/text_masks/xxx.mp4,processed/glyph_videos/xxx.mp4,Change ATP to NAD
```

### 6.4 训练脚本

创建 `scripts/train_textvace_14b.sh`：

```bash
#!/bin/bash
# TextVACE v2 (Glyph) - 14B Model Training
# 480P, 49 frames
# 需要显存估算：14B DiT(冻结) + VACE(~2.4B可训练) + GlyphEncoder + CrossAttn
# 建议至少 80GB VRAM (A100/H100)，可能需要多卡

MODEL_DIR="<14B模型路径，例如 ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-14B/snapshots/xxx>"

mkdir -p logs

export PYTHONUNBUFFERED=1

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/ \
  --dataset_metadata_path data/metadata.csv \
  --data_file_keys "video,vace_video,vace_video_mask,glyph_video" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 10 \
  --model_paths "[\"${MODEL_DIR}/diffusion_pytorch_model.safetensors\",\"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_DIR}/Wan2.1_VAE.pth\"]" \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/TextVACE_14B_sft" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_video_mask,glyph_video" \
  --use_gradient_checkpointing_offload \
  --initialize_model_on_cpu
```

**注意事项**：
- `MODEL_DIR` 需改为实际 14B 模型路径
- 如果模型文件是分片的（`diffusion_pytorch_model-00001-of-00006.safetensors`），传所有分片路径或模型目录
- 49 帧 + 14B 模型显存需求很大，可能需要：
  - 单卡 80GB（A100/H100）+ gradient checkpointing offload
  - 或多卡并行（accelerate 会自动处理）
  - 如果 OOM：降到 33 帧（`--num_frames 33`）或降分辨率
- 清除 pycache：`find . -name "__pycache__" -exec rm -rf {} +`

### 6.5 训练命令

```bash
# 清除缓存（重要！确保使用最新代码）
find . -name "__pycache__" -exec rm -rf {} +

# 启动训练
conda activate DiffSynth-Studio
bash scripts/train_textvace_14b.sh 2>&1 | tee logs/train_14b.log

# 监控
tail -f models/train/TextVACE_14B_sft/training_log.txt
nvidia-smi -l 5  # 监控显存
```

### 6.6 推理

```bash
python scripts/inference_textvace.py \
    --checkpoint models/train/TextVACE_14B_sft/epoch-4.safetensors \
    --num_frames 49
```

推理脚本中的 `MODEL_DIR` 也需要改为 14B 模型路径。

---

## 7. 已有资产

### 模型检查点
| 检查点 | 方案 | 模型 | 位置 |
|--------|------|------|------|
| epoch-0 ~ epoch-4 | v2 Glyph | 1.3B | `models/train/TextVACE_sft/` |
| epoch-0 ~ epoch-2 | v3 CharEncoder | 1.3B | `models/train/TextVACE_v3_sft/` |
| 待训练 | v2 Glyph | 14B | `models/train/TextVACE_14B_sft/` |

### 推理结果
| 结果 | 样本数 | 位置 |
|------|--------|------|
| v2 (1.3B) 未见视频 | 30 | `outputs/textvace_inference/unseen_final/` |
| v3 (1.3B) 训练集 + 未见 | 30 | `outputs/textvace_v3_inference/` |

### 关键代码文件
| 文件 | 说明 |
|------|------|
| `diffsynth/models/wan_video_vace.py` | GlyphEncoder + TargetTextEncoder + ConditionCrossAttn |
| `diffsynth/pipelines/wan_video.py` | Pipeline（支持 glyph_video 和 target_text） |
| `diffsynth/configs/model_configs.py` | 模型配置（1.3B 已改，14B 需手动改） |
| `scripts/render_glyph_ocr.py` | OCR 驱动的 glyph 渲染 |
| `scripts/train_textvace.sh` | 1.3B v2 训练脚本 |
| `scripts/train_textvace_v3.sh` | 1.3B v3 训练脚本 |

---

## 8. TODO

- [ ] **14B 训练**：在新机器上用 v2 方案训练 14B 模型（480p, 49帧）
- [ ] **量化评测**：在 230 对上跑 benchmark 评测 pipeline
- [ ] **Baseline**：TextCtrl 逐帧 / VACE 原始微调
- [ ] **论文写作**：Pipeline 方法 + Benchmark 设计 + 实验
- [ ] **人工评测**：设计 MOS 评分问卷
