# TextVACE：基于字形感知交叉注意力的视频场景文字编辑

---

## 1. 研究背景

### 1.1 任务定义

**视频场景文字编辑（Video Scene Text Editing）：** 给定输入视频、文字区域掩码、编辑指令（"Change X to Y"），生成仅在文字区域发生变化的编辑后视频。要求目标文字匹配原始的字体/颜色/透视，非文字区域像素级不变，帧间时序一致。

### 1.2 研究空白

| 方向 | 代表工作 | 局限 |
|------|---------|------|
| 图像场景文字编辑 | TextCtrl (NeurIPS'24), FLUX-Text, GlyphMastero (CVPR'25) | 仅处理单帧，无法保证视频时序一致 |
| 视频场景文字替换 | STRIVE (ICCV'21) | 基于GAN+style transfer，非扩散模型，5年无后续 |
| 通用视频编辑 | VACE (ICCV'25), VideoPainter (SIGGRAPH'25) | 不理解"字形"概念，无法精确控制文字渲染 |

**核心空白：不存在基于视频扩散模型的场景文字编辑方法。**

---

## 2. 方法：TextVACE

### 2.1 核心思路

在 VACE 的视频条件注入框架中引入**专用的字形感知通路**——不是简单拼接通道，而是通过独立编码器+逐层交叉注意力让字形信息深度参与每一层的特征生成。

### 2.2 架构设计

```
原始 VACE:
  vace_context(96ch) → Conv3D → 15×[SelfAttn → TextCrossAttn → FFN] → hints → 注入DiT

TextVACE (ours):
  vace_context(96ch) → Conv3D → 15×[SelfAttn → TextCrossAttn → GlyphCrossAttn → FFN] → hints
                                                                    ↑
  glyph_latent(16ch) → GlyphEncoder → 64 tokens ───────────────────┘
```

三个关键组件：

**① GlyphEncoder（字形编码器）**
- 输入：glyph视频的VAE latent (16ch)
- `Conv3D(16→1536)` patch embedding → 空间特征序列
- Cross-attention pooling → 压缩为64个token
- 输出projection zero-init

**② GlyphCrossAttention × 15（字形交叉注意力）**
- 每个VaceWanAttentionBlock新增一个cross-attention层
- Q来自VACE隐状态，K/V来自glyph tokens
- 让每个空间位置查询"我这里应该渲染glyph的哪个部分"
- 输出projection zero-init → 从预训练行为渐进过渡

**③ OCR-Driven Geometric Glyph Rendering（OCR驱动的几何字形渲染）**
- EasyOCR逐帧检测原始文字 → 精确4角点坐标（位置/大小/旋转/透视）
- 用匹配字体渲染目标文字 → 透视变换到检测角点
- 帧间平滑消除检测抖动
- VLM字体识别（ollama qwen3-vl）匹配原始字体风格

### 2.3 设计选择的理由

**为什么用glyph视觉特征做cross-attention，而不是目标文本语义特征？**

| | Glyph视觉特征 | 目标文本语义（T5编码） |
|--|--------------|-------------------|
| 空间信息 | ✅ 包含每个字符的精确位置/大小 | ❌ 无空间信息 |
| 和已有text cross-attn的区别 | ✅ 完全不同的信息 | ❌ T5已经编码了完整prompt |
| 时序追踪 | ✅ 随视频帧变化（反映运动） | ❌ 每帧相同 |

**为什么不concat到vace_context而是用独立cross-attention？**
- Concat方案：glyph的16通道和inactive/reactive/mask的96通道混在Conv3D入口，字形信息在后续层被稀释
- Cross-attention方案：字形信息在每一层被独立查询和注入，不与其他条件信号混淆

### 2.4 参数量

| 组件 | 参数量 | 可训练 |
|------|--------|--------|
| 原始VACE（15个DiTBlock + Conv3D） | 735M | ✅ |
| GlyphEncoder | ~30M | ✅（新增） |
| GlyphCrossAttention × 15 | ~124M | ✅（新增） |
| DiT 1.3B（冻结） | 1300M | ❌ |
| T5（冻结） | ~4800M | ❌ |

---

## 3. 数据与训练

### 3.1 数据

- **训练集：** 230个视频样本，1280×720，24fps，5秒
- 每个样本包含：原始视频、编辑后视频（GT）、文字掩码、编辑指令
- **Glyph视频（OCR版）：** 214个用EasyOCR精准渲染，16个回退到bbox方法
- **VLM字体识别：** ollama qwen3-vl:8b，字体分布 Arial(108), Impact(91), 其他(31)
- **Unicode脚本检测：** CJK→Noto Sans CJK, Cyrillic→FreeSans, 数学→DejaVu

### 3.2 训练配置

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA RTX 5090 (32GB) |
| 模型 | Wan2.1-VACE-1.3B + GlyphEncoder + GlyphCrossAttn |
| 训练模式 | SFT（VACE全模块+glyph模块可训练，DiT/T5/VAE冻结） |
| 分辨率 | 480×832, 17帧 |
| 步数 | 2300步/epoch × 5 epochs = 11500步 |
| 学习率 | 5e-5, AdamW |
| 显存 | ~29GB / 32GB |
| 速度 | ~3.4s/step |
| 预计时间 | ~11小时 |

---

## 4. 实验结果

### 4.1 第一版实验（glyph concat方案，bbox渲染）

使用简单的通道拼接（vace_in_dim 96→112）+ mask bbox内居中渲染glyph。

**训练集样本推理：**

| 样本 | 编辑 | PSNR | SSIM |
|------|------|------|------|
| 0000007 | ATP→NAD | 17.21 | 0.6816 |
| 0000051 | TFLCAR→CARLIFE | 21.90 | 0.8278 |
| 0001273 | 慈→善 | 18.45 | 0.6982 |

**发现：**
- ✅ 文字编辑能力验证成功，目标文字清晰可读
- ✅ 泛化能力存在——训练时没见过的目标文字（GDP、AMP、福、道）也能渲染
- ✅ 未见视频上也有编辑效果（VLM自动识别原文+生成替换指令）
- ❌ 背景保真度不足（PSNR 17-22，理想应>30）
- ❌ Pixel-Anchored Denoising（推理时锚定非mask区域）效果微弱（+0.3dB），**结论：背景保真需从训练端解决**

### 4.2 改进方向（当前进行中）

| 改进 | 状态 | 预期效果 |
|------|------|---------|
| OCR精准glyph渲染（透视变换） | ✅ 已完成 | glyph和原文精确对齐 |
| GlyphCrossAttention架构 | 🔄 训练中 | 字形信息逐层深度注入 |
| Mask-weighted loss | 📋 待做 | 提升背景保真度 |
| 扩充训练数据 | 📋 待做 | 提升整体质量 |

---

## 5. 论文规划

### 5.1 标题（草案）

**"TextVACE: Glyph-Aware Cross-Attention for Video Scene Text Editing"**

### 5.2 贡献点

1. **任务+Benchmark：** 首个基于视频扩散模型的场景文字编辑任务定义，含专用数据集
2. **OCR-Driven Temporal Glyph Conditioning：** OCR检测→透视跟踪→字形渲染的几何条件生成pipeline，解决视频文字的时序几何变化
3. **Glyph-Aware Cross-Attention：** 在VACE block中引入独立的字形交叉注意力通路，通过GlyphEncoder压缩字形特征+逐层cross-attention注入
4. **全面实验：** 消融实验验证各组件，和STRIVE/TextCtrl-per-frame/vanilla VACE对比

### 5.3 对比实验计划

| 方法 | 类型 |
|------|------|
| STRIVE (ICCV'21) | 唯一前作（GAN方法） |
| TextCtrl逐帧应用 | 图像STE方法的简单扩展 |
| VACE原始微调（无glyph） | 通用视频编辑baseline |
| TextVACE-concat（glyph通道拼接） | 消融：简单注入方式 |
| **TextVACE（完整方法）** | 我们的方法 |

### 5.4 消融实验

- 有/无 GlyphCrossAttention
- 有/无 OCR精准渲染（OCR vs bbox）
- 有/无 VLM字体匹配
- 不同glyph_num_tokens（32/64/128）
- 不同训练数据量

---

## 6. 文件结构

```
DiffSynth-Studio-TextVACE/
├── data/
│   ├── raw/{original_videos, edited_videos, text_masks, edit_instructions}  (230样本)
│   ├── processed/{glyph_videos, font_info, parsed_records.json}            (OCR版glyph)
│   ├── inference_raw/{target_video, mask_video}                            (470样本，240未见)
│   ├── inference_processed/{dilated_masks, glyph_videos, inference_records.json}
│   └── metadata.csv                                                        (训练元数据)
├── diffsynth/
│   ├── models/wan_video_vace.py      ← GlyphEncoder + GlyphCrossAttention + VaceWanModel
│   ├── pipelines/wan_video.py        ← glyph_video独立编码, glyph_latent传入VACE
│   └── configs/model_configs.py      ← glyph_channels=16
├── scripts/
│   ├── prepare_textvace_data.py      # 数据准备（指令解析+glyph渲染v1）
│   ├── recognize_fonts.py            # VLM字体识别
│   ├── render_glyph_ocr.py           # OCR驱动的精准glyph渲染v2
│   ├── prepare_inference_data.py     # 推理数据准备（去重+膨胀mask+VLM识别）
│   ├── inference_textvace.py         # 推理脚本
│   ├── inference_novel_text.py       # 新文字泛化测试
│   ├── inference_unseen.py           # 未见视频推理
│   └── train_textvace.sh             # 训练启动脚本
├── models/train/TextVACE_sft/        # Checkpoints + training_log.txt
├── outputs/textvace_inference/       # 推理结果
│   ├── train_samples/                # 训练集推理
│   ├── novel_text/                   # 同视频新文字
│   └── unseen_videos/                # 全新视频
├── logs/                             # 训练/推理日志
└── docs/TextVACE.md                  # 本文档
```

---

## 7. 使用指南

```bash
# 数据准备
conda run -n DiffSynth-Studio python scripts/prepare_textvace_data.py
conda run -n DiffSynth-Studio python scripts/recognize_fonts.py
conda run -n DiffSynth-Studio python scripts/render_glyph_ocr.py

# 训练
conda activate DiffSynth-Studio
bash scripts/train_textvace.sh
tail -f models/train/TextVACE_sft/training_log.txt  # 监控loss

# 推理
conda run -n DiffSynth-Studio python scripts/inference_textvace.py \
    --checkpoint models/train/TextVACE_sft/epoch-4.safetensors
```
