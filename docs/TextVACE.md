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

### 2.2 架构演进

#### v2（Glyph方案 — 已完成训练）

```
vace_context(96ch) → Conv3D → 15×[SelfAttn → TextCrossAttn → GlyphCrossAttn → FFN] → hints
                                                                    ↑
glyph_latent(16ch) → GlyphEncoder → 64 tokens ─────────────────────┘
```

- GlyphEncoder：glyph视频VAE latent → Conv3D → cross-attention pooling → 64 tokens
- GlyphCrossAttention × 15：每层注入glyph视觉特征
- **问题**：依赖外部渲染的glyph视频质量，OCR检测不准时模型上限被卡住

#### v3（Character Encoder方案 — 当前版本）

```
vace_context(96ch) → Conv3D → 15×[SelfAttn → TextCrossAttn → CharCrossAttn → FFN] → hints
                                                                    ↑
"NAD" → CharTokenize → TargetTextEncoder(Embed+Transformer) ───────┘
```

三个关键组件：

**① Character Tokenizer（字符分词器）**
- 将目标文字逐字符转为token ID：`ord(char) % (vocab_size-1) + 1`
- 支持ASCII、CJK、Cyrillic等任意Unicode字符
- vocab_size=8192，max_len=64，0为PAD token

**② TargetTextEncoder（目标文字编码器）**
- 字符嵌入：`nn.Embedding(8192, 1536, padding_idx=0)`
- 位置嵌入：`nn.Embedding(64, 1536)` — 编码字符顺序
- 2层Transformer Encoder（`norm_first=True, activation='gelu'`）— 建模字符间关系
- 零初始化输出投影 → 从预训练VACE行为渐进过渡
- **关键**：字符级编码提供T5 subword tokenization无法保证的精确字符身份

**③ ConditionCrossAttention × 15（条件交叉注意力）**
- 每个VaceWanAttentionBlock新增一个cross-attention层
- Q来自VACE隐状态（空间位置），K/V来自字符tokens
- 每个空间位置查询"我这里应该关注哪个字符"
- 输出projection zero-init

### 2.3 v3设计选择的理由

**为什么用字符级编码，而不是依赖T5？**

| | 字符级编码（TargetTextEncoder） | T5 prompt编码 |
|--|------|------|
| 编码粒度 | ✅ 逐字符独立嵌入 | ❌ subword tokenization，字符边界模糊 |
| 注入位置 | ✅ VACE blocks内（专控编辑区域） | DiT backbone（全局影响） |
| 信息互补 | ✅ 精确字符身份 | 编辑意图的语义理解 |
| 外部依赖 | ✅ 无（纯编码） | 无 |

**为什么不用glyph视频？**
- glyph方案依赖OCR检测精度、字体匹配、透视变换等外部渲染步骤
- 任何一步出错都会降低模型上限
- 字符编码方案让模型自己学习文字渲染，不受外部质量限制

**三路信息汇聚（v3的核心优势）：**
1. **T5 prompt**：语义级理解编辑意图（"Change ATP to NAD"）
2. **TargetTextEncoder**：字符级精确标识目标文字（N-A-D）
3. **VACE context**：原视频mask区域的视觉上下文（位置、风格、透视参考）

**灵感来源（ConsID-Gen, CVPR'25）：**
- ConsID-Gen使用双流编码（外观+几何）+ MMDiT双向注意力融合
- 我们类比：T5提供编辑语义（类似外观流），TargetTextEncoder提供字符身份（类似几何流）
- 两路信息在VACE blocks中汇聚，让模型同时理解"编辑什么"和"具体是哪些字符"

### 2.4 参数量

| 组件 | 参数量 | 可训练 | 版本 |
|------|--------|--------|------|
| 原始VACE（15个DiTBlock + Conv3D） | 735M | ✅ | all |
| TargetTextEncoder（Embed+2层Transformer） | ~72M | ✅（新增） | v3 |
| ConditionCrossAttention × 15 | ~124M | ✅（新增） | v3 |
| GlyphEncoder（仅v2） | ~30M | ✅ | v2 |
| DiT 1.3B（冻结） | 1300M | ❌ | all |
| T5（冻结） | ~4800M | ❌ | all |

---

## 3. 数据与训练

### 3.1 数据

- **训练集：** 230个视频样本，1280×720，24fps，5秒
- 每个样本包含：原始视频、编辑后视频（GT）、文字掩码、编辑指令
- **Glyph视频（OCR版）：** 214个用EasyOCR精准渲染，16个回退到bbox方法
- **VLM字体识别：** ollama qwen3-vl:8b，字体分布 Arial(108), Impact(91), 其他(31)
- **Unicode脚本检测：** CJK→Noto Sans CJK, Cyrillic→FreeSans, 数学→DejaVu

### 3.2 训练配置

**v2（Glyph方案）：**

| 参数 | 值 |
|------|-----|
| 模型 | Wan2.1-VACE-1.3B + GlyphEncoder + GlyphCrossAttn |
| 步数 | 2300步/epoch × 5 epochs = 11500步 |
| 速度 | ~3.4s/step |
| 总时间 | ~11小时 |

**v3（Character Encoder方案）：**

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA RTX 5090 (32GB) |
| 模型 | Wan2.1-VACE-1.3B + TargetTextEncoder + ConditionCrossAttn |
| 训练模式 | SFT（VACE全模块+text encoder模块可训练，DiT/T5/VAE冻结） |
| 分辨率 | 480×832, 17帧 |
| 步数 | 2300步/epoch × 4 epochs = 9200步 |
| 学习率 | 5e-5, AdamW |
| 显存 | ~25-27GB / 32GB（无glyph VAE编码，更省显存） |
| 预计时间 | ~6-7小时 |

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

### 4.2 v2 GlyphCrossAttention 实验结果

- 5 epochs训练完成，最终loss: 0.005221
- 30个未见视频推理结果在 `outputs/textvace_inference/unseen_final/`
- **发现**：效果受glyph视频渲染质量限制，OCR检测不准的样本效果差

### 4.3 v3 Character Encoder 方案（当前进行中）

| 改进 | 状态 | 预期效果 |
|------|------|---------|
| TargetTextEncoder + ConditionCrossAttn | 🔄 训练中 | 字符级精确编码，不依赖外部渲染 |
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

- **v3 vs v2**：Character Encoder vs Glyph Encoder（核心对比）
- 有/无 ConditionCrossAttention（只靠T5 prompt vs 加字符编码）
- 不同Transformer层数（1/2/3层）
- 不同vocab策略（char-level vs subword）
- 有/无 GlyphCrossAttention（v2消融）
- 有/无 OCR精准渲染（v2消融）
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
# === v3 Character Encoder 方案（当前） ===
# 数据准备：无需额外步骤，metadata_v3.csv直接使用原始数据

# 训练
conda activate DiffSynth-Studio
bash scripts/train_textvace_v3.sh
tail -f models/train/TextVACE_v3_sft/training_log.txt

# 推理
conda run -n DiffSynth-Studio python scripts/inference_textvace.py \
    --checkpoint models/train/TextVACE_v3_sft/epoch-3.safetensors

# === v2 Glyph 方案（已完成） ===
# 数据准备
conda run -n DiffSynth-Studio python scripts/prepare_textvace_data.py
conda run -n DiffSynth-Studio python scripts/recognize_fonts.py
conda run -n DiffSynth-Studio python scripts/render_glyph_ocr.py

# 训练
bash scripts/train_textvace.sh
tail -f models/train/TextVACE_sft/training_log.txt

# 推理
conda run -n DiffSynth-Studio python scripts/inference_textvace.py \
    --checkpoint models/train/TextVACE_sft/epoch-4.safetensors
```
