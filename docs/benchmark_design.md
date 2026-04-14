# VideoSTE-Bench: 视频场景文字编辑评测基准设计

> 参考论文：VBench（多维度分解评测）、Physics-IQ（真实视频对比评测）、LegiT（文字可读性评测）

---

## 1. 设计理念

借鉴三篇参考论文的核心思想：

| 参考论文 | 核心思想 | 我们借鉴的点 |
|--------|--------|----------|
| **VBench** | 将"视频生成质量"分解为 16 个细粒度维度，每个维度独立评测 | 将"视频文字编辑质量"分解为多个独立维度 |
| **Physics-IQ** | 用真实视频作为 GT 对比生成结果，多指标互补（Spatial-IoU、MSE 等） | 用 230 对真实 GT 做像素级对比评测 |
| **LegiT** | 专门评测文字可读性（legibility），结合 OCR + 人工 MOS 评分 | 文字区域的可读性/质量专项评测 |

---

## 2. 数据集构成

### 2.1 数据划分

| 子集 | 数量 | 用途 | 包含 GT |
|------|------|------|--------|
| **Test-Paired** | 230 个视频对 | 自动评测（有 GT，可算像素级指标） | 有 |
| **Test-Wild** | 240 个视频 | 人工评测 / VLM 评测（无 GT，测泛化能力） | 无 |
| **总计** | 470 个视频 | — | — |

说明：
- 我们的分解式方法（pipeline）不需要用这些数据训练，所以**全部 230 对都用于评测**
- 如果未来有方法想用部分数据做微调，可以自行划分 train/test，但我们建议 230 对全部作为 test set

### 2.2 数据标注内容

每个 Test-Paired 样本包含：

| 字段 | 说明 | 示例 |
|------|------|------|
| `original_video` | 原始视频（含源文字） | 1280×720, 24fps, ~5s |
| `edited_video` | GT 编辑后视频（含目标文字） | 同上 |
| `mask_video` | 文字区域掩码视频 | 二值 mask |
| `source_text` | 原始文字内容 | "ATP" |
| `target_text` | 目标文字内容 | "NAD" |
| `instruction` | 编辑指令 | "Change ATP to NAD" |
| `script_type` | 文字类型 | Latin / CJK / Cyrillic / Symbol |
| `text_length` | 目标文字字符数 | 3 |
| `mask_area_ratio` | 掩码占画面比例 | 0.05 |

### 2.3 数据多样性统计

| 维度 | 分布 |
|------|------|
| 文字类型 | 英文 ~170, 中文 ~30, 韩文 ~10, 数学/符号 ~20 |
| 文字长度 | 1-3 字符 ~80, 4-10 字符 ~100, 10+ 字符 ~50 |
| 场景类型 | 黑板/白板, 街头招牌, 屏幕显示, 印刷品, 手写 等 |
| 运动程度 | 静态 ~100, 轻微运动 ~80, 明显运动 ~50 |

---

## 3. 评测维度设计

借鉴 VBench 的多维度分解思想，将"视频文字编辑质量"分解为 **5 个独立维度**：

```
                     VideoSTE Quality
                    /        |        \
              Text Quality   |   Video Quality
              /      \       |       /      \
    Text        Text    Background  Temporal   Overall
  Accuracy   Legibility Preservation Consistency Realism
```

### 3.1 维度一：Text Accuracy（文字准确性）

**定义**：编辑后视频中，目标文字是否被正确渲染

**评测方法**：
- 对编辑后视频的文字区域（mask 内）逐帧 OCR 识别
- 将识别结果与 `target_text` 对比
- 使用 PARSeq（当前 SOTA 场景文字识别模型）

**指标**：
- **Word Accuracy**：完全匹配的帧占比
- **Character Accuracy**：字符级编辑距离归一化后的准确率（1 - NED）
- **Frame-level OCR Rate**：所有帧中能检测到文字的帧占比

**计算方式**：
```
Word_Acc = (识别完全正确的帧数) / (总帧数)
Char_Acc = 1 - mean(EditDistance(pred, target) / max(len(pred), len(target)))
OCR_Rate = (检测到文字的帧数) / (总帧数)
```

**灵感来源**：LegiT 使用 COCO-Text 的 legibility 标注 + OCR 模型验证；VBench 对每个维度用专门的评测 pipeline

### 3.2 维度二：Text Legibility（文字可读性/质量）

**定义**：即使文字正确，它是否清晰可读、质量高（而非模糊/扭曲/有伪影）

**评测方法**（借鉴 LegiT）：
- 从编辑后视频的 mask 区域裁剪文字 patch
- 用预训练的图像质量评估模型（MUSIQ / CLIPIQA）评估 patch 质量
- 或直接用 OCR 模型的置信度作为可读性代理指标

**指标**：
- **OCR Confidence**：OCR 模型对识别结果的平均置信度（0-1）
- **Text Patch Quality**：文字区域裁剪图的 MUSIQ / CLIPIQA 分数
- **Legibility MOS**（人工评测）：人工打分 1-5，文字是否清晰可读

**灵感来源**：LegiT 论文的核心贡献就是文字可读性预测，使用 IQA 模型 + 文字质量特征

### 3.3 维度三：Background Preservation（背景保真度）

**定义**：非文字区域是否保持与原视频一致（像素级不变）

**评测方法**：
- 使用 mask 将文字区域排除，只计算非 mask 区域的相似度
- 对比编辑后视频与原始视频（注意：与**原始视频**对比，不是与 GT 对比）

**指标**：
- **BG-PSNR**：非 mask 区域的 PSNR（Peak Signal-to-Noise Ratio）
- **BG-SSIM**：非 mask 区域的 SSIM（Structural Similarity）
- **BG-LPIPS**：非 mask 区域的 LPIPS（Learned Perceptual Image Patch Similarity）

**计算方式**：
```
mask_inv = 1 - mask  # 非文字区域
BG-PSNR = PSNR(original * mask_inv, edited * mask_inv)
BG-SSIM = SSIM(original * mask_inv, edited * mask_inv)
```

**理想值**：BG-PSNR → ∞, BG-SSIM → 1.0（完全不变）

**灵感来源**：Physics-IQ 的 MSE 指标衡量像素级精度；VBench 的 Background Consistency 维度

### 3.4 维度四：Temporal Consistency（时序一致性）

**定义**：编辑后的文字在帧间是否稳定连贯，没有闪烁/抖动/突变

**评测方法**：
- 计算编辑后视频相邻帧在文字区域的一致性
- 用光流 warp 相邻帧，计算 warp error
- 或用 CLIP 帧间特征相似度

**指标**：
- **Text Temporal Stability**：文字区域相邻帧的 SSIM 均值
- **Warp Error**：光流 warp 后的像素差异（类似 Physics-IQ 的 Spatiotemporal-IoU 思想）
- **CLIP Frame Consistency**：CLIP 特征帧间余弦相似度（类似 VBench 的 Temporal Consistency）

**计算方式**：
```
TTS = mean([SSIM(frame[t][mask], frame[t+1][mask]) for t in range(T-1)])
CFC = mean([cos_sim(CLIP(frame[t]), CLIP(frame[t+1])) for t in range(T-1)])
```

**灵感来源**：VBench 的 Temporal Flickering + Subject Consistency 维度；Physics-IQ 的 Spatiotemporal-IoU

### 3.5 维度五：Overall Realism（整体真实感）

**定义**：编辑后的视频整体看起来是否自然真实

**评测方法**：
- **自动**：FID（编辑后帧 vs 真实视频帧的分布距离）
- **VLM 评测**（借鉴 Physics-IQ 的 MLLM 评测）：给 VLM（如 Gemini / GPT-4V）展示原视频和编辑后视频，问"哪个更自然"（2AFC 范式）
- **人工 MOS**：人工整体质量打分 1-5

**指标**：
- **FID**：Frechet Inception Distance
- **VLM Realism Score**：VLM 能分辨出编辑视频的比例（越低越好，类似 Physics-IQ 的 MLLM score）
- **Overall MOS**：人工整体质量评分

**灵感来源**：Physics-IQ 的 MLLM evaluation（让 Gemini 做 2AFC 判断）；VBench 的 Aesthetic Quality + Imaging Quality

---

## 4. 评测协议

### 4.1 自动评测流程

```
输入：method(original_video, mask_video, instruction) → edited_video

对每个 Test-Paired 样本：
  1. Text Accuracy:
     - PARSeq OCR on mask region of edited_video → 与 target_text 对比
     - 输出: Word_Acc, Char_Acc, OCR_Rate

  2. Text Legibility:
     - Crop text patches from mask region
     - MUSIQ/CLIPIQA quality score
     - OCR confidence score
     - 输出: OCR_Conf, Patch_Quality

  3. Background Preservation:
     - Compare non-mask regions: edited_video vs original_video
     - 输出: BG-PSNR, BG-SSIM, BG-LPIPS

  4. Temporal Consistency:
     - Frame-to-frame stability in mask region
     - CLIP frame consistency
     - 输出: TTS, CFC

  5. Overall Realism:
     - FID across all edited frames vs all GT frames
     - 输出: FID

汇总：每个维度取所有样本的均值，得到方法在各维度的分数
```

### 4.2 人工评测流程

**评测规模**：从 230 对中抽取 50 个代表性样本

**评测内容**（每个样本每人打 3 个分）：
1. **文字可读性** (1-5)：编辑后的文字是否清晰可辨
2. **自然度** (1-5)：编辑结果看起来是否自然（无伪影/不协调）
3. **整体质量** (1-5)：综合考虑的整体满意度

**评测方式**（借鉴 Physics-IQ 的 2AFC + VBench 的人工标注）：
- 同时展示原始视频和编辑后视频
- 评测者知道编辑指令（如"Change ATP to NAD"）
- 每个样本至少 3 人标注，取均值
- 计算 inter-annotator agreement（SRCC，参考 LegiT 的做法）

### 4.3 按类别分析（借鉴 VBench 的 Per-Category 分析）

除了整体分数，还按以下类别拆分分析：

| 分类维度 | 类别 | 分析目的 |
|---------|------|--------|
| 文字类型 | Latin / CJK / Symbol | 不同文字系统的难度差异 |
| 文字长度 | 短(1-3) / 中(4-10) / 长(10+) | 文字复杂度的影响 |
| 运动程度 | 静态 / 轻微 / 明显 | 运动对编辑质量的影响 |
| 掩码大小 | 小(<3%) / 中(3-10%) / 大(>10%) | 编辑区域大小的影响 |

---

## 5. 评测方法（Baselines）

### 5.1 待评测方法列表

| 方法 | 类型 | 说明 |
|------|------|------|
| **Ours (Decomposed Pipeline)** | 分解式 | SAM3 + PISCO去除 + 图像编辑 + PISCO插入 |
| TextCtrl (NeurIPS'24) per-frame | 图像 STE 逐帧 | 每帧独立编辑，测时序不一致 |
| VACE Inpainting | 通用视频编辑 | VACE 直接微调做文字编辑（无特殊文字处理） |
| Glyph-VACE (v2) | 端到端 | Glyph 视觉编码 + cross-attention（我们实现） |
| CharEncoder-VACE (v3) | 端到端 | 字符级编码 + cross-attention（我们实现） |
| STRIVE (ICCV'21) | 前作 | GAN 方法（如可复现） |

### 5.2 预期结论

| 维度 | 预期最优方法 | 原因 |
|------|-----------|------|
| Text Accuracy | Ours (Pipeline) | 使用专用图像编辑模型，文字渲染质量高 |
| Text Legibility | Ours (Pipeline) | 同上 |
| Background Preservation | Ours (Pipeline) | PISCO 的去除+插入保证背景一致 |
| Temporal Consistency | Ours (Pipeline) / VACE | PISCO 和 VACE 都有时序建模 |
| Overall Realism | Ours (Pipeline) | 各子任务使用最优专用模型 |

同时我们预期发现：
- TextCtrl per-frame 在 Text Accuracy 上表现不错，但 Temporal Consistency 很差
- VACE Inpainting 背景保真好，但文字渲染差
- Glyph-VACE 受限于 glyph 渲染质量
- CharEncoder-VACE 文字完全不可读（数据量不足）

---

## 6. 与参考论文的对应关系

| 我们的设计 | VBench 的对应 | Physics-IQ 的对应 | LegiT 的对应 |
|---------|------------|---------------|-----------|
| 5 个评测维度 | 16 个维度 | 4 个指标 | legibility + quality |
| 自动评测 pipeline | Evaluation Method Suite | Spatial-IoU / MSE | IQA model + OCR |
| 人工评测 MOS | Human Preference Annotation | MLLM 2AFC evaluation | Human Study MOS |
| 按类别分析 | Per-Category (8类) | Per-Physics-Category (5类) | — |
| 230 真实对 GT | Prompt Suite + 生成视频 | 396 真实视频 GT | COCO-Text + LIVE-YT |
| 与人工一致性验证 | Spearman correlation | Pearson correlation | SRCC 0.88 |

---

## 7. 论文结构建议

```
40% 数据集制作 Pipeline
├── 3.1 任务定义与问题形式化
├── 3.2 分解式方法概述（流程图）
├── 3.3 Step 1: 文字区域分割（SAM3）
├── 3.4 Step 2: 视频文字去除（PISCO）
├── 3.5 Step 3: 单帧文字编辑（Nano Banana Pro）
├── 3.6 Step 4: 编辑文字重插入（微调 PISCO，核心技术贡献）
├── 3.7 数据集统计与质量分析
└── 3.8 与合成数据的质量对比

40% 评测基准
├── 4.1 评测维度设计（5 维度）
├── 4.2 自动评测方法（每个维度的具体指标 + pipeline）
├── 4.3 人工评测协议
├── 4.4 Baseline 方法
├── 4.5 实验结果（主表 + 雷达图 like VBench）
├── 4.6 按类别分析（文字类型 / 长度 / 运动）
└── 4.7 评测指标与人工感知的一致性验证

20% 端到端方法探索
├── 5.1 Glyph-VACE（方法 + 结果 + 局限分析）
├── 5.2 CharEncoder-VACE（方法 + 结果 + 失败分析）
└── 5.3 端到端 vs 分解式：讨论与启示
```
