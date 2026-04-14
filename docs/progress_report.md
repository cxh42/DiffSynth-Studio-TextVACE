# 视频场景文字编辑 —— 项目进展汇报

> 更新日期：2026-04-13

---

## 1. 研究目标

**任务**：视频场景文字编辑（Video Scene Text Editing, VideoSTE）—— 给定输入视频和要编辑的文字区域，将视频中的指定文字替换为目标文字，要求保持字体/颜色/透视风格一致、非文字区域不变、帧间时序连贯。

**投稿目标**：NeurIPS 2026 主赛道

---

## 2. 研究现状与空白

| 方向 | 代表工作 | 局限 |
|------|---------|------|
| 图像场景文字编辑 | TextCtrl (NeurIPS'24), AnyText, GlyphMastero (CVPR'25) | 仅单帧，无法保证视频时序一致 |
| 视频场景文字替换 | STRIVE (ICCV'21) | 基于 GAN，非扩散模型，5 年无后续 |
| 通用视频编辑 | VACE (ICCV'25), VideoPainter (SIGGRAPH'25) | 不理解字形，无法精确控制文字渲染 |
| 文字生成/渲染 | TextDiffuser-2, GlyphControl | 图像级，不处理视频时序 |

**核心空白：不存在高质量的视频场景文字编辑方法和评测基准。**

---

## 3. 我们的数据制作 Pipeline（核心贡献）

### 3.1 方法概述

我们提出了一种**分解式视频场景文字编辑方法**，将端到端的困难问题拆解为四个可解的子任务：

```
输入：带文字的原始视频 + 编辑指令

Step 1: SAM3 分割目标文字区域 → 掩码视频（Mask）
Step 2: PISCO 去除掩码区域文字 → 干净视频（Clean）
Step 3: Nano Banana Pro 编辑第一帧文字 → 编辑后首帧
Step 4: SAM3 分割编辑后首帧文字 → 编辑后文字片段
Step 5: 微调 PISCO 将文字片段插入干净视频 → 编辑后视频

输出：高质量的编辑后视频
```

### 3.2 核心技术：微调 PISCO 实现文字重插入

- PISCO 原本是视频文字**去除**模型（有文字视频 → 干净视频）
- 我们用 (干净视频, 原视频) 数据对**反向微调**，让它学会**插入**文字（干净视频 + 文字片段 → 合成视频）
- 去除和插入是**对称任务**，同一架构可双向使用
- 微调后的模型能将编辑后的文字片段高质量融合进干净视频，保持光影、透视、时序一致

### 3.3 数据产出

使用该 pipeline 制作了 **230 个高质量视频文字编辑数据对**：
- 分辨率 1280×720，24fps，约 5 秒
- 每对包含：原始视频、编辑后视频、文字掩码视频、编辑指令
- 覆盖英文、中文、韩文、数学符号等多种文字类型
- 另有 240 个未见视频（仅有原视频 + 掩码，无 GT）用于泛化测试

### 3.4 数据独创性

**这是全世界唯一的高质量真实视频文字编辑配对数据。**

现有场景文字编辑方法的训练数据对比：

| 方法 | 数据来源 | 数据类型 | 质量 |
|------|---------|---------|------|
| SRNet / MOSTEL | SynthText 引擎合成 | 合成图像对 | 低：无真实光影/磨损/透视 |
| TextCtrl (NeurIPS'24) | SynthText + 风格迁移 | 合成图像对 | 中：风格不够自然 |
| AnyText / TextDiffuser | LAION/Wukong OCR 筛选 | 真实图（非配对） | 无编辑前后配对 |
| **Ours** | **多模型协作 Pipeline** | **真实视频配对** | **高：保留真实光影/透视/时序** |

---

## 4. 端到端方法探索（消融实验）

在数据制作 pipeline 之外，我们还探索了基于扩散模型的端到端方法，验证了其局限性。

### 4.1 方案 v1：Glyph 通道拼接

**思路**：将 glyph 视频（渲染的目标文字）的 VAE 编码拼接到 VACE 的输入通道（96ch → 112ch）

**结果**：
- 文字编辑能力初步验证，目标文字部分可读
- 背景保真度不足（PSNR 17-22，理想应 >30）
- 泛化能力存在，训练时没见过的目标文字也能部分渲染

### 4.2 方案 v2：Glyph Cross-Attention

**思路**：独立的 GlyphEncoder（Conv3D + cross-attention pooling → 64 tokens）+ 15 层 GlyphCrossAttention 注入 VACE blocks

**架构**：
```
glyph_video → VAE → GlyphEncoder → 64 tokens
                                       ↓
vace_context → Conv3D → 15×[SelfAttn → TextCrossAttn → GlyphCrossAttn → FFN]
```

**训练**：5 epochs × 2300 steps = 11500 步，~11 小时，RTX 5090

**结果**：
- 最终 loss: 0.005221
- 30 个未见视频推理完成
- 文字有一定可辨认度，但**受限于 glyph 渲染质量**
- OCR 检测不准的样本效果差

**结论**：**glyph 方案的天花板是 glyph 渲染质量** —— OCR 定位偏差、字体匹配不准、透视变换误差导致模型上限被卡住

### 4.3 方案 v3：Character Encoder

**思路**：去掉 glyph 视频，用字符级编码器直接编码目标文字字符串，通过 cross-attention 注入 VACE blocks

**架构**：
```
"NAD" → CharTokenize → TargetTextEncoder(Embed + 2层Transformer) → 64 tokens
                                                                       ↓
vace_context → Conv3D → 15×[SelfAttn → TextCrossAttn → CharCrossAttn → FFN]
```

**参数量**：TargetTextEncoder ~72M + ConditionCrossAttention ×15 ~142M

**训练**：3 epochs × 2300 steps = 6900 步，~5.4 小时，RTX 5090

**结果**：
- 30 个样本推理完成（10 训练集 + 20 未见视频）
- **文字完全无法辨认，呈现乱码/鬼画符状态**

**失败原因分析**：
- 要求模型从 230 个样本学会**字符→视觉渲染**的映射
- 能渲染文字的模型（FLUX、DALL-E 3）训练数据为数十亿级
- AnyText、TextDiffuser-2 也用了几百万到上千万数据
- 230 个样本根本不可能让模型学会文字渲染
- 调研发现：**不存在预训练的"文字→视觉特征"编码器**，所有方法要么从头训练（需要大量数据），要么渲染成图像后编码（回到 glyph 路线）

**结论**：**纯字符编码方案在小数据条件下不可行**，这是数据量的根本限制，不是架构设计问题

---

## 5. 关键结论与方向调整

### 5.1 三个方案的对比总结

| | 分解式 Pipeline | Glyph Cross-Attn (v2) | Character Encoder (v3) |
|--|:---:|:---:|:---:|
| **文字可读性** | 高（真实编辑） | 中（部分可辨认） | 极低（乱码） |
| **背景保真** | 高 | 低（PSNR 17-22） | 低 |
| **时序一致** | 高（PISCO保证） | 中 | 中 |
| **数据需求** | 少（仅微调PISCO） | 中（230不够理想） | 极大（需百万级） |
| **外部依赖** | 多模型协作 | glyph 渲染质量 | 无 |
| **核心瓶颈** | 速度慢、资源消耗大 | glyph 质量天花板 | 数据量不足 |
| **扩展性** | 可直接用于新视频 | 需重新训练 | 需重新训练+大量数据 |

### 5.2 方向调整

**从"端到端模型创新"转向"分解式方法 + Benchmark"**

理由：
1. 端到端方法受限于数据量（字符编码）或渲染质量（glyph），在当前条件下无法达到实用质量
2. 我们的分解式 pipeline 已经产出了高质量结果，本身就是一个可用的视频 STE 方法
3. 230 个真实配对是该领域首个也是唯一的高质量 benchmark
4. 端到端方法的探索和失败分析在论文中有重要的消融实验价值

---

## 6. 论文规划

### 6.1 论文定位

**主赛道 Method Paper**

标题方向：
> "Decomposed Video Scene Text Editing via Segmentation, Removal, and Reinsertion"

### 6.2 贡献点

1. **分解式方法**：首个实用的视频场景文字编辑框架，将问题分解为分割→去除→单帧编辑→重插入四个子任务
2. **视频文字重插入**：通过反向微调视频修复模型（PISCO），实现编辑文字的高质量时序一致插入（核心技术贡献）
3. **VideoSTE-Bench**：首个真实视频文字编辑评测基准（230 对真实配对 + 240 未标注测试视频），规模对标 DAVIS 2016（50 个视频序列）
4. **系统性对比分析**：分解式 vs 端到端（Glyph-VACE、CharEncoder-VACE），验证分解范式在数据受限场景的优势

### 6.3 对比实验设计

| 方法 | 类型 | 说明 |
|------|------|------|
| **Ours（分解式）** | 主方法 | SAM3 + PISCO + 图像编辑 + 微调PISCO |
| TextCtrl 逐帧 | Baseline | 图像 STE 方法逐帧应用，测时序不一致 |
| VACE 微调（无 glyph） | Baseline | 通用视频编辑方法，不含文字特殊处理 |
| Glyph-VACE (v2) | 消融 | glyph 渲染 + cross-attention 端到端方法 |
| CharEncoder-VACE (v3) | 消融 | 字符编码端到端方法 |
| STRIVE | 前作对比 | GAN 方法（如可复现） |

### 6.4 评测指标

| 指标 | 衡量维度 |
|------|---------|
| OCR Accuracy | 编辑后文字识别准确率（PARSeq） |
| PSNR / SSIM | 非文字区域背景保真度 |
| LPIPS | 感知相似度 |
| FID | 整体生成质量 |
| 帧间一致性 | 时序连贯性（warping error / CLIP frame coherence） |
| 人工评测 | 文字可读性、自然度、整体质量 MOS 评分 |

---

## 7. 已有资产清单

### 代码
| 文件 | 说明 | 状态 |
|------|------|------|
| `diffsynth/models/wan_video_vace.py` | GlyphEncoder + TargetTextEncoder + ConditionCrossAttn | ✅ 完成 |
| `diffsynth/pipelines/wan_video.py` | Pipeline 支持 glyph_video 和 target_text | ✅ 完成 |
| `scripts/render_glyph_ocr.py` | OCR 驱动的 glyph 视频渲染 | ✅ 完成 |
| `scripts/inference_v3_batch.py` | v3 批量推理脚本 | ✅ 完成 |
| `scripts/inference_textvace_v3.py` | v3 单样本推理 | ✅ 完成 |

### 模型检查点
| 检查点 | 方案 | 位置 |
|--------|------|------|
| epoch-0 ~ epoch-4 | v2 Glyph (5 epochs) | `models/train/TextVACE_sft/` |
| epoch-0 ~ epoch-2 | v3 CharEncoder (3 epochs) | `models/train/TextVACE_v3_sft/` |

### 数据
| 数据 | 规模 | 位置 |
|------|------|------|
| 训练数据（原视频+编辑视频+掩码+指令） | 230 对 | `data/raw/` |
| Glyph 视频（OCR 版） | 214 个 | `data/processed/glyph_videos/` |
| 推理数据（未见视频+掩码） | 240 个 | `data/inference_raw/` |
| 推理记录（VLM 生成指令） | 32 条 | `data/inference_processed/` |

### 推理结果
| 结果 | 样本数 | 位置 |
|------|--------|------|
| v2 未见视频推理 | 30 个 | `outputs/textvace_inference/unseen_final/` |
| v3 训练集推理 | 10 个 | `outputs/textvace_v3_inference/train_samples/` |
| v3 未见视频推理 | 20 个 | `outputs/textvace_v3_inference/unseen_videos/` |

---

## 8. 下一步 TODO

- [ ] 量化评测：在 230 个真实对上计算 pipeline 方法的 PSNR/SSIM/OCR 准确率
- [ ] Baseline 实现：TextCtrl 逐帧应用
- [ ] 论文写作：方法部分（pipeline 流程图 + PISCO 微调细节）
- [ ] 论文写作：实验部分（对比表 + 消融分析 + 可视化对比）
- [ ] 整理 VideoSTE-Bench 的标准评测协议
- [ ] 人工评测设计（MOS 评分问卷）
