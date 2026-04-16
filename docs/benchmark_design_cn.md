# ReWording Benchmark：视频场景文字编辑评测基准

---

## 总览

ReWording Benchmark 是首个面向**视频场景文字编辑**的综合评测框架，围绕 **3 个评测轴、9 个指标**设计：

| 轴 | 核心问题 | 指标 | 方向 |
|:--:|---------|------|:---:|
| **文字正确性** | 文字编辑对不对？ | SeqAcc↑, CharAcc↑, TTS↑ | 越高越好 |
| **视觉质量** | 看起来好不好？稳不稳？ | Flickering↑, MUSIQ↑, FVD↓ | 见标注 |
| **上下文保真度** | 文字以外的区域动没动？ | PSNR↑, SSIM↑, LPIPS↓ | 见标注 |

评测数据集包含 **230 个高质量真实视频编辑配对**（1280×720, 24fps, 120帧），每个样本含原始视频、编辑后GT视频、文字区域掩码和编辑指令。

---

## 轴 1：文字正确性（Text Correctness）

> 编辑后的视频里，目标文字有没有被正确渲染？

三个指标共用一次 **PP-OCRv5** 推理：对每帧裁剪文字掩码 bbox 区域，送入 OCR 识别，输出统一转大写后与目标文字对比。

### SeqAcc（序列准确率）

$$\text{SeqAcc} = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[\text{OCR}(t) = \text{target}]$$

逐帧**完全匹配**率。最严格的指标，差一个字符即判为失败。

> 目标 "CLOSED"，OCR 每帧识别为 "CLOSD" → SeqAcc = 0

### CharAcc（字符准确率）

$$\text{CharAcc} = \frac{1}{T} \sum_{t=1}^{T} \left(1 - \frac{\text{EditDist}(\text{OCR}(t),\ \text{target})}{\max(|\text{OCR}(t)|,\ |\text{target}|)}\right)$$

逐帧**字符级**准确率，基于归一化 Levenshtein 编辑距离。允许部分正确。

> 目标 "CLOSED"，OCR 识别为 "CLOSD" → CharAcc = 1 - 1/6 ≈ 0.83

### TTS（时序文字稳定性）

$$\text{TTS} = \frac{1}{T-1} \sum_{t=1}^{T-1} \mathbb{1}[\text{OCR}(t) = \text{OCR}(t+1)]$$

相邻帧 OCR 结果是否一致。**不关心文字对错，只关心帧间稳定性**。

TTS 必须与 SeqAcc/CharAcc 搭配看：一个方法每帧渲染错误但一致的文字，TTS = 1.0 但 SeqAcc = 0。

**设计目的**：逐帧图像编辑方法（如 TextCtrl 逐帧应用）每帧独立推理导致帧间文字抖动，TTS 专门暴露此弱点。

---

## 轴 2：视觉质量（Visual Quality）

> 编辑后的视频看起来好不好？时序上稳不稳？

本轴所有指标均在**完整帧**（非裁剪）上计算，评估整体视觉质量。

### Flickering（时序稳定性）↑

$$\text{Flickering} = \frac{255 - \frac{1}{T-1}\sum_{t=1}^{T-1}\text{MAE}(f_t, f_{t+1})}{255}$$

全帧相邻帧间的像素级 MAE，经 VBench 风格归一化到 0~1 区间。**越高越稳定**（1.0 表示完全无闪烁）。纯像素运算，无需任何预训练模型。

> 参考 VBench Temporal Flickering 的计算方式：`cv2.absdiff` → 均值 → `(255 - MAE) / 255`

### MUSIQ（图像质量）↑

对每帧 resize 到 max 512px 后，输入 **MUSIQ**（Google，多尺度图像质量评估 Transformer, KonIQ 预训练）得到质量分（0~100），取所有帧均值。

> 参考 VBench Imaging Quality 的计算方式：resize → MUSIQ 逐帧评分 → 取均值

### FVD（Frechet 视频距离）↓

对测试样本的输出视频的数据集中已有的的高质量编辑后视频，用预训练 **R3D-18**（Kinetics-400 预训练）提取视频级特征，拟合高斯分布后计算 Frechet 距离。

**全局指标**（整个测试集算一个值），视频生成/编辑领域广泛使用的整体质量指标，同时捕捉空间质量和时序连贯性。

---

## 轴 3：上下文保真度（Context Fidelity）

> 文字以外的区域有没有被破坏？

三个指标均只在**非文字区域**（掩码取反）上计算，对比**编辑后视频 vs 原始输入视频**。

### PSNR（峰值信噪比）↑

$$\text{PSNR}_{\text{bg}} = \frac{1}{T}\sum_{t=1}^{T} 10 \cdot \log_{10}\frac{255^2}{\text{MSE}_{\bar{M}}(f^{\text{orig}}_t,\ f^{\text{edit}}_t)}$$

非文字区域的像素级信噪比。完美保留背景时 PSNR → ∞，整张图被重新生成时 PSNR 显著下降。

### SSIM（结构相似度）↑

非文字区域的结构相似度，综合考虑**亮度、对比度、结构**三个维度。比 PSNR 更贴近人类视觉感知：轻微色差不影响 SSIM，但结构破坏（如边缘模糊）会被敏感捕捉。

### LPIPS（深度感知距离）↓

将两帧被掩码的文字区域像素置为相同值后，用 **LPIPS** 计算深度感知距离。能捕捉 PSNR/SSIM 遗漏的感知差异。

---


### 指标选取依据

| 指标 | 灵感来源 | 入选理由 |
|------|---------|---------|
| SeqAcc / CharAcc | LegiT (CVPR'24) | 严格+宽松的文字准确度，场景文字评测标准做法 |
| TTS | **本工作原创** | 视频特有——捕捉帧间文字一致性，区分逐帧 vs 视频方法 |
| Flickering | VBench (CVPR'24) | 全帧像素级时序稳定性，VBench 归一化标准 |
| MUSIQ | VBench (CVPR'24) | 图像感知质量，无参考指标，resize 到 512px 评估 |
| FVD | 视频生成标准指标 | 编辑视频与 GT 视频的分布级质量度量 |
| PSNR / SSIM | 标准图像质量指标 | 非文字区域像素/结构保真度 |
| LPIPS | Zhang et al. (CVPR'18) | 深度感知距离，补充 PSNR/SSIM |

---

## 实现方案

精确到使用的模型、库和计算方式：

### 轴1：文字正确性（3个指标）

| 指标 | 工具 | 具体计算 |
|------|------|---------|
| **SeqAcc** | PP-OCRv5 | 对输出视频的每一帧，根据对应帧的掩码取外接矩形裁剪出文字区域，送入 OCR 识别。OCR 输出与目标文字**完全相同**的帧数 / 总帧数。 |
| **CharAcc** | PP-OCRv5 | 对每一帧的 OCR 输出，计算与目标文字之间的 Levenshtein 编辑距离，归一化为 `1 - 编辑距离 / max(预测长度, 目标长度)`，然后取所有帧的均值。允许部分字符正确（例如目标 "CLOSED"、识别为 "CLOSD"，得分 0.83）。 |
| **TTS** | PP-OCRv5 | 对第1帧和第2帧、第2帧和第3帧……这样成对比较相邻两帧的 OCR 结果，**完全相同**记 1 否则记 0，取所有 T-1 个帧对的均值。只关心帧间一致性，不关心内容对错。 |

三个指标共用一次 OCR 推理结果。PP-OCRv5 在独立 conda 环境 `paddleocr` 中运行，CPU 模式推理。比较时统一 `strip().upper()` 消除大小写和首尾空格差异。

### 轴2：视觉质量（3个指标）

| 指标 | 工具 | 具体计算 |
|------|------|---------|
| **Flickering** | `cv2.absdiff`（纯像素运算，无模型） | 对第 t 帧和第 t+1 帧取**整幅画面**的像素级绝对差的均值（MAE），对所有 T-1 个帧对取平均得到平均 MAE，然后按 VBench 风格归一化为 `(255 − 平均MAE) / 255`，映射到 0~1。**越高越稳定**（1.0 表示完全无变化）。 |
| **MUSIQ** | pyiqa 库（`musiq` 模型，KonIQ 预训练） | 对每一帧先 resize 使最大边不超过 512 像素（VBench 做法），然后送入 MUSIQ 模型得到一个 0~100 的质量分，取所有帧均值。评估图像是否清晰锐利、有无模糊/锯齿/伪影。↑越高越好。 |
| **FVD** | PyTorch, R3D-18（Kinetics-400 预训练） | **全局指标**，整个测试集只算一个分数。对每条输出视频和参考分布中的每条视频（数据集自带的全部高质量编辑后视频，**不按样本 ID 配对**，而是作为整体参考分布），各自用 R3D-18 均匀采样 16 帧、resize 到 112×112，提取 512 维视频级特征。两组特征分别拟合多维高斯分布，计算两分布之间的 Frechet 距离。↓越低越好。 |

### 轴3：上下文保真度（3个指标）

本轴评估"编辑过程有没有破坏文字以外的内容"。所有指标都**对比输出视频 vs 原始输入视频**，并**只在非文字区域**（掩码取反）上计算。

| 指标 | 工具 | 具体计算 |
|------|------|---------|
| **PSNR** | numpy | 对每一帧，用该帧掩码取反得到"非文字区域"的布尔掩码，只取输出视频和原始视频在该区域的像素计算 MSE，然后 `PSNR = 10 × log10(255² / MSE)`，取所有帧均值。MSE 为 0 时封顶 100 dB。↑越高越好。 |
| **SSIM** | scikit-image | 先把**输出视频的文字区域像素替换为原始视频对应像素**，让文字区域在两张图中完全相同（对 SSIM 贡献满分、自然抵消），然后对整幅画面算 SSIM，取所有帧均值。↑越高越好。 |
| **LPIPS** | lpips 库（AlexNet backbone，Zhang et al. CVPR 2018） | 同 SSIM 的掩码处理：把输出视频的文字区域像素替换为原始视频对应像素，使文字区域贡献 0 距离。再把两帧归一化到 [-1, 1] 送入 LPIPS 得到感知距离，取所有帧均值。↓越低越好。 |

---

## 汇总表

| # | 指标 | 所属轴 | 方向 | 工具 | 需要的参考数据 |
|---|------|--------|:---:|------|-----------|
| 1 | SeqAcc | 文字正确性 | ↑ | PP-OCRv5 | 目标文字字符串 |
| 2 | CharAcc | 文字正确性 | ↑ | PP-OCRv5 | 目标文字字符串 |
| 3 | TTS | 文字正确性 | ↑ | PP-OCRv5 | 无（仅需编辑后视频） |
| 4 | Flickering | 视觉质量 | ↑ | cv2.absdiff + 归一化 | 无（仅需编辑后视频） |
| 5 | MUSIQ | 视觉质量 | ↑ | pyiqa（KonIQ 预训练） | 无（仅需编辑后视频） |
| 6 | FVD | 视觉质量 | ↓ | R3D-18（Kinetics-400） | 数据集中自带的高质量编辑后视频 |
| 7 | PSNR | 上下文保真度 | ↑ | scikit-image | 原始视频 + 掩码 |
| 8 | SSIM | 上下文保真度 | ↑ | scikit-image | 原始视频 + 掩码 |
| 9 | LPIPS | 上下文保真度 | ↓ | lpips（AlexNet） | 原始视频 + 掩码 |


---

## 评测运行方式

```bash
# Step 1: OCR 提取（paddleocr 环境，CPU 模式）
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
conda run -n paddleocr python benchmark/ocr_extract.py \
    --video_dir outputs/<method_name>/ \
    --output outputs/<method_name>/ocr_results.json

# Step 2: 全指标评测（DiffSynth-Studio 环境，GPU）
conda run -n DiffSynth-Studio python benchmark/evaluate.py \
    --video_dir outputs/<method_name>/ \
    --ocr_results outputs/<method_name>/ocr_results.json
```
