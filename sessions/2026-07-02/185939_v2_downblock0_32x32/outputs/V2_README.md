# V2: UNet 瓶颈频域分析 — down_block_0 (32×32)

> **与 V1 的关键区别**: V1 分析 mid_block (8×8, 4 freq bins)——频率分辨率严重不足。
> V2 使用 **down_block_0 (320ch × 32×32, 16 freq bins)**——与原始 UNet 瓶颈 (~28×28) 分辨率相当。

## V1 vs V2 对比

| | V1 (mid_block) | V2 (down_block_0) |
|---|---|---|
| 空间尺寸 | 8×8 | **32×32** |
| 径向频率 bin 数 | 4 | **16** |
| 频谱斜率 | 全部 0.000（无意义） | **-1.22 ~ -0.37**（有意义） |
| 低/中/高频段 | 无法划分（bin 不足） | **正常划分** |
| DC 比率范围 | 42% ~ 80% | **50% ~ 96%**（更宽） |
| 输入类型区分度 | 低（周期信号和纯色混淆） | **高**（各类型清晰聚类） |

## V2 核心发现（32×32 分辨率）

### 频谱斜率（最有区分力的单一指标）

```
噪声/纹理:         slope > -0.5   (平坦谱, 能量均匀分布)
自然图像:          slope ~ -0.73  (中等衰减)
图案/正弦:         slope ~ -0.6~-0.9 (有结构衰减)
纯色:              slope ~ -0.8~-0.9 (强衰减)
渐变:              slope < -1.0   (最陡, 能量集中在极低频)
```

### 频段能量分布（16 bins → 真正的三频段分析）

| 输入类型 | Low | Mid | High | 特征 |
|---------|:---:|:---:|:---:|------|
| gradient_radial | 63.6% | 20.2% | 16.1% | 极低频主导 |
| natural_astronaut | 42.4% | 28.0% | 29.6% | 均衡分布 |
| noise_std_0.5 | 29.7% | 29.8% | 40.5% | 高频偏重 |

### 为何 V2 更可信

1. **斜率有意义**: 16 个数据点做 log-log 拟合，r² > 0.8
2. **频段可划分**: 低(1-2), 中(2-5), 高(5+) bins 各有足够样本
3. **区分度提升**: 输入类型间的频谱差异在 32×32 下被放大

## 文件结构

```
185939_v1_midblock_8x8/        ← V1（旧，mid_block 8×8）
  ├── src/
  ├── outputs/
  │   ├── data/spectra_results.json
  │   ├── inputs/ (45 png)
  │   ├── spectra/ (12 detail png)
  │   ├── report/ (9 figures)
  │   └── UNet_Bottleneck_FFT_Analysis_Report.md

185939_v2_downblock0_32x32/    ← V2（新，down_block_0 32×32）
  ├── src/
  └── outputs/
      ├── data/spectra_results.json
      ├── inputs/ (45 png)
      ├── spectra/ (12 detail png)
      ├── report/ (3 figures: slope comparison, band energy, slope vs DC)
      └── V2_README.md
```
