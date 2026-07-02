#!/usr/bin/env python3
"""Generate v2 report: down_block_0 (32x32) as primary analysis layer."""

import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from PIL import Image

_fp = '/home/tianyi/.local/share/fonts/wqy-microhei.ttc'
if os.path.exists(_fp):
    fontManager.addfont(_fp)
    plt.rcParams['font.family'] = FontProperties(fname=_fp).get_name()
plt.rcParams['axes.unicode_minus'] = False

V2_DIR = "/mnt/data/tianyi/ClaudeMisc/sessions/2026-07-02/185939_v2_downblock0_32x32"
V1_DIR = "/mnt/data/tianyi/ClaudeMisc/sessions/2026-07-02/185939_v1_midblock_8x8"
OUT_DIR = os.path.join(V2_DIR, "outputs/report")
os.makedirs(OUT_DIR, exist_ok=True)
INPUT_DIR = os.path.join(V2_DIR, "outputs/inputs")

with open(os.path.join(V2_DIR, "outputs/data/spectra_results.json")) as f:
    v2_results = json.load(f)
with open(os.path.join(V1_DIR, "outputs/data/spectra_results.json")) as f:
    v1_results = json.load(f)

LAYER_V2 = "down_block_0"
LAYER_V1 = "mid_block"
TIMESTEP = "500"

# ---------------------------------------------------------------------------
# FIG A: v1 vs v2 spectral slope comparison — the key improvement
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Compute slopes for both versions
def get_slope(spec):
    n = len(spec)
    if n <= 3: return 0
    freqs = np.arange(1, n)
    return np.polyfit(np.log10(freqs), np.log10(spec[1:]+1e-10), 1)[0]

names_v1, slopes_v1 = [], []
for r in v1_results:
    ts = r['timesteps'].get(TIMESTEP, {})
    if LAYER_V1 in ts:
        spec = np.array(ts[LAYER_V1]['mean_spectrum'])
        names_v1.append(r['name']); slopes_v1.append(get_slope(spec))

names_v2, slopes_v2 = [], []
for r in v2_results:
    ts = r['timesteps'].get(TIMESTEP, {})
    if LAYER_V2 in ts:
        spec = np.array(ts[LAYER_V2]['mean_spectrum'])
        names_v2.append(r['name']); slopes_v2.append(get_slope(spec))

# V1 panel
ax = axes[0]
for n, s in sorted(zip(names_v1, slopes_v1), key=lambda x: x[1]):
    c = 'red' if 'noise' in n else 'gray' if 'solid' in n else 'blue' if 'gradient' in n else 'orange' if 'checkerboard' in n or 'sine' in n else 'green' if 'texture' in n else 'purple'
    ax.barh(n[:22], s, color=c, height=0.6, alpha=0.8)
ax.axvline(0, color='black', ls='--', alpha=0.3)
ax.set_title(f'V1: mid_block (8×8, 4 freq bins)\nAll slopes = 0 — no frequency discrimination', fontsize=11, fontweight='bold', color='#C0392B')
ax.set_xlabel('Spectral Slope (log-log)')
ax.grid(axis='x', alpha=0.2)
ax.set_xlim(-0.05, 0.05)

# V2 panel
ax = axes[1]
sorted_v2 = sorted(zip(names_v2, slopes_v2), key=lambda x: x[1])
for n, s in sorted_v2:
    c = 'red' if 'noise' in n else 'gray' if 'solid' in n else 'blue' if 'gradient' in n else 'orange' if 'checkerboard' in n or 'sine' in n else 'green' if 'texture' in n else 'purple'
    ax.barh(n[:22], s, color=c, height=0.6, alpha=0.8)
ax.axvline(0, color='black', ls='--', alpha=0.3)
ax.set_title(f'V2: down_block_0 (32×32, 16 freq bins)\nSlopes range -1.22 to -0.37 — meaningful frequency discrimination', fontsize=11, fontweight='bold', color='#27AE60')
ax.set_xlabel('Spectral Slope (log-log)')
ax.grid(axis='x', alpha=0.2)

fig.suptitle('V1 vs V2: 频谱斜率的分辨率对比\n'
             'V1 (8×8瓶颈) 所有斜率=0，V2 (32×32) 斜率分布有意义，可区分不同输入类型',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figA_v1_vs_v2_slope_comparison.png"), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("[*] FigA saved.")

# ---------------------------------------------------------------------------
# FIG B: V2 frequency band energy (Low/Mid/High — now meaningful)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Stacked bar: Low/Mid/High for each input
band_data = {}
for r in v2_results:
    ts = r['timesteps'].get(TIMESTEP, {})
    if LAYER_V2 not in ts: continue
    spec = np.array(ts[LAYER_V2]['mean_spectrum'])
    n = len(spec)
    total_ac = np.sum(spec[1:]) + 1e-10
    low = np.sum(spec[1:max(1,n//8)]) / total_ac * 100
    mid = np.sum(spec[n//8:n//3]) / total_ac * 100
    high = np.sum(spec[n//3:]) / total_ac * 100
    band_data[r['name']] = [low, mid, high]

sorted_names = sorted(band_data.keys(), key=lambda n: band_data[n][2])  # sort by high-freq
low_v = [band_data[n][0] for n in sorted_names]
mid_v = [band_data[n][1] for n in sorted_names]
high_v = [band_data[n][2] for n in sorted_names]

ax = axes[0]
y = range(len(sorted_names))
ax.barh(y, low_v, color='#1f77b4', label=f'Low (bin 1-{max(1,16//8)})', height=0.7)
ax.barh(y, mid_v, color='#ff7f0e', label=f'Mid (bin {16//8}-{16//3})', left=low_v, height=0.7)
ax.barh(y, high_v, color='#d62728', label=f'High (bin {16//3}-16)', left=[l+m for l,m in zip(low_v,mid_v)], height=0.7)
ax.set_yticks(y)
ax.set_yticklabels([n[:25] for n in sorted_names], fontsize=5)
ax.set_xlabel('% of AC Power')
ax.set_title('V2: Frequency Band Energy Distribution (16 bins)\nLow/Mid/High bands now have actual bins allocated', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')

# V2 DC ratio bar chart
ax = axes[1]
dc_data = {}
for r in v2_results:
    ts = r['timesteps'].get(TIMESTEP, {})
    if LAYER_V2 not in ts: continue
    spec = np.array(ts[LAYER_V2]['mean_spectrum'])
    dc_data[r['name']] = spec[0] / np.sum(spec) * 100

sorted_dc = sorted(dc_data.items(), key=lambda x: x[1])
names_dc, vals_dc = zip(*sorted_dc)
colors = []
for n in names_dc:
    if 'noise' in n: colors.append('#E74C3C')
    elif 'solid' in n: colors.append('#95A5A6')
    elif 'gradient' in n: colors.append('#3498DB')
    elif 'checkerboard' in n or 'sine' in n: colors.append('#E67E22')
    elif 'texture' in n: colors.append('#1ABC9C')
    elif 'natural' in n: colors.append('#C0392B')
    elif 'edge' in n or 'contour' in n: colors.append('#9B59B6')
    else: colors.append('#7F8C8D')
ax.barh(range(len(names_dc)), vals_dc, color=colors, height=0.7)
ax.set_yticks(range(len(names_dc)))
ax.set_yticklabels([n[:25] for n in names_dc], fontsize=5)
ax.set_xlabel('DC Power %')
ax.set_title('V2: DC Ratio by Input (32×32)\nWider spread than V1: 50% (natural) to 96% (checkerboard)', fontsize=10, fontweight='bold')
ax.grid(axis='x', alpha=0.2)

fig.suptitle('V2 (down_block_0, 32×32): 有意义的频段分析', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figB_v2_band_energy.png"), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("[*] FigB saved.")

# ---------------------------------------------------------------------------
# FIG C: Slope vs DC scatter — V2 shows clear clustering
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

for r in v2_results:
    ts = r['timesteps'].get(TIMESTEP, {})
    if LAYER_V2 not in ts: continue
    spec = np.array(ts[LAYER_V2]['mean_spectrum'])
    dc = spec[0] / np.sum(spec) * 100
    slope = get_slope(spec)
    n = r['name']
    c = '#E74C3C' if 'noise' in n else '#95A5A6' if 'solid' in n else '#3498DB' if 'gradient' in n else '#E67E22' if 'checkerboard' in n or 'sine' in n else '#1ABC9C' if 'texture' in n else '#C0392B' if 'natural' in n else '#9B59B6' if 'edge' in n or 'contour' in n else '#7F8C8D'
    ax.scatter(slope, dc, c=c, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.annotate(n[:18], (slope, dc), fontsize=5, alpha=0.8, xytext=(3,3), textcoords='offset points')

ax.set_xlabel('Spectral Slope (log-log, more negative = faster decay)', fontsize=11)
ax.set_ylabel('DC Ratio %', fontsize=11)
ax.set_title('V2 (32×32): Slope vs DC Ratio — Clear Input Clustering\n'
             'Upper-right = uniform (high DC, steep decay) | Lower-right = structured (low DC, flat spectrum)',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2)
# Annotations
ax.annotate('渐变\n(陡峭衰减+高DC)', xy=(-1.2, 85), fontsize=9, color='#3498DB', fontweight='bold',
            bbox=dict(facecolor='#D6EAF8', alpha=0.7))
ax.annotate('周期图案\n(中等斜率+高DC)', xy=(-0.8, 92), fontsize=9, color='#E67E22', fontweight='bold',
            bbox=dict(facecolor='#FDEBD0', alpha=0.7))
ax.annotate('纹理/噪声\n(平坦谱+中高DC)', xy=(-0.4, 82), fontsize=9, color='#1ABC9C', fontweight='bold',
            bbox=dict(facecolor='#D1F2EB', alpha=0.7))
ax.annotate('自然图像\n(中等斜率+低DC)', xy=(-0.72, 50), fontsize=9, color='#C0392B', fontweight='bold',
            bbox=dict(facecolor='#FADBD8', alpha=0.7))

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', label='噪声'), Patch(facecolor='#95A5A6', label='纯色'),
    Patch(facecolor='#3498DB', label='渐变'), Patch(facecolor='#E67E22', label='棋盘格/正弦'),
    Patch(facecolor='#1ABC9C', label='纹理'), Patch(facecolor='#C0392B', label='自然图像'),
    Patch(facecolor='#9B59B6', label='边缘/轮廓'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "figC_v2_slope_vs_dc.png"), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("[*] FigC saved.")

print("Done. V2 report figures in:", OUT_DIR)
