#!/usr/bin/env python3
"""
Generate the key figure: input images → bottleneck frequency response.
Goal: one glance shows the relationship between input features and freq-domain behavior.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import os

# Register and use CJK font
_font_path = '/home/tianyi/.local/share/fonts/wqy-microhei.ttc'
if os.path.exists(_font_path):
    fontManager.addfont(_font_path)
    # Find the actual family name
    from matplotlib.font_manager import FontProperties
    _fp = FontProperties(fname=_font_path)
    _family = _fp.get_name()
    plt.rcParams['font.family'] = _family
else:
    _family = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
print(f"Using font: {_family}")
from matplotlib.colors import LogNorm
from PIL import Image
import os

DATA = "/mnt/data/tianyi/ClaudeMisc/sessions/2026-07-02/185939/outputs/data/spectra_results.json"
INPUT_DIR = "/mnt/data/tianyi/ClaudeMisc/sessions/2026-07-02/185939/outputs/inputs"
OUT_DIR = "/mnt/data/tianyi/ClaudeMisc/sessions/2026-07-02/185939/outputs/report"

with open(DATA) as f:
    results = json.load(f)

TIMESTEP = '500'
LAYER = 'mid_block'

# ---------------------------------------------------------------------------
# 1. Collect metrics and sort by DC ratio (low → high)
# ---------------------------------------------------------------------------
entries = []
for r in results:
    ts = r['timesteps'].get(TIMESTEP, {})
    if LAYER not in ts:
        continue
    spec = np.array(ts[LAYER]['mean_spectrum'])
    total = np.sum(spec) + 1e-10
    dc = spec[0] / total
    # high = bins beyond midpoint
    n = len(spec)
    high = np.sum(spec[n//2:]) / total
    entries.append({'name': r['name'], 'dc': dc, 'high': high, 'spec': spec})

entries.sort(key=lambda x: x['dc'])  # lowest DC first = most structured

# ---------------------------------------------------------------------------
# 2. Compute input-image FFT for selected representatives
# ---------------------------------------------------------------------------
def input_fft_radial(img_path, size=512):
    img = Image.open(img_path).convert('L').resize((size, size))
    arr = np.array(img).astype(float)
    fft = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    h, w = fft.shape
    y, x = np.indices((h, w))
    r = np.sqrt((y - h//2)**2 + (x - w//2)**2).astype(int)
    tbin = np.bincount(r.ravel(), fft.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)

# Pick representative subset (every ~5th entry to keep readable + extremes)
indices = []
# Always include extremes and a few from each category
must_include = [
    'natural_astronaut', 'contours', 'edge_shapes', 'shapes_circle_rect',
    'random_dots', 'low_freq_only', 'high_freq_only',
    'noise_std_0.5', 'solid_gray_50', 'solid_white', 'solid_black',
    'checkerboard_16', 'checkerboard_2',
    'sine_freq8_ang0', 'sine_freq2_ang0', 'sine_freq32_ang0',
    'gradient_horizontal', 'gradient_radial',
    'texture_white_noise', 'texture_brick', 'band_pass_ring',
]

selected = []
for e in entries:
    if e['name'] in must_include:
        selected.append(e)

# ---------------------------------------------------------------------------
# 3. THE MAIN FIGURE: two-panel layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(28, 14))

# --- LEFT HALF: Input thumbnails + DC/High bars, sorted ---
gs_left = fig.add_gridspec(len(selected), 3, left=0.02, right=0.48,
                            hspace=0.15, wspace=0.15)

# Category colors
def cat_color(name):
    if 'noise' in name: return '#E74C3C'
    if 'solid' in name: return '#95A5A6'
    if 'gradient' in name: return '#3498DB'
    if 'checkerboard' in name: return '#E67E22'
    if 'sine' in name: return '#2ECC71'
    if 'edge' in name or 'contour' in name: return '#9B59B6'
    if 'texture' in name: return '#1ABC9C'
    if 'natural' in name: return '#C0392B'
    if 'freq' in name or 'band_pass' in name: return '#16A085'
    if 'shape' in name or 'random_dots' in name: return '#F39C12'
    return '#7F8C8D'

def cat_label(name):
    if 'noise' in name: return '噪声'
    if 'solid' in name: return '纯色'
    if 'gradient' in name: return '渐变'
    if 'checkerboard' in name: return '棋盘格'
    if 'sine' in name: return '正弦'
    if 'edge' in name or 'contour' in name: return '边缘/轮廓'
    if 'texture' in name: return '纹理'
    if 'natural' in name: return '自然图像'
    if 'freq' in name or 'band_pass' in name: return '频率过滤'
    if 'shape' in name or 'random_dots' in name: return '形状'
    return '其他'

for i, e in enumerate(selected):
    row = i

    # Thumbnail
    ax_img = fig.add_subplot(gs_left[row, 0])
    img_path = os.path.join(INPUT_DIR, f"{e['name']}.png")
    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax_img.imshow(img)
    ax_img.set_ylabel(f"{e['name'][:22]}", fontsize=6, labelpad=2,
                      color=cat_color(e['name']), fontweight='bold',
                      ha='right', va='center', rotation=0)
    ax_img.yaxis.set_label_position("right")
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # DC/High bar
    ax_bar = fig.add_subplot(gs_left[row, 1:])
    dc_pct = e['dc'] * 100
    high_pct = e['high'] * 100
    ax_bar.barh(0, dc_pct, color='#4472C4', alpha=0.85, label='DC (均匀)')
    ax_bar.barh(0, high_pct, left=dc_pct, color='#ED7D31', alpha=0.85,
                label='高频 (细节)')
    ax_bar.set_xlim(0, 100)
    ax_bar.set_ylim(-0.6, 0.6)
    ax_bar.axis('off')

    # Category tag
    ax_bar.text(101, 0, cat_label(e['name']), fontsize=5.5, va='center',
                color=cat_color(e['name']), fontweight='bold')

    if i == 0:
        ax_bar.text(10, 0.75, 'DC (低频/均匀)', fontsize=7, color='#4472C4',
                    fontweight='bold', transform=ax_bar.transData)
        ax_bar.text(70, 0.75, '高频 (细节/结构)', fontsize=7, color='#ED7D31',
                    fontweight='bold', transform=ax_bar.transData)

# --- RIGHT HALF: Spectra overlay ---
ax_right = fig.add_axes([0.55, 0.08, 0.42, 0.88])

# Draw spectra for each selected entry, color-coded by DC ratio
for e in selected:
    spec = e['spec']
    freqs = np.arange(len(spec))
    # Color: low DC = red/warm (rich structure), high DC = blue/cool (uniform)
    dc_norm = (e['dc'] - 0.42) / (0.80 - 0.42)  # normalize to observed range
    dc_norm = np.clip(dc_norm, 0, 1)
    color = plt.cm.RdYlBu_r(dc_norm)
    ax_right.loglog(freqs[1:], spec[1:], color=color, linewidth=1.2,
                    alpha=0.8, label=f"{e['name'][:25]} (DC={e['dc']*100:.0f}%)")

# DC ratio colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(0.42, 0.80))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax_right, shrink=0.6, pad=0.02)
cbar.set_label('DC Ratio (低=暖色/结构丰富, 高=冷色/均匀平滑)', fontsize=8)

ax_right.set_xlabel('Spatial Frequency (radial bin)', fontsize=10)
ax_right.set_ylabel('Power (log scale)', fontsize=10)
ax_right.set_title('Bottleneck Radial Power Spectra\nColor = DC Ratio: '
                   'warm → structured input, cool → uniform input',
                   fontsize=10, fontweight='bold')
ax_right.grid(True, alpha=0.2)

# Key annotations on the spectra plot
ax_right.annotate('自然图像\n边缘/轮廓\n复杂形状',
                   xy=(1.2, 1e-1), fontsize=8, color='#C0392B', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEAA7', alpha=0.8))
ax_right.annotate('噪声\n纹理',
                   xy=(2.5, 2e-2), fontsize=8, color='#E67E22', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', alpha=0.8))
ax_right.annotate('纯色\n周期图案\n渐变',
                   xy=(3.5, 5e-3), fontsize=8, color='#2C3E50', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5DBDB', alpha=0.8))

plt.suptitle('瓶颈频域特征与输入图像的关系\n'
             '左侧: 输入缩略图 + DC/高频能量比 (按DC从低到高排列)  |  '
             '右侧: 对应瓶颈径向功率谱 (暖色=结构丰富, 冷色=均匀)',
             fontsize=12, fontweight='bold', y=0.99)

plt.savefig(os.path.join(OUT_DIR, "fig5_input_to_frequency_mapping.png"),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("[*] Fig5: Input-to-frequency mapping saved.")

# ---------------------------------------------------------------------------
# 4. SUPPLEMENTARY: the "predictability" scatter plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 9))

all_entries = []
for e in entries:
    img_path = os.path.join(INPUT_DIR, f"{e['name']}.png")
    in_fft = input_fft_radial(img_path)
    # Compute input entropy as measure of "unpredictability"
    in_spec_norm = in_fft[1:32] / (np.sum(in_fft[1:32]) + 1e-10)
    in_entropy = -np.sum(in_spec_norm * np.log(in_spec_norm + 1e-10))
    max_ent = np.log(len(in_spec_norm))
    in_ent_norm = in_entropy / max_ent if max_ent > 0 else 0
    all_entries.append({**e, 'in_ent': in_ent_norm})

# Scatter: input entropy (predictability) vs bottleneck DC
for e in all_entries:
    color = cat_color(e['name'])
    ax.scatter(e['in_ent'], e['dc'] * 100, c=color, s=80, alpha=0.8,
               edgecolors='white', linewidth=0.5, zorder=3)
    # Label key points
    if e['name'] in must_include:
        offset = 0.3 if e['name'] not in ['sine_freq8_ang0', 'sine_freq32_ang0'] else 0.6
        ax.annotate(e['name'][:18], (e['in_ent'], e['dc'] * 100),
                    fontsize=5.5, alpha=0.9,
                    xytext=(5, 5), textcoords='offset points')

# Draw quadrant lines
ax.axhline(y=65, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0.75, color='gray', linestyle='--', alpha=0.3)

# Quadrant labels
ax.text(0.98, 85, '可预测 + 高DC\n(周期信号, 纯色, 渐变)\n→ 瓶颈强力平滑', fontsize=9,
        ha='right', va='top', color='#7F8C8D',
        bbox=dict(facecolor='#F2F3F4', alpha=0.7))
ax.text(0.55, 85, '可预测 + 低DC\n(罕见组合)', fontsize=9,
        ha='center', va='top', color='#BDC3C7')
ax.text(0.98, 45, '不可预测 + 高DC\n(白噪声被滤除)', fontsize=9,
        ha='right', va='bottom', color='#E67E22',
        bbox=dict(facecolor='#FDEBD0', alpha=0.7))
ax.text(0.55, 45, '不可预测 + 低DC\n(自然图像, 边缘, 复杂形状)\n→ 瓶颈保留高频', fontsize=9,
        ha='center', va='bottom', color='#C0392B',
        bbox=dict(facecolor='#FADBD8', alpha=0.7))

# Legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', label='噪声'), Patch(facecolor='#95A5A6', label='纯色'),
    Patch(facecolor='#3498DB', label='渐变'), Patch(facecolor='#E67E22', label='棋盘格'),
    Patch(facecolor='#2ECC71', label='正弦'), Patch(facecolor='#9B59B6', label='边缘/轮廓'),
    Patch(facecolor='#1ABC9C', label='纹理'), Patch(facecolor='#C0392B', label='自然图像'),
    Patch(facecolor='#F39C12', label='形状/散点'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='lower left', ncol=3)

ax.set_xlabel('Input Image Spectral Entropy (不可预测性)', fontsize=11)
ax.set_ylabel('Bottleneck DC Ratio % (均匀度)', fontsize=11)
ax.set_title('核心规律: 瓶颈DC比率取决于输入的"不可预测性"而非"变化快慢"\n'
             'x轴 = 输入频谱平坦度 (高=不可预测, 低=可预测)  |  '
             'y轴 = 瓶颈均匀度 (高=被平滑, 低=保留结构)',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.15)
ax.set_xlim(0.45, 1.05)
ax.set_ylim(38, 92)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig6_predictability_vs_dc.png"),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("[*] Fig6: Predictability vs DC scatter saved.")

print("Done. Two figures generated.")
