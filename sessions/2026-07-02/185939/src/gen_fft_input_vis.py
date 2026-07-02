#!/usr/bin/env python3
"""
Show side-by-side: original image → bottleneck feature map → 2D FFT magnitude.
Clarify what exactly is being FFT'd.
"""

import os, json, torch, numpy as np
torch.backends.cudnn.enabled = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties, fontManager
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# Font
_fp = '/home/tianyi/.local/share/fonts/wqy-microhei.ttc'
if os.path.exists(_fp):
    fontManager.addfont(_fp)
    plt.rcParams['font.family'] = FontProperties(fname=_fp).get_name()
plt.rcParams['axes.unicode_minus'] = False

MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda:0"
DTYPE = torch.float16
TIMESTEP = 500
OUT_DIR = "/mnt/data/tianyi/ClaudeMisc/sessions/2026-07-02/185939/outputs/report"
INPUT_DIR = "/mnt/data/tianyi/ClaudeMisc/sessions/2026-07-02/185939/outputs/inputs"

# ---------------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------------
print("[*] Loading models...")
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(DEVICE, dtype=DTYPE).eval()
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE, dtype=DTYPE).eval()
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE, dtype=DTYPE).eval()

# Hook
feat_cache = {}
def hook_fn(name):
    def h(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        feat_cache[name] = out.detach().cpu().float().numpy()
    return h
unet.mid_block.register_forward_hook(hook_fn("mid_block"))
unet.down_blocks[0].register_forward_hook(hook_fn("down_0"))

# Null text embedding
uncond = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
text_emb = text_encoder(uncond.input_ids.to(DEVICE))[0]

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

# ---------------------------------------------------------------------------
# Select representative inputs — one per category
# ---------------------------------------------------------------------------
key_inputs = [
    ('natural_astronaut',   '自然图像'),
    ('contours',            '边缘/轮廓'),
    ('shapes_circle_rect',  '几何形状'),
    ('random_dots',         '随机散点'),
    ('noise_std_0.5',       '高斯噪声'),
    ('texture_white_noise', '白噪声纹理'),
    ('texture_brick',       '砖纹纹理'),
    ('gradient_horizontal', '水平渐变'),
    ('checkerboard_16',     '棋盘格 16'),
    ('sine_freq8_ang0',     '正弦光栅 8Hz'),
    ('solid_gray_50',       '纯灰'),
    ('solid_white',         '纯白'),
]

# ---------------------------------------------------------------------------
# Run forward and collect features
# ---------------------------------------------------------------------------
print("[*] Running forward passes...")
data = []
for name, label in key_inputs:
    img_path = os.path.join(INPUT_DIR, f"{name}.png")
    if not os.path.exists(img_path):
        continue
    img_pil = Image.open(img_path).convert('RGB')

    feat_cache.clear()
    img_t = trans(img_pil).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        latent = vae.encode(img_t).latent_dist.sample() * vae.config.scaling_factor
        unet(latent, torch.tensor([TIMESTEP], device=DEVICE),
             encoder_hidden_states=text_emb, return_dict=False)

    mid_feat = feat_cache.get("mid_block")[0]   # (1280, 8, 8)
    down_feat = feat_cache.get("down_0")[0]      # (320, 32, 32)

    # 2D FFT on mean feature map
    mean_feat = mid_feat.mean(axis=0)  # (8, 8)
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(mean_feat)))

    # First 3 channels for visualization
    ch3 = mid_feat[:3]  # (3, 8, 8)
    ch3 = (ch3 - ch3.min()) / (ch3.max() - ch3.min() + 1e-8)
    ch3_rgb = ch3.transpose(1, 2, 0)

    # Down_0 for comparison (first 3 ch)
    d3 = down_feat[:3]  # (3, 32, 32)
    d3 = (d3 - d3.min()) / (d3.max() - d3.min() + 1e-8)
    d3_rgb = d3.transpose(1, 2, 0)

    data.append({
        'name': name, 'label': label,
        'img': np.array(img_pil),
        'mid_feat_rgb': ch3_rgb,
        'mid_feat_mean': mean_feat,
        'fft_mag': fft_mag,
        'down_feat_rgb': d3_rgb,
        'shape': mid_feat.shape,
    })

# ---------------------------------------------------------------------------
# FIGURE: 4 columns × N rows
# Col 1: Original 512x512    Col 2: down_0 feat (32x32)
# Col 3: mid feat (8x8)      Col 4: 2D FFT magnitude
# ---------------------------------------------------------------------------
N = len(data)
fig, axes = plt.subplots(N, 4, figsize=(16, N * 2.8 + 0.5))

col_titles = ['原始图像\n512×512',
              'Down Block 0 特征图\n(前3ch, 32×32)',
              'Bottleneck 特征图 ★FFT输入★\n(前3ch, 8×8)',
              '2D FFT 幅度谱\n(log scale)']

for i, d in enumerate(data):
    axes[i, 0].imshow(d['img'])
    axes[i, 0].set_ylabel(f"{d['label']}\n{d['name'][:18]}", fontsize=7, fontweight='bold')
    axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

    axes[i, 1].imshow(d['down_feat_rgb'], aspect='auto')
    axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])

    axes[i, 2].imshow(d['mid_feat_rgb'], aspect='auto')
    axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])
    # Annotate shape
    axes[i, 2].text(0.98, 0.02, f'{d["shape"][0]}ch×{d["shape"][1]}×{d["shape"][2]}',
                    transform=axes[i, 2].transAxes, fontsize=5.5, ha='right', va='bottom',
                    color='white', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

    im = axes[i, 3].imshow(d['fft_mag'], cmap='inferno', norm=LogNorm(), aspect='auto')
    axes[i, 3].set_xticks([]); axes[i, 3].set_yticks([])

for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=9, fontweight='bold', pad=8)

# Colorbar for FFT
cbar_ax = fig.add_axes([0.92, 0.06, 0.008, 0.88])
plt.colorbar(im, cax=cbar_ax, label='FFT Magnitude (log)')

fig.suptitle('FFT 的输入是什么？—— 原始图像 → CNN特征图 → 2D FFT 的完整链路\n'
             '第3列 = 瓶颈处 1280 通道特征图的空间可视化（前3通道映射为RGB），第4列 = 对特征图逐通道做2D FFT后跨通道取均值',
             fontsize=11, fontweight='bold', y=1.01)

plt.tight_layout(rect=[0, 0, 0.91, 0.99])
plt.savefig(os.path.join(OUT_DIR, "fig7_fft_input_chain.png"),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("[*] Fig7: FFT input chain saved.")

# ---------------------------------------------------------------------------
# Bonus: single detailed example (astronaut) with all 4 feature map channels
# ---------------------------------------------------------------------------
astro = data[0]  # astronaut
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 9))

axes2[0, 0].imshow(astro['img'])
axes2[0, 0].set_title('原始图像 (512×512)', fontsize=10, fontweight='bold')
axes2[0, 0].axis('off')

# Mean feature map
axes2[0, 1].imshow(astro['mid_feat_mean'], cmap='viridis', aspect='auto')
axes2[0, 1].set_title('Bottleneck 特征图\n(1280ch均值, 8×8)', fontsize=10, fontweight='bold')
axes2[0, 1].axis('off')
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes2[0, 1], fraction=0.046)

# FFT
axes2[0, 2].imshow(astro['fft_mag'], cmap='inferno', norm=LogNorm(), aspect='auto')
axes2[0, 2].set_title('2D FFT 幅度谱\n(跨通道均值, log scale)', fontsize=10, fontweight='bold')
axes2[0, 2].axis('off')

# Per-channel radial spectra (first 20 ch)
axes2[1, 0].set_title('逐通道径向功率谱 (前20通道)', fontsize=9, fontweight='bold')
json_data = os.path.join(os.path.dirname(INPUT_DIR), "data/spectra_results.json")
with open(json_data) as f:
    results = json.load(f)
r = next((x for x in results if x['name'] == 'natural_astronaut'), None)
if r:
    ts = r['timesteps']['500']['mid_block']
    # We need per-channel data — recompute quickly
    mid_full = feat_cache.get("mid_block")[0]  # already in feat_cache from astronaut run
    # Wait, feat_cache was cleared — let's re-run astronaut
    img_pil = Image.open(os.path.join(INPUT_DIR, "natural_astronaut.png")).convert('RGB')
    feat_cache.clear()
    img_t = trans(img_pil).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        latent = vae.encode(img_t).latent_dist.sample() * vae.config.scaling_factor
        unet(latent, torch.tensor([TIMESTEP], device=DEVICE),
             encoder_hidden_states=text_emb, return_dict=False)
    mid_full = feat_cache["mid_block"][0]  # (1280, 8, 8)

    from scipy.ndimage import map_coordinates
    for c in range(min(20, mid_full.shape[0])):
        ch = mid_full[c]
        fft_ch = np.abs(np.fft.fftshift(np.fft.fft2(ch)))
        h, w = fft_ch.shape
        y, x = np.indices((h, w))
        r_coord = np.sqrt((y-h//2)**2 + (x-w//2)**2).astype(int)
        tbin = np.bincount(r_coord.ravel(), fft_ch.ravel())
        nr = np.bincount(r_coord.ravel())
        radial = tbin / np.maximum(nr, 1)
        axes2[1, 0].semilogy(radial[:h//2], alpha=0.3, linewidth=0.5, color=f'C{c%10}')
    axes2[1, 0].set_xlabel('径向频率 bin')
    axes2[1, 0].set_ylabel('幅度 (log)')
    axes2[1, 0].grid(True, alpha=0.3)

# Mean spectrum (from saved data)
if r:
    spec = np.array(ts['mean_spectrum'])
    axes2[1, 1].loglog(np.arange(len(spec)), spec, 'o-', color='#C0392B', linewidth=2, markersize=6)
    axes2[1, 1].set_xlabel('径向频率 bin')
    axes2[1, 1].set_ylabel('幅度 (log-log)')
    axes2[1, 1].set_title('跨通道平均径向功率谱', fontsize=9, fontweight='bold')
    axes2[1, 1].grid(True, alpha=0.3)
    # Annotate DC
    axes2[1, 1].annotate(f'DC = {spec[0]/np.sum(spec)*100:.0f}%',
                         xy=(0, spec[0]), fontsize=9, fontweight='bold', color='#C0392B',
                         xytext=(0.5, spec[0]*1.5), arrowprops=dict(arrowstyle='->', color='gray'))

# Annotation box
axes2[1, 2].axis('off')
text = (
    "FFT 流水线说明:\n\n"
    "1. 输入图像 512×512×3\n"
    "    ↓ VAE encode\n"
    "2. Latent 64×64×4\n"
    "    ↓ UNet down blocks\n"
    "3. Bottleneck feature map\n"
    "    1280 通道 × 8×8\n"
    "    ↓ 逐通道 2D FFT\n"
    "4. 幅度谱 (每通道 8×8)\n"
    "    ↓ 径向平均 + 跨通道均值\n"
    "5. 1D 径向功率谱 P(k)\n\n"
    "★ '高频'指的是特征图上的\n"
    "   高频空间变化，不是原图像素变化"
)
axes2[1, 2].text(0.05, 0.95, text, transform=axes2[1, 2].transAxes,
                 fontsize=8.5, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#F8F9F9', alpha=0.9))

fig2.suptitle('案例详解: natural_astronaut 的 FFT 分析链路', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig8_astronaut_fft_detail.png"),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("[*] Fig8: Astronaut FFT detail saved.")

print("Done.")
