#!/usr/bin/env python3
"""
UNet Bottleneck FFT Analysis
=============================
Analyze frequency-domain characteristics of UNet bottleneck latent vectors
under diverse input types: Gaussian noise, solid colors, gradients, edges,
textures, natural images, etc.

Focus: relationship between input spatial frequency content and
bottleneck FFT spectra.
"""

import os
import sys
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cuDNN due to compatibility issues on this system
torch.backends.cudnn.enabled = False
from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from scipy.ndimage import sobel, gaussian_filter
from scipy import signal
from skimage import data as skimage_data
from skimage.color import rgb2gray

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda:0"
DTYPE = torch.float16
IMAGE_SIZE = 512
LATENT_SIZE = IMAGE_SIZE // 8  # 64
TIMESTEP = 500  # mid-range timestep for meaningful bottleneck features
OUTPUT_DIR = "/mnt/data/tianyi/ClaudeMisc/sessions/2026-07-02/185939_v2_downblock0_32x32/outputs"
PRIMARY_LAYER = "down_block_0"  # 32×32 → 16 freq bins (comparable to original UNet ~28×28)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "inputs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "spectra"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)


def radial_profile(image_fft, center=None):
    """Compute radially averaged 1D power spectrum from 2D FFT magnitude."""
    h, w = image_fft.shape
    if center is None:
        center = (h // 2, w // 2)
    y, x = np.indices((h, w))
    r = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), image_fft.ravel())
    nr = np.bincount(r.ravel())
    radial = tbin / np.maximum(nr, 1)
    max_r = min(h, w) // 2
    return radial[:max_r]


def compute_fft_spectrum(tensor_2d):
    """Compute 2D FFT magnitude spectrum (shifted, log scale)."""
    fft = np.fft.fft2(tensor_2d)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    return magnitude


def compute_channel_spectra(feature_map):
    """
    feature_map: numpy array (C, H, W) or (H, W)
    Returns: per-channel radial spectra, mean spectrum, and 2D magnitude maps
    """
    if feature_map.ndim == 2:
        feature_map = feature_map[np.newaxis, :, :]

    C, H, W = feature_map.shape
    channel_spectra = []
    magnitude_maps = []

    for c in range(C):
        mag = compute_fft_spectrum(feature_map[c])
        magnitude_maps.append(mag)
        radial = radial_profile(mag)
        channel_spectra.append(radial)

    channel_spectra = np.array(channel_spectra)  # (C, max_r)
    mean_spectrum = channel_spectra.mean(axis=0)
    std_spectrum = channel_spectra.std(axis=0)
    mean_magnitude = np.mean(magnitude_maps, axis=0)

    return {
        'channel_spectra': channel_spectra,
        'mean_spectrum': mean_spectrum,
        'std_spectrum': std_spectrum,
        'mean_magnitude': mean_magnitude,
        'magnitude_maps': magnitude_maps,
    }


# ============================================================================
# Input Generators
# ============================================================================

def generate_inputs():
    """Generate diverse input images for analysis."""
    inputs = {}
    S = IMAGE_SIZE

    # --- 1. Gaussian Noise (various std) ---
    for std in [0.1, 0.3, 0.5, 0.8, 1.0]:
        noise = np.random.randn(S, S, 3).astype(np.float32) * std + 0.5
        noise = np.clip(noise, 0, 1)
        inputs[f"noise_std_{std}"] = (noise * 255).astype(np.uint8)

    # --- 2. Solid Colors ---
    colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray_25": (64, 64, 64),
        "gray_50": (128, 128, 128),
        "gray_75": (192, 192, 192),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
    }
    for name, color in colors.items():
        img = np.ones((S, S, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8).reshape(1, 1, 3)
        inputs[f"solid_{name}"] = img

    # --- 3. Gradients ---
    # Horizontal gradient (black to white left-right)
    grad = np.linspace(0, 1, S).reshape(1, S).repeat(S, axis=0)
    inputs["gradient_horizontal"] = (np.stack([grad, grad, grad], axis=-1) * 255).astype(np.uint8)

    # Vertical gradient
    grad = np.linspace(0, 1, S).reshape(S, 1).repeat(S, axis=1)
    inputs["gradient_vertical"] = (np.stack([grad, grad, grad], axis=-1) * 255).astype(np.uint8)

    # Radial gradient
    y, x = np.ogrid[-S//2:S//2, -S//2:S//2]
    r = np.sqrt(x**2 + y**2) / (S//2)
    grad_r = np.clip(1 - r, 0, 1)
    inputs["gradient_radial"] = (np.stack([grad_r, grad_r, grad_r], axis=-1) * 255).astype(np.uint8)

    # Diagonal gradient
    grad_d = np.linspace(0, 1, S)
    grad_diag = (grad_d.reshape(1, S) + grad_d.reshape(S, 1)) / 2
    inputs["gradient_diagonal"] = (np.stack([grad_diag, grad_diag, grad_diag], axis=-1) * 255).astype(np.uint8)

    # --- 4. Checkerboard / Grid Patterns ---
    for freq in [2, 4, 8, 16, 32]:
        pattern = np.indices((S, S)).sum(axis=0) % (S // freq) < (S // (2 * freq))
        pattern = pattern.astype(np.float32)
        inputs[f"checkerboard_{freq}"] = (np.stack([pattern, pattern, pattern], axis=-1) * 255).astype(np.uint8)

    # --- 5. Sine Gratings ---
    for freq, angle in [(2, 0), (8, 0), (32, 0), (8, 45), (8, 90), (16, 30)]:
        y, x = np.ogrid[:S, :S]
        theta = np.deg2rad(angle)
        grating = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (x * np.cos(theta) + y * np.sin(theta)) / S)
        inputs[f"sine_freq{freq}_ang{angle}"] = (np.stack([grating, grating, grating], axis=-1) * 255).astype(np.uint8)

    # --- 6. Edge / Contour Maps ---
    # Canny-like edge on a circle
    edge_img = Image.new('L', (S, S), 128)
    draw = ImageDraw.Draw(edge_img)
    draw.ellipse([S//4, S//4, 3*S//4, 3*S//4], fill=200, outline=50, width=3)
    draw.rectangle([S//8, S//8, 3*S//8, 3*S//8], fill=180, outline=10, width=2)
    edge_arr = np.array(edge_img)
    edge_sobel = np.sqrt(sobel(edge_arr.astype(float), axis=0)**2 + sobel(edge_arr.astype(float), axis=1)**2)
    edge_sobel = np.clip(edge_sobel / edge_sobel.max(), 0, 1)
    inputs["edge_shapes"] = (np.stack([edge_sobel, edge_sobel, edge_sobel], axis=-1) * 255).astype(np.uint8)

    # Clean edges (contour drawing)
    contour_img = Image.new('RGB', (S, S), (255, 255, 255))
    draw_c = ImageDraw.Draw(contour_img)
    for i in range(10):
        offset = i * S // 20
        draw_c.ellipse([offset, offset, S-offset, S-offset], outline=(0, 0, 0), width=2)
    draw_c.line([(0, S//2), (S, S//2)], fill=(0, 0, 0), width=2)
    draw_c.line([(S//2, 0), (S//2, S)], fill=(0, 0, 0), width=2)
    inputs["contours"] = np.array(contour_img)

    # --- 7. Simple Shapes on Background ---
    # Circle
    shapes_img = Image.new('RGB', (S, S), (128, 128, 128))
    draw_s = ImageDraw.Draw(shapes_img)
    draw_s.ellipse([S//4, S//4, 3*S//4, 3*S//4], fill=(200, 100, 50))
    draw_s.rectangle([3*S//8, 3*S//8, 5*S//8, 5*S//8], fill=(50, 100, 200))
    inputs["shapes_circle_rect"] = np.array(shapes_img)

    # Multiple small circles (texture-like)
    dots_img = Image.new('RGB', (S, S), (255, 255, 255))
    draw_d = ImageDraw.Draw(dots_img)
    rng = np.random.RandomState(42)
    for _ in range(200):
        x, y = rng.randint(0, S), rng.randint(0, S)
        r = rng.randint(2, 10)
        color = tuple(rng.randint(0, 255, 3).tolist())
        draw_d.ellipse([x-r, y-r, x+r, y+r], fill=color)
    inputs["random_dots"] = np.array(dots_img)

    # --- 8. Textures ---
    # White noise texture (high frequency)
    tex_noise = (np.random.randn(S, S, 3) * 0.2 + 0.5).clip(0, 1)
    inputs["texture_white_noise"] = (tex_noise * 255).astype(np.uint8)

    # Low-pass filtered noise (blur)
    tex_blur = gaussian_filter(tex_noise.astype(float), sigma=(5, 5, 0))
    tex_blur = np.clip(tex_blur, 0, 1)
    inputs["texture_blur_noise"] = (tex_blur * 255).astype(np.uint8)

    # Wood-like texture (banded noise)
    y, x = np.ogrid[:S, :S]
    wood = 0.5 + 0.3 * np.sin(0.05 * y + 5 * np.sin(0.02 * x))
    wood += 0.05 * np.random.randn(S, S)
    wood = np.clip(wood, 0, 1)
    inputs["texture_wood_like"] = (np.stack([wood, wood*0.7, wood*0.4], axis=-1) * 255).astype(np.uint8)

    # Brick-like pattern
    brick = np.zeros((S, S))
    brick_h = S // 16
    brick_w = S // 8
    for row in range(0, S, brick_h):
        offset = (row // brick_h) % 2 * (brick_w // 2)
        for col in range(-brick_w, S + brick_w, brick_w):
            c_start = col + offset
            if 0 <= c_start < S:
                brick[row:row+brick_h-2, c_start:c_start+brick_w-2] = 0.7
    brick += 0.05 * np.random.randn(S, S)
    brick = np.clip(brick, 0, 1)
    inputs["texture_brick"] = (np.stack([brick*0.8, brick*0.45, brick*0.3], axis=-1) * 255).astype(np.uint8)

    # --- 9. Natural Images (from skimage) ---
    try:
        astronaut = skimage_data.astronaut()
        astronaut = np.array(Image.fromarray(astronaut).resize((S, S)))
        inputs["natural_astronaut"] = astronaut
    except Exception:
        pass

    try:
        coffee = skimage_data.coffee()
        coffee = np.array(Image.fromarray(coffee).resize((S, S)))
        inputs["natural_coffee"] = coffee
    except Exception:
        pass

    # --- 10. Frequency-Specific Patterns ---
    # Low frequency only (reconstructed from low freq components)
    lf_base = gaussian_filter(np.random.randn(S, S), sigma=20)
    lf_base = (lf_base - lf_base.min()) / (lf_base.max() - lf_base.min())
    inputs["low_freq_only"] = (np.stack([lf_base, lf_base, lf_base], axis=-1) * 255).astype(np.uint8)

    # High frequency only (high-pass filtered noise)
    hf_noise = np.random.randn(S, S)
    hf_low = gaussian_filter(hf_noise, sigma=3)
    hf_high = hf_noise - hf_low
    hf_high = (hf_high - hf_high.min()) / (hf_high.max() - hf_high.min())
    inputs["high_freq_only"] = (np.stack([hf_high, hf_high, hf_high], axis=-1) * 255).astype(np.uint8)

    # Band-pass pattern (ring in frequency domain)
    bp = np.zeros((S, S))
    cy, cx = S//2, S//2
    for i in range(S):
        for j in range(S):
            d = np.sqrt((i-cy)**2 + (j-cx)**2)
            bp[i, j] = np.exp(-((d - S//8)**2) / (2 * (S//32)**2))
    bp = (bp - bp.min()) / (bp.max() - bp.min())
    inputs["band_pass_ring"] = (np.stack([bp, bp, bp], axis=-1) * 255).astype(np.uint8)

    # --- 11. Real-world-like Compositions ---
    # Sky gradient with sun
    sky = np.zeros((S, S, 3))
    y_grad = np.linspace(0.4, 0.9, S).reshape(S, 1)
    sky[:, :, 2] = y_grad  # blue
    sky[:, :, 0] = y_grad * 0.5
    sky[:, :, 1] = y_grad * 0.6
    # Sun
    sy, sx = np.ogrid[-S//2:S//2, -S//2:S//2]
    sun_r = np.sqrt((sx - S//4)**2 + (sy + S//4)**2)
    sun = np.exp(-sun_r**2 / (2 * (S//12)**2))
    sun = np.clip(sun * 1.5, 0, 1)
    for ch in range(3):
        sky[:, :, ch] = np.clip(sky[:, :, ch] + sun * 0.8, 0, 1)
    inputs["composition_sky_sun"] = (sky * 255).astype(np.uint8)

    return inputs


# ============================================================================
# Model Loading and Feature Extraction
# ============================================================================

class UNetBottleneckExtractor:
    """Extract bottleneck (mid-block) features from SD UNet."""

    def __init__(self, model_id=MODEL_ID, device=DEVICE, dtype=DTYPE):
        self.device = device
        self.dtype = dtype

        print(f"[*] Loading VAE from {model_id}...")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=dtype)
        self.vae.eval()

        print(f"[*] Loading UNet from {model_id}...")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device, dtype=dtype)
        self.unet.eval()

        print(f"[*] Loading CLIP text encoder...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device, dtype=dtype)
        self.text_encoder.eval()

        # Hook storage
        self.bottleneck_features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on bottleneck layers and down blocks."""
        def make_hook(name):
            def hook(module, input, output):
                # Handle tuple outputs (some blocks return (hidden_states, ...))
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                self.bottleneck_features[name] = out.detach().cpu().float().numpy()
            return hook

        # Hook into mid_block output (the true bottleneck)
        self.unet.mid_block.register_forward_hook(make_hook("mid_block"))

        # Hook individual mid-block components
        if hasattr(self.unet.mid_block, 'attentions'):
            for i, attn in enumerate(self.unet.mid_block.attentions):
                attn.register_forward_hook(make_hook(f"mid_attn_{i}"))
        if hasattr(self.unet.mid_block, 'resnets'):
            for i, resnet in enumerate(self.unet.mid_block.resnets):
                resnet.register_forward_hook(make_hook(f"mid_resnet_{i}"))

        # Hook down blocks for multi-resolution analysis
        for i, down_block in enumerate(self.unet.down_blocks):
            down_block.register_forward_hook(make_hook(f"down_block_{i}"))

        # Hook up blocks too
        for i, up_block in enumerate(self.unet.up_blocks):
            up_block.register_forward_hook(make_hook(f"up_block_{i}"))

    def encode_image(self, image_pil):
        """Encode PIL image to latent space."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        img_tensor = transform(image_pil).unsqueeze(0).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            latent = self.vae.encode(img_tensor).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def get_null_text_embedding(self, batch_size=1):
        """Get null text embedding for unconditional forward pass."""
        max_length = self.tokenizer.model_max_length
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            uncond_emb = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
        return uncond_emb

    def extract_bottleneck(self, image_pil, timestep=TIMESTEP):
        """
        Encode image, run UNet forward, extract bottleneck features.
        Returns dict of hook outputs.
        """
        self.bottleneck_features = {}

        latent = self.encode_image(image_pil)
        encoder_hidden_states = self.get_null_text_embedding()

        # Create timestep tensor
        timestep_tensor = torch.tensor([timestep], device=self.device, dtype=torch.long)

        with torch.no_grad():
            _ = self.unet(
                latent,
                timestep_tensor,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )

        return {k: v.copy() for k, v in self.bottleneck_features.items()}

    def analyze_input(self, name, image_array, timesteps=[100, 300, 500, 700, 900]):
        """
        Full analysis pipeline for one input image.
        Returns: dict with spectra at each timestep for each hooked layer.
        """
        image_pil = Image.fromarray(image_array.astype(np.uint8))
        results = {'name': name, 'timesteps': {}}

        for t in timesteps:
            features = self.extract_bottleneck(image_pil, timestep=t)
            ts_results = {}
            for layer_name, feat_map in features.items():
                # feat_map: (B, C, H, W)
                feat_2d = feat_map[0]  # remove batch dim -> (C, H, W)
                spectra = compute_channel_spectra(feat_2d)
                ts_results[layer_name] = {
                    'mean_spectrum': spectra['mean_spectrum'].tolist(),
                    'std_spectrum': spectra['std_spectrum'].tolist(),
                    'spatial_shape': list(feat_2d.shape),
                }
            results['timesteps'][str(t)] = ts_results

        return results


# ============================================================================
# Visualization
# ============================================================================

def plot_input_gallery(inputs, output_dir):
    """Create a gallery of all input images."""
    n = len(inputs)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, (name, img) in enumerate(sorted(inputs.items())):
        axes[i].imshow(img)
        axes[i].set_title(name[:20], fontsize=6)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_gallery.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Input gallery saved.")


def plot_spectra_comparison(results, output_dir):
    """
    Compare radial power spectra across different input types.
    Focus on the mid_block output at a representative timestep.
    """
    timestep = '500'
    layer = PRIMARY_LAYER  # 32×32, 16 freq bins

    # Check if we have valid results
    valid_results = [r for r in results
                     if r['timesteps'].get(timestep, {}).get(layer) is not None]
    if not valid_results:
        print("[!] No valid results for spectra comparison, skipping.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- Panel 1: All spectra overlaid (log-log) ---
    ax = axes[0, 0]
    for r in valid_results:
        spec = np.array(r['timesteps'][timestep][layer]['mean_spectrum'])
        freqs = np.arange(len(spec))
        ax.loglog(freqs[1:], spec[1:], alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Spatial Frequency (radial bins)')
    ax.set_ylabel('Power')
    ax.set_title(f'All Spectra Overlay ({len(valid_results)} inputs, log-log)')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: By category (colored groups) ---
    ax = axes[0, 1]
    prefixes = ['noise', 'solid', 'gradient', 'checkerboard', 'sine',
                'edge', 'contour', 'texture', 'natural', 'shape',
                'low_freq', 'high_freq', 'band_pass', 'random_dots', 'composition']
    color_map = {}
    for i, p in enumerate(prefixes):
        color_map[p] = plt.cm.tab20(i % 20)
    for r in valid_results:
        name = r['name']
        spec = np.array(r['timesteps'][timestep][layer]['mean_spectrum'])
        freqs = np.arange(len(spec))
        match_pref = 'other'
        for p in prefixes:
            if name.startswith(p):
                match_pref = p
                break
        ax.loglog(freqs[1:], spec[1:], alpha=0.5, linewidth=0.7,
                  color=color_map.get(match_pref, 'gray'), label=name[:25])
    ax.set_xlabel('Spatial Frequency (radial bins)')
    ax.set_ylabel('Power')
    ax.set_title('Spectra by Input Type')
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 15:
        step = len(handles) // 15
        ax.legend(handles[::step], labels[::step], fontsize=5, loc='lower left')
    elif handles:
        ax.legend(fontsize=5, loc='lower left')

    # --- Panel 3: Power ratio (high/low freq) bar chart ---
    ax = axes[0, 2]
    ratios = []
    names_list = []
    for r in valid_results:
        spec = np.array(r['timesteps'][timestep][layer]['mean_spectrum'])
        low_freq_power = np.sum(spec[1:5]) if len(spec) > 5 else 0
        high_freq_power = np.sum(spec[len(spec)//4:]) if len(spec) > 8 else 0
        ratio = high_freq_power / (low_freq_power + 1e-10)
        ratios.append(np.log10(ratio + 1))
        names_list.append(r['name'][:20])
    if ratios:
        sorted_idx = np.argsort(ratios)
        ratios_sorted = [ratios[i] for i in sorted_idx]
        names_sorted = [names_list[i] for i in sorted_idx]
        bars = ax.barh(range(len(names_sorted)), ratios_sorted, height=0.7)
        for i, (r_val, n) in enumerate(zip(ratios_sorted, names_sorted)):
            if 'noise' in n: bars[i].set_color('red')
            elif 'solid' in n: bars[i].set_color('gray')
            elif 'gradient' in n: bars[i].set_color('blue')
            elif 'checkerboard' in n or 'sine' in n: bars[i].set_color('orange')
            elif 'texture' in n: bars[i].set_color('green')
            elif 'natural' in n: bars[i].set_color('purple')
        ax.set_yticks(range(len(names_sorted)))
        ax.set_yticklabels(names_sorted, fontsize=4.5)
        ax.set_xlabel('log10(High/Low Power Ratio + 1)')
        ax.set_title('Frequency Balance (High/Low Ratio)')

    # --- Panel 4: Selected representative spectra ---
    ax = axes[1, 0]
    selected = ['noise_std_0.5', 'solid_gray_50', 'checkerboard_16',
                'sine_freq8_ang0', 'texture_white_noise', 'gradient_horizontal']
    for sel_name in selected:
        found = next((r for r in valid_results if r['name'] == sel_name), None)
        if found is None:
            continue
        spec = np.array(found['timesteps'][timestep][layer]['mean_spectrum'])
        freqs = np.arange(len(spec))
        ax.loglog(freqs[1:], spec[1:], linewidth=1.2, label=sel_name[:25])
    ax.set_xlabel('Spatial Frequency (radial bins)')
    ax.set_ylabel('Power')
    ax.set_title('Selected Representative Spectra')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Spectral slope analysis ---
    ax = axes[1, 1]
    slopes = []
    for r in valid_results:
        spec = np.array(r['timesteps'][timestep][layer]['mean_spectrum'])
        freqs = np.arange(1, len(spec))
        log_freqs = np.log10(freqs[:len(spec)//2])
        log_spec = np.log10(spec[1:len(spec)//2 + 1] + 1e-10)
        if len(log_freqs) > 2:
            slope = np.polyfit(log_freqs, log_spec, 1)[0]
            slopes.append((r['name'], slope))

    if slopes:
        names_s, vals_s = zip(*sorted(slopes, key=lambda x: x[1]))
        colors_s = []
        for n in names_s:
            if 'noise' in n: colors_s.append('red')
            elif 'solid' in n: colors_s.append('gray')
            elif 'gradient' in n: colors_s.append('blue')
            else: colors_s.append('steelblue')
        ax.barh(range(len(names_s)), vals_s, color=colors_s, height=0.7)
        ax.set_yticks(range(len(names_s)))
        ax.set_yticklabels([n[:20] for n in names_s], fontsize=4.5)
        ax.set_xlabel('Spectral Slope (log-log)')
        ax.set_title('Spectral Decay Slope (steeper = more low-freq)')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title('Spectral Decay Slope')

    # --- Panel 6: Timestep evolution for one input ---
    ax = axes[1, 2]
    rep_name = 'noise_std_0.5'
    for r in valid_results:
        if r['name'] == rep_name:
            for ts in sorted(r['timesteps'].keys(), key=int):
                ts_data = r['timesteps'][ts]
                if layer in ts_data:
                    spec = np.array(ts_data[layer]['mean_spectrum'])
                    freqs = np.arange(len(spec))
                    ax.loglog(freqs[1:], spec[1:], linewidth=1,
                              alpha=0.7, label=f't={ts}')
            break
    ax.set_xlabel('Spatial Frequency (radial bins)')
    ax.set_ylabel('Power')
    ax.set_title(f'Spectrum vs Timestep ({rep_name})')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spectra_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Spectra comparison saved.")


def plot_2d_fft_maps(results, output_dir, timestep='500', layer=PRIMARY_LAYER):
    """Plot 2D FFT magnitude maps (averaged over channels) for diverse inputs."""
    # Select representative inputs
    selected = []
    for prefix in ['noise_std_0.5', 'solid_white', 'solid_black', 'solid_gray_50',
                   'gradient_horizontal', 'gradient_radial', 'checkerboard_16',
                   'sine_freq8_ang0', 'sine_freq32_ang0', 'edge_shapes',
                   'texture_white_noise', 'texture_blur_noise', 'shapes_circle_rect',
                   'random_dots', 'low_freq_only', 'high_freq_only',
                   'natural_astronaut', 'contours']:
        for r in results:
            if r['name'] == prefix:
                selected.append(r)
                break

    n = len(selected)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, r in enumerate(selected):
        ts_data = r['timesteps'].get(timestep, {})
        if layer not in ts_data:
            axes[i].axis('off')
            continue
        # We need the full magnitude map — we stored only radial profiles
        # Let's compute it from raw features (handled separately)
        axes[i].text(0.5, 0.5, r['name'][:25], ha='center', va='center',
                     fontsize=7, transform=axes[i].transAxes)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle('2D FFT Magnitude Maps (placeholder - see separate plots)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fft_2d_maps_overview.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_2d_fft_detailed(extractor, inputs, output_dir):
    """
    Detailed 2D FFT magnitude map plots for key inputs.
    Shows: input image, latent, bottleneck feature, and its 2D FFT.
    """
    key_inputs = [
        'noise_std_0.5', 'solid_gray_50', 'gradient_horizontal',
        'checkerboard_16', 'sine_freq8_ang0', 'edge_shapes',
        'texture_white_noise', 'low_freq_only', 'high_freq_only',
        'shapes_circle_rect', 'contours', 'texture_brick',
    ]

    available = [k for k in key_inputs if k in inputs]

    for name in available:
        img_arr = inputs[name]
        img_pil = Image.fromarray(img_arr.astype(np.uint8))

        # Get bottleneck features
        features = extractor.extract_bottleneck(img_pil, timestep=500)
        if PRIMARY_LAYER not in features:
            continue

        feat_map = features[PRIMARY_LAYER][0]  # (C, H, W) — 320×32×32 for down_block_0
        C, H, W = feat_map.shape

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))

        # Input image
        axes[0, 0].imshow(img_arr)
        axes[0, 0].set_title('Input Image', fontsize=9)
        axes[0, 0].axis('off')

        # Bottleneck feature map (first 3 channels as RGB)
        ch_display = feat_map[:3].transpose(1, 2, 0)
        ch_display = (ch_display - ch_display.min()) / (ch_display.max() - ch_display.min() + 1e-8)
        axes[0, 1].imshow(ch_display)
        axes[0, 1].set_title(f'Bottleneck Features\n(first 3/{C} channels)', fontsize=9)
        axes[0, 1].axis('off')

        # Feature map statistics (mean across channels)
        mean_feat = feat_map.mean(axis=0)
        axes[0, 2].imshow(mean_feat, cmap='viridis')
        axes[0, 2].set_title('Mean Across Channels', fontsize=9)
        axes[0, 2].axis('off')
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[0, 2], fraction=0.046)

        # 2D FFT magnitude (mean across channels)
        fft_mags = []
        for c in range(C):
            mag = np.abs(np.fft.fftshift(np.fft.fft2(feat_map[c])))
            fft_mags.append(mag)
        mean_fft = np.mean(fft_mags, axis=0)

        axes[1, 0].imshow(mean_fft, cmap='inferno', norm=LogNorm())
        axes[1, 0].set_title('2D FFT Magnitude (log scale)', fontsize=9)
        axes[1, 0].axis('off')

        # Radial profile
        radial = radial_profile(mean_fft)
        axes[1, 1].semilogy(radial, linewidth=1)
        axes[1, 1].set_xlabel('Spatial Frequency (radial bins)')
        axes[1, 1].set_ylabel('Power (log)')
        axes[1, 1].set_title('Radial Power Spectrum', fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)

        # Per-channel spectra
        axes[1, 2].set_title('Per-Channel Spectra', fontsize=9)
        for c in range(min(C, 20)):
            ch_radial = radial_profile(fft_mags[c])
            axes[1, 2].loglog(ch_radial[1:], alpha=0.3, linewidth=0.4, color=f'C{c % 10}')
        axes[1, 2].set_xlabel('Spatial Frequency')
        axes[1, 2].set_ylabel('Power')
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f'Bottleneck FFT Analysis: {name}', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"fft_detail_{name}.png"), dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [*] Detail plot for {name}")


def plot_frequency_energy_distribution(results, output_dir, timestep='500', layer=PRIMARY_LAYER):
    """Plot how energy is distributed across frequency bands for each input."""
    valid_results = [r for r in results
                     if r['timesteps'].get(timestep, {}).get(layer) is not None]
    if not valid_results:
        print("[!] No valid results for energy distribution, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Band energy stacked bar ---
    ax = axes[0]
    band_energies = {}
    for r in valid_results:
        spec = np.array(r['timesteps'][timestep][layer]['mean_spectrum'])
        max_r = len(spec)
        # Define bands as fractions of max frequency
        band_low = (0, max(1, max_r // 8))
        band_mid = (max_r // 8, max_r // 3)
        band_high = (max_r // 3, max_r)
        total = np.sum(spec[1:]) + 1e-10
        low_e = np.sum(spec[band_low[0]+1:band_low[1]]) / total
        mid_e = np.sum(spec[band_mid[0]:band_mid[1]]) / total
        high_e = np.sum(spec[band_high[0]:]) / total
        band_energies[r['name']] = [low_e, mid_e, high_e]

    # Sort by high-freq energy
    sorted_names = sorted(band_energies.keys(), key=lambda n: band_energies[n][2])
    low_vals = [band_energies[n][0] for n in sorted_names]
    mid_vals = [band_energies[n][1] for n in sorted_names]
    high_vals = [band_energies[n][2] for n in sorted_names]

    y_pos = range(len(sorted_names))
    ax.barh(y_pos, low_vals, color='#1f77b4', label='Low Freq', height=0.7)
    ax.barh(y_pos, mid_vals, color='#ff7f0e', label='Mid Freq', left=low_vals, height=0.7)
    ax.barh(y_pos, high_vals, color='#d62728', label='High Freq',
            left=[l+m for l, m in zip(low_vals, mid_vals)], height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n[:25] for n in sorted_names], fontsize=5)
    ax.set_xlabel('Fraction of Total Spectral Energy')
    ax.set_title('Energy Distribution Across Frequency Bands')
    ax.legend(fontsize=8, loc='lower right')

    # --- DC component analysis ---
    ax = axes[1]
    dc_values = []
    for r in valid_results:
        spec = np.array(r['timesteps'][timestep][layer]['mean_spectrum'])
        dc = spec[0]
        total = np.sum(spec) + 1e-10
        dc_values.append((r['name'], dc / total))

    dc_sorted = sorted(dc_values, key=lambda x: x[1])
    names_dc, vals_dc = zip(*dc_sorted)
    colors_dc = []
    for n in names_dc:
        if 'noise' in n: colors_dc.append('red')
        elif 'solid' in n: colors_dc.append('gray')
        elif 'gradient' in n: colors_dc.append('blue')
        elif 'checkerboard' in n or 'sine' in n: colors_dc.append('orange')
        else: colors_dc.append('steelblue')
    ax.barh(range(len(names_dc)), vals_dc, color=colors_dc, height=0.7)
    ax.set_yticks(range(len(names_dc)))
    ax.set_yticklabels([n[:25] for n in names_dc], fontsize=5)
    ax.set_xlabel('DC / Total Energy Ratio')
    ax.set_title('DC Component Dominance (higher = more uniform)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frequency_energy_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Frequency energy distribution saved.")


def plot_input_vs_bottleneck_correlation(inputs, results, output_dir,
                                          timestep='500', layer=PRIMARY_LAYER):
    """
    Compare input image FFT spectra with bottleneck FFT spectra.
    """
    valid_results = {r['name']: r for r in results
                     if r['timesteps'].get(timestep, {}).get(layer) is not None}
    if not valid_results:
        print("[!] No valid results for correlation plot, skipping.")
        return
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    key_pairs = [
        ('noise_std_0.5', 'Gaussian Noise'),
        ('solid_gray_50', 'Solid Gray'),
        ('gradient_horizontal', 'H. Gradient'),
        ('checkerboard_16', 'Checkerboard 16'),
        ('sine_freq8_ang0', 'Sine 8 Hz'),
        ('sine_freq32_ang0', 'Sine 32 Hz'),
        ('edge_shapes', 'Edges'),
        ('texture_white_noise', 'White Noise'),
        ('low_freq_only', 'Low-Freq Only'),
        ('high_freq_only', 'High-Freq Only'),
        ('texture_blur_noise', 'Blurred Noise'),
        ('contours', 'Contours'),
    ]

    for idx, (name, title) in enumerate(key_pairs):
        ax = axes[idx]
        if name not in inputs:
            ax.axis('off')
            continue

        # Input FFT
        img_gray = np.array(Image.fromarray(inputs[name]).convert('L')).astype(float)
        img_fft = np.abs(np.fft.fftshift(np.fft.fft2(img_gray)))
        input_radial = radial_profile(img_fft)
        input_radial_norm = input_radial / (input_radial.max() + 1e-10)

        # Bottleneck FFT
        r = valid_results.get(name)
        if r is None:
            ax.axis('off')
            continue
        ts_data = r['timesteps'].get(timestep, {})
        if layer not in ts_data:
            ax.axis('off')
            continue
        bn_spec = np.array(ts_data[layer]['mean_spectrum'])
        bn_spec_norm = bn_spec / (bn_spec.max() + 1e-10)

        # Normalize to same length for comparison
        min_len = min(len(input_radial_norm), len(bn_spec_norm))
        ax.semilogy(input_radial_norm[:min_len], label='Input Image', linewidth=1.2, alpha=0.8)
        ax.semilogy(bn_spec_norm[:min_len], label='Bottleneck', linewidth=1.2, alpha=0.8)

        # Correlation
        if min_len > 3:
            corr = np.corrcoef(input_radial_norm[1:min_len], bn_spec_norm[1:min_len])[0, 1]
            ax.set_title(f'{title}\ncorr={corr:.3f}', fontsize=8)
        else:
            ax.set_title(title, fontsize=8)

        ax.set_xlabel('Freq bins')
        ax.set_ylabel('Norm. Power')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Input vs Bottleneck Frequency Spectra', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_vs_bottleneck_correlation.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Input-bottleneck correlation saved.")


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("=" * 60)
    print("UNet Bottleneck FFT Analysis")
    print("=" * 60)

    # Step 1: Generate inputs
    print("\n[1/5] Generating input images...")
    inputs = generate_inputs()
    print(f"  Generated {len(inputs)} input types")

    # Save inputs
    for name, img in inputs.items():
        Image.fromarray(img.astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, "inputs", f"{name}.png"))

    # Step 2: Load models
    print("\n[2/5] Loading models...")
    extractor = UNetBottleneckExtractor()

    # Step 3: Extract bottleneck features and compute FFT
    print(f"\n[3/5] Extracting bottleneck features for {len(inputs)} inputs...")
    results = []
    for i, (name, img_array) in enumerate(sorted(inputs.items())):
        print(f"  [{i+1}/{len(inputs)}] {name}...")
        try:
            result = extractor.analyze_input(name, img_array)
            results.append(result)
        except Exception as e:
            print(f"    ERROR: {e}")

    # Save raw data
    print(f"\n[4/5] Saving results...")
    with open(os.path.join(OUTPUT_DIR, "data", "spectra_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Step 4: Visualizations
    print(f"\n[5/5] Generating visualizations...")
    plot_input_gallery(inputs, OUTPUT_DIR)
    plot_spectra_comparison(results, OUTPUT_DIR)
    plot_frequency_energy_distribution(results, OUTPUT_DIR)
    plot_input_vs_bottleneck_correlation(inputs, results, OUTPUT_DIR)
    plot_2d_fft_detailed(extractor, inputs, os.path.join(OUTPUT_DIR, "spectra"))

    print(f"\n{'=' * 60}")
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
