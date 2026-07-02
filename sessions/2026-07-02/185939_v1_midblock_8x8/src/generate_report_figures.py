#!/usr/bin/env python3
"""Generate the final analysis report with enhanced visualizations."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

OUTPUT_DIR = "/mnt/data/tianyi/sessions/2026-07-02/185939/outputs"
REPORT_DIR = os.path.join(OUTPUT_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "data", "spectra_results.json")) as f:
    results = json.load(f)

timestep = '500'

# ============================================================================
# Figure 1: Multi-Resolution Spectral Evolution
# ============================================================================
def plot_multiresolution_evolution():
    """Show how spectra evolve across UNet layers for key input types."""
    layers = ['down_block_0', 'down_block_1', 'mid_block', 'up_block_0', 'up_block_1', 'up_block_2']
    key_inputs = ['noise_std_0.5', 'solid_gray_50', 'gradient_horizontal',
                  'checkerboard_16', 'sine_freq8_ang0', 'edge_shapes',
                  'texture_white_noise', 'natural_astronaut', 'contours',
                  'low_freq_only', 'high_freq_only']

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    for li, layer in enumerate(layers):
        ax = axes[li]
        for name in key_inputs:
            r = next((x for x in results if x['name'] == name), None)
            if r is None:
                continue
            ts_data = r['timesteps'].get(timestep, {})
            if layer not in ts_data:
                continue
            spec = np.array(ts_data[layer]['mean_spectrum'])
            shape = ts_data[layer]['spatial_shape']
            freqs = np.arange(len(spec))
            ax.loglog(freqs[1:], spec[1:], linewidth=1.2, alpha=0.8, label=name[:20])

        ax.set_xlabel('Spatial Frequency (radial bins)')
        ax.set_ylabel('Power')
        ax.set_title(f'{layer}\n(shape={ts_data[layer]["spatial_shape"] if layer in ts_data else "?"})', fontsize=10)
        ax.grid(True, alpha=0.3)
        if li == 0:
            ax.legend(fontsize=5, loc='lower left', ncol=2)

    plt.suptitle('Frequency Spectrum Evolution Through UNet Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig1_multiresolution_evolution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[*] Fig1: Multi-resolution evolution saved.")


# ============================================================================
# Figure 2: Category Comparison at Bottleneck
# ============================================================================
def plot_bottleneck_category_comparison():
    """Radar/spider chart comparing frequency characteristics by input category at bottleneck."""
    layer = 'mid_block'

    categories = {
        'Gaussian Noise': ['noise_std_0.1', 'noise_std_0.3', 'noise_std_0.5', 'noise_std_0.8', 'noise_std_1.0'],
        'Solid Colors': ['solid_black', 'solid_white', 'solid_gray_50', 'solid_red', 'solid_green', 'solid_blue'],
        'Gradients': ['gradient_horizontal', 'gradient_vertical', 'gradient_radial', 'gradient_diagonal'],
        'Checkerboard': ['checkerboard_2', 'checkerboard_4', 'checkerboard_8', 'checkerboard_16', 'checkerboard_32'],
        'Sine Gratings': ['sine_freq2_ang0', 'sine_freq8_ang0', 'sine_freq32_ang0', 'sine_freq8_ang45', 'sine_freq8_ang90'],
        'Edges/Contours': ['edge_shapes', 'contours'],
        'Textures': ['texture_white_noise', 'texture_blur_noise', 'texture_brick', 'texture_wood_like'],
        'Natural Images': ['natural_astronaut', 'natural_coffee'],
        'Freq-Filtered': ['low_freq_only', 'high_freq_only', 'band_pass_ring'],
        'Shapes': ['shapes_circle_rect', 'random_dots'],
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # --- Subplot 1: DC Ratio by category (bar) ---
    ax = axes[0, 0]
    cat_names = []
    cat_dc = []
    cat_dc_std = []
    cat_high = []
    cat_high_std = []
    for cat, members in categories.items():
        dc_vals = []
        high_vals = []
        for name in members:
            r = next((x for x in results if x['name'] == name), None)
            if r is None:
                continue
            ts_data = r['timesteps'].get(timestep, {})
            if layer not in ts_data:
                continue
            spec = np.array(ts_data[layer]['mean_spectrum'])
            total = np.sum(spec)
            dc_vals.append(spec[0] / total * 100)
            high_vals.append(np.sum(spec[len(spec)//3:]) / total * 100)
        if dc_vals:
            cat_names.append(cat)
            cat_dc.append(np.mean(dc_vals))
            cat_dc_std.append(np.std(dc_vals))
            cat_high.append(np.mean(high_vals))
            cat_high_std.append(np.std(high_vals))

    x = np.arange(len(cat_names))
    w = 0.35
    bars1 = ax.bar(x - w/2, cat_dc, w, yerr=cat_dc_std, label='DC Power %', color='#4472C4', capsize=3)
    bars2 = ax.bar(x + w/2, cat_high, w, yerr=cat_high_std, label='High-Freq Power %', color='#ED7D31', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('% of Total Power')
    ax.set_title(f'DC vs High-Freq Power by Category (Bottleneck: {layer})')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # --- Subplot 2: DC vs High-Freq scatter ---
    ax = axes[0, 1]
    for cat, members in categories.items():
        dc_vals = []
        high_vals = []
        names_list = []
        for name in members:
            r = next((x for x in results if x['name'] == name), None)
            if r is None: continue
            ts_data = r['timesteps'].get(timestep, {})
            if layer not in ts_data: continue
            spec = np.array(ts_data[layer]['mean_spectrum'])
            total = np.sum(spec)
            dc_vals.append(spec[0] / total * 100)
            high_vals.append(np.sum(spec[len(spec)//3:]) / total * 100)
            names_list.append(name)
        if dc_vals:
            ax.scatter(dc_vals, high_vals, label=cat, s=30, alpha=0.7)

    ax.set_xlabel('DC Power %')
    ax.set_ylabel('High-Freq Power %')
    ax.set_title('DC vs High-Freq: Each Point = One Input')
    ax.legend(fontsize=5, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: Spectral Entropy by category ---
    ax = axes[0, 2]
    cat_entropy = []
    cat_entropy_std = []
    for cat, members in categories.items():
        ent_vals = []
        for name in members:
            r = next((x for x in results if x['name'] == name), None)
            if r is None: continue
            ts_data = r['timesteps'].get(timestep, {})
            if layer not in ts_data: continue
            spec = np.array(ts_data[layer]['mean_spectrum'])
            spec_norm = spec[1:] / (np.sum(spec[1:]) + 1e-10)
            entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-10))
            max_entropy = np.log(len(spec) - 1)
            ent_vals.append(entropy / max_entropy)
        if ent_vals:
            cat_entropy.append(np.mean(ent_vals))
            cat_entropy_std.append(np.std(ent_vals))

    ax.barh(range(len(cat_names)), cat_entropy, xerr=cat_entropy_std, capsize=3,
            color=plt.cm.viridis(np.linspace(0.2, 0.9, len(cat_names))))
    ax.set_yticks(range(len(cat_names)))
    ax.set_yticklabels(cat_names, fontsize=8)
    ax.set_xlabel('Normalized Spectral Entropy')
    ax.set_title('Spectral Flatness (higher = more uniform spectrum)')
    ax.grid(axis='x', alpha=0.3)

    # --- Subplot 4: Frequency spectrum overlaid by category ---
    ax = axes[1, 0]
    cat_colors = {
        'Gaussian Noise': 'red', 'Solid Colors': 'gray', 'Gradients': 'blue',
        'Checkerboard': 'orange', 'Sine Gratings': 'green', 'Edges/Contours': 'purple',
        'Textures': 'brown', 'Natural Images': 'darkgreen', 'Freq-Filtered': 'cyan',
        'Shapes': 'pink'
    }
    for cat, members in categories.items():
        all_specs = []
        for name in members:
            r = next((x for x in results if x['name'] == name), None)
            if r is None: continue
            ts_data = r['timesteps'].get(timestep, {})
            if layer not in ts_data: continue
            spec = np.array(ts_data[layer]['mean_spectrum'])
            all_specs.append(spec)
        if all_specs:
            mean_spec = np.mean(all_specs, axis=0)
            std_spec = np.std(all_specs, axis=0)
            freqs = np.arange(len(mean_spec))
            ax.loglog(freqs[1:], mean_spec[1:], linewidth=2, label=cat, color=cat_colors.get(cat, 'black'))
            ax.fill_between(freqs[1:],
                           np.maximum(mean_spec[1:] - std_spec[1:], 1e-10),
                           mean_spec[1:] + std_spec[1:],
                           alpha=0.15, color=cat_colors.get(cat, 'black'))

    ax.set_xlabel('Spatial Frequency (radial bins)')
    ax.set_ylabel('Power')
    ax.set_title(f'Mean Spectra by Category (±1 std) - Bottleneck')
    ax.legend(fontsize=5, loc='lower left')
    ax.grid(True, alpha=0.3)

    # --- Subplot 5: DC Ratio evolution across layers ---
    ax = axes[1, 1]
    layers_list = ['down_block_0', 'down_block_1', 'mid_block', 'up_block_0', 'up_block_1', 'up_block_2']
    for cat, members in [('Natural', ['natural_astronaut']), ('Noise', ['noise_std_0.5']),
                          ('Sine', ['sine_freq8_ang0']), ('Solid', ['solid_gray_50']),
                          ('Edge', ['edge_shapes']), ('Checker', ['checkerboard_16']),
                          ('Gradient', ['gradient_horizontal'])]:
        dc_vals = []
        for layer_l in layers_list:
            r = next((x for x in results if x['name'] == members[0]), None)
            if r is None: continue
            ts_data = r['timesteps'].get(timestep, {})
            if layer_l in ts_data:
                spec = np.array(ts_data[layer_l]['mean_spectrum'])
                dc_vals.append(spec[0] / np.sum(spec) * 100)
            else:
                dc_vals.append(np.nan)
        ax.plot(range(len(layers_list)), dc_vals, 'o-', linewidth=1.5, label=cat, markersize=5)

    ax.set_xticks(range(len(layers_list)))
    ax.set_xticklabels([l.replace('_', '\n') for l in layers_list], fontsize=7)
    ax.set_ylabel('DC Power %')
    ax.set_title('DC Component Evolution Through UNet')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Subplot 6: Input Spatial Frequency vs Bottleneck Response ---
    ax = axes[1, 2]
    # For sine gratings: compare input frequency vs bottleneck high-freq ratio
    sine_freqs = [2, 8, 16, 32]
    sine_angles = [0, 45, 90]
    for ang in sine_angles:
        high_ratios = []
        for freq in sine_freqs:
            name = f"sine_freq{freq}_ang{ang}"
            if freq == 16 and ang not in [30]:  # skip special case
                if ang == 45: continue
            if freq == 16:
                name = "sine_freq16_ang30"
            r = next((x for x in results if x['name'] == name), None)
            if r is None: continue
            ts_data = r['timesteps'].get(timestep, {})
            if layer not in ts_data: continue
            spec = np.array(ts_data[layer]['mean_spectrum'])
            total_ac = np.sum(spec[1:])
            high = np.sum(spec[len(spec)//3:]) / total_ac if total_ac > 0 else 0
            high_ratios.append(high * 100)
            # Filter for valid freq-angle combos
        valid_freqs = [2, 8, 32]
        if ang == 0:
            valid_freqs = [2, 8, 32]
            valid_ratios = high_ratios[:3]
        elif ang == 45:
            valid_freqs = [8]
            valid_ratios = high_ratios[:1]
        else:
            valid_freqs = [8]
            valid_ratios = high_ratios[:1]

    # Broader comparison: input type vs DC ratio
    input_labels = ['Noise\n0.5', 'Solid\nGray', 'Checker\n16', 'Sine\n8Hz',
                    'Texture\nNoise', 'Gradient\nHoriz', 'Edge', 'Natural\nAstro',
                    'LowFreq\nOnly', 'HighFreq\nOnly']
    input_names = ['noise_std_0.5', 'solid_gray_50', 'checkerboard_16',
                   'sine_freq8_ang0', 'texture_white_noise', 'gradient_horizontal',
                   'edge_shapes', 'natural_astronaut', 'low_freq_only', 'high_freq_only']

    dc_vals_plot = []
    high_vals_plot = []
    for name in input_names:
        r = next((x for x in results if x['name'] == name), None)
        if r is None: continue
        ts_data = r['timesteps'].get(timestep, {})
        if layer not in ts_data: continue
        spec = np.array(ts_data[layer]['mean_spectrum'])
        total = np.sum(spec)
        dc_vals_plot.append(spec[0] / total * 100)
        high_vals_plot.append(np.sum(spec[len(spec)//3:]) / total * 100)

    x_pos = np.arange(len(dc_vals_plot))
    w = 0.35
    ax.bar(x_pos - w/2, dc_vals_plot, w, label='DC %', color='#4472C4', alpha=0.8)
    ax.bar(x_pos + w/2, high_vals_plot, w, label='High-Freq %', color='#ED7D31', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(input_labels, fontsize=7)
    ax.set_ylabel('% of Total Power')
    ax.set_title('Frequency Profile by Input Type')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Bottleneck Frequency Analysis: Input Category Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig2_category_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[*] Fig2: Category comparison saved.")


# ============================================================================
# Figure 3: Timestep Evolution
# ============================================================================
def plot_timestep_evolution():
    """How bottleneck spectra change with diffusion timestep."""
    layer = 'mid_block'
    key_inputs = ['noise_std_0.5', 'solid_gray_50', 'checkerboard_16', 'sine_freq8_ang0',
                  'edge_shapes', 'natural_astronaut', 'texture_white_noise']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, name in enumerate(key_inputs):
        ax = axes[idx]
        r = next((x for x in results if x['name'] == name), None)
        if r is None: continue

        for ts_str in sorted(r['timesteps'].keys(), key=int):
            ts_data = r['timesteps'][ts_str]
            if layer not in ts_data: continue
            spec = np.array(ts_data[layer]['mean_spectrum'])
            freqs = np.arange(len(spec))
            ax.loglog(freqs[1:], spec[1:], linewidth=1, alpha=0.7, label=f't={ts_str}')

        ax.set_xlabel('Freq bins')
        ax.set_ylabel('Power')
        ax.set_title(f'{name[:25]}')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6)

    # DC evolution plot
    ax = axes[7]
    for name in key_inputs:
        r = next((x for x in results if x['name'] == name), None)
        if r is None: continue
        ts_list = sorted(r['timesteps'].keys(), key=int)
        dc_vals = []
        for ts_str in ts_list:
            ts_data = r['timesteps'][ts_str]
            if layer in ts_data:
                spec = np.array(ts_data[layer]['mean_spectrum'])
                dc_vals.append(spec[0] / np.sum(spec) * 100)
        ax.plot([int(t) for t in ts_list], dc_vals, 'o-', linewidth=1.5, label=name[:20], markersize=4)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('DC Power %')
    ax.set_title('DC Component vs Timestep')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Bottleneck Spectrum vs Diffusion Timestep', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig3_timestep_evolution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[*] Fig3: Timestep evolution saved.")


# ============================================================================
# Figure 4: Input-to-Bottleneck Transfer Function
# ============================================================================
def plot_transfer_function():
    """Characterize the UNet bottleneck as a spatial frequency filter."""
    layer = 'down_block_0'  # Use higher-res layer for better freq analysis
    key_inputs = ['noise_std_0.5', 'solid_gray_50', 'checkerboard_16',
                  'sine_freq2_ang0', 'sine_freq8_ang0', 'sine_freq32_ang0',
                  'gradient_horizontal', 'gradient_radial',
                  'low_freq_only', 'high_freq_only', 'band_pass_ring']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, name in enumerate(key_inputs[:6]):
        ax = axes[idx]
        r = next((x for x in results if x['name'] == name), None)
        if r is None: continue

        # Get spectra at different layers
        for layer_l, color, ls in [('down_block_0', 'blue', '-'),
                                     ('down_block_1', 'cyan', '--'),
                                     ('mid_block', 'red', '-'),
                                     ('up_block_1', 'orange', '--'),
                                     ('up_block_2', 'green', ':')]:
            ts_data = r['timesteps'].get(timestep, {})
            if layer_l not in ts_data: continue
            spec = np.array(ts_data[layer_l]['mean_spectrum'])
            spec_norm = spec / (np.max(spec) + 1e-10)
            freqs_norm = np.linspace(0, 1, len(spec))
            ax.semilogy(freqs_norm, spec_norm, color=color, linestyle=ls, linewidth=1,
                       alpha=0.8, label=layer_l)

        ax.set_xlabel('Normalized Frequency')
        ax.set_ylabel('Normalized Power')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=5)

    # Transfer function: ratio of output/input spectrum
    ax = axes[4]
    for name in ['sine_freq2_ang0', 'sine_freq8_ang0', 'sine_freq32_ang0',
                 'checkerboard_16', 'noise_std_0.5']:
        r = next((x for x in results if x['name'] == name), None)
        if r is None: continue
        ts_data = r['timesteps'].get(timestep, {})
        if 'down_block_0' not in ts_data or 'mid_block' not in ts_data: continue
        input_spec = np.array(ts_data['down_block_0']['mean_spectrum'])
        output_spec = np.array(ts_data['mid_block']['mean_spectrum'])
        # Interpolate to same length
        min_len = min(len(input_spec), len(output_spec))
        ratio = output_spec[:min_len] / (input_spec[:min_len] + 1e-10)
        freqs = np.arange(min_len)
        ax.semilogy(freqs, ratio, linewidth=1.5, label=name[:20])

    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Frequency bin')
    ax.set_ylabel('Bottleneck / Early-Down Ratio')
    ax.set_title('Effective Frequency Transfer Function')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Summary: High-freq attenuation factor by input type
    ax = axes[5]
    input_cats = ['noise_std_0.5', 'solid_gray_50', 'checkerboard_16', 'checkerboard_2',
                  'sine_freq2_ang0', 'sine_freq8_ang0', 'sine_freq32_ang0',
                  'edge_shapes', 'texture_white_noise', 'natural_astronaut',
                  'low_freq_only', 'high_freq_only', 'gradient_horizontal', 'gradient_radial']
    atten_factors = []
    labels = []
    for name in input_cats:
        r = next((x for x in results if x['name'] == name), None)
        if r is None: continue
        ts_data = r['timesteps'].get(timestep, {})
        if 'down_block_0' not in ts_data or 'mid_block' not in ts_data: continue
        input_spec = np.array(ts_data['down_block_0']['mean_spectrum'])
        output_spec = np.array(ts_data['mid_block']['mean_spectrum'])
        # High-freq energy ratio
        in_high = np.sum(input_spec[len(input_spec)//2:]) / (np.sum(input_spec[1:]) + 1e-10)
        out_high = np.sum(output_spec[len(output_spec)//2:]) / (np.sum(output_spec[1:]) + 1e-10)
        atten = out_high / (in_high + 1e-10)
        atten_factors.append(atten)
        labels.append(name[:20])

    sorted_idx = np.argsort(atten_factors)
    ax.barh(range(len(sorted_idx)), [atten_factors[i] for i in sorted_idx],
            color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_idx))))
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([labels[i] for i in sorted_idx], fontsize=6)
    ax.set_xlabel('High-Freq Preservation Ratio (mid/down_block_0)')
    ax.set_title('High-Frequency Attenuation\nby Bottleneck (lower = more filtered)')
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.3)
    ax.grid(axis='x', alpha=0.3)

    plt.suptitle('UNet as a Spatial Frequency Filter', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig4_transfer_function.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[*] Fig4: Transfer function saved.")


# ============================================================================
# Run all
# ============================================================================
plot_multiresolution_evolution()
plot_bottleneck_category_comparison()
plot_timestep_evolution()
plot_transfer_function()
print(f"\nAll report figures saved to: {REPORT_DIR}")
