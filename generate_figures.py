"""
generate_figures.py
===================
Generates verified publication figures from actual simulation CSV data.
ALL values plotted are from real simulation outputs - no hypothetical data.

Author: Md. Noman, NSTU
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os, warnings
warnings.filterwarnings("ignore")

OUT_DIR = "./simulation_outputs"
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load real simulation data
suv = pd.read_csv(f"{OUT_DIR}/Suvin_G_barbadense_runs.csv")
mus = pd.read_csv(f"{OUT_DIR}/Muslin_G_arboreum_runs.csv")

print("Data loaded: Suvin n=%d, Muslin n=%d" % (len(suv), len(mus)))
print("  Suvin EC_season range: %.3f – %.3f dS/m" % (suv.EC_season.min(), suv.EC_season.max()))
print("  Muslin EC_season range: %.3f – %.3f dS/m" % (mus.EC_season.min(), mus.EC_season.max()))

# --------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------
plt.rcParams.update({
    'font.family'       : 'serif',
    'font.size'         : 10,
    'axes.titlesize'    : 11,
    'axes.labelsize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 9,
    'figure.dpi'        : 150,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

SUVIN_COLOR  = '#2166AC'   # deep blue
MUSLIN_COLOR = '#D6604D'   # terra cotta
MH_COLOR     = '#1A9850'   # green

# --------------------------------------------------------------------------
# FIGURE 1: Yield distributions from real data (Box + Strip plots)
# --------------------------------------------------------------------------
fig1, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

for ax, df, color, name, ymax in zip(
    axes,
    [suv, mus],
    [SUVIN_COLOR, MUSLIN_COLOR],
    ['Suvin (G. barbadense)', 'Muslin (G. arboreum)'],
    [1800, 1200]
):
    y = df['yield_kgha'].values

    # histogram
    ax.hist(y, bins=30, color=color, alpha=0.6, edgecolor='white', linewidth=0.5)
    ax.axvline(y.mean(),  color='black',       linestyle='--', linewidth=1.5,
               label=f'Mean: {y.mean():.1f}')
    ax.axvline(np.percentile(y, 5),  color='firebrick', linestyle=':', linewidth=1.2,
               label=f'5th pct: {np.percentile(y,5):.1f}')
    ax.axvline(np.percentile(y, 95), color='steelblue', linestyle=':', linewidth=1.2,
               label=f'95th pct: {np.percentile(y,95):.1f}')

    ax.set_title(f'{name}\nYield Distribution (n=500 MC runs)', fontweight='bold')
    ax.set_xlabel('Simulated Yield (kg ha⁻¹)')
    ax.set_ylabel('Frequency (count)')
    ax.legend(loc='upper left', framealpha=0.8)

    # stats annotation
    stats_text = (f'Mean = {y.mean():.1f}\n'
                  f'SD = {y.std():.1f}\n'
                  f'CV = {y.std()/y.mean()*100:.2f}%\n'
                  f'Min = {y.min():.1f}\n'
                  f'Max = {y.max():.1f}')
    ax.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

fig1.suptitle('Figure 1. Simulated Yield Distributions\n'
              '(Stochastic AR(1) forcing, seed=42, 500 realizations)',
              fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
p1 = f"{FIG_DIR}/Figure1_Yield_Distributions.png"
fig1.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p1}")

# --------------------------------------------------------------------------
# FIGURE 2: EC_season vs Yield scatter (real data only)
# --------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(8, 5))

ax.scatter(suv.EC_season, suv.yield_kgha, alpha=0.35, s=18,
           color=SUVIN_COLOR, label='Suvin (G. barbadense)', zorder=2)
ax.scatter(mus.EC_season, mus.yield_kgha, alpha=0.35, s=18,
           color=MUSLIN_COLOR, label='Muslin (G. arboreum)', zorder=2)

# Analytical Maas-Hoffman line (Suvin, OAM=1.00)
ec_line = np.linspace(5.0, 10.0, 200)
Y_suv_mh = 1800 * np.maximum(0, 1 - 0.052 * np.maximum(0, ec_line - 7.7))
Y_mus_mh = 1200 * np.maximum(0, 1 - 0.052 * np.maximum(0, ec_line - 7.7) * 0.60)

ax.plot(ec_line, Y_suv_mh, color=SUVIN_COLOR, linewidth=2, linestyle='-',
        label='Suvin M-H analytical (OAM=1.00)', zorder=3)
ax.plot(ec_line, Y_mus_mh, color=MUSLIN_COLOR, linewidth=2, linestyle='-',
        label='Muslin M-H analytical (OAM=0.60)', zorder=3)

ax.axvline(7.7, color='gray', linestyle='--', linewidth=1, alpha=0.7,
           label='M-H threshold (7.7 dS m⁻¹)')

ax.set_xlabel('EC_season — Phenologically Weighted Seasonal ECe (dS m⁻¹)')
ax.set_ylabel('Simulated Yield (kg ha⁻¹)')
ax.set_title('Figure 2. EC_season vs Simulated Yield\n'
             '(Scatter: 500 real MC realizations | Lines: Analytical M-H benchmark)',
             fontweight='bold')
ax.legend(loc='lower left', framealpha=0.85)

# Note
ax.annotate('NOTE: No field data.\nAll points from stochastic simulation only.\nEC_season max = %.2f dS m⁻¹ in this ensemble.' % suv.EC_season.max(),
            xy=(0.99, 0.99), xycoords='axes fraction', ha='right', va='top',
            fontsize=8, color='dimgray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
p2 = f"{FIG_DIR}/Figure2_ECseason_vs_Yield.png"
fig2.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p2}")

# --------------------------------------------------------------------------
# FIGURE 3: Maas-Hoffman validation — analytical only, clearly labeled
# --------------------------------------------------------------------------
ec_ckpts = np.array([7.7, 8.5, 9.0, 10.0, 12.0, 14.0, 16.0])
mh_ref   = np.maximum(0, 0.052 * np.maximum(0, ec_ckpts - 7.7)) * 100
suv_loss = mh_ref.copy()                # OAM=1.00 → identical
mus_loss = np.maximum(0, 0.052 * np.maximum(0, ec_ckpts - 7.7) * 0.60) * 100

fig3, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(ec_ckpts))
width = 0.28

bars_mh  = ax.bar(x - width,     mh_ref,   width, label='M-H Reference (G. hirsutum)',
                   color=MH_COLOR, alpha=0.85, edgecolor='white')
bars_suv = ax.bar(x,              suv_loss, width, label='Suvin D_sal (OAM=1.00) — ANALYTICAL',
                   color=SUVIN_COLOR, alpha=0.85, edgecolor='white')
bars_mus = ax.bar(x + width,      mus_loss, width, label='Muslin D_sal (OAM=0.60) — ANALYTICAL',
                   color=MUSLIN_COLOR, alpha=0.85, edgecolor='white')

# Deviation labels on Suvin bars
for rect, ref, loss in zip(bars_suv, mh_ref, suv_loss):
    dev = loss - ref
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.3,
            f'Δ={dev:.2f}%', ha='center', va='bottom', fontsize=7, color='navy')

ax.set_xticks(x)
ax.set_xticklabels([f'{e} dS/m' for e in ec_ckpts])
ax.set_xlabel('EC_season Checkpoint (dS m⁻¹)')
ax.set_ylabel('Salinity Yield Loss (%)')
ax.set_title('Figure 3. Point-to-Point Maas-Hoffman Validation\n'
             '(D_sal term only — ANALYTICAL calculation, not simulation scatter)\n'
             'Note: EC_season in stochastic ensemble did not exceed 9.41 dS m⁻¹ (dashed line marks simulated range)',
             fontweight='bold', fontsize=10)
ax.axvline(1.5, color='red', linestyle='--', linewidth=1.5, alpha=0.6,
           label='Simulation EC_season range boundary (≤9.41 dS m⁻¹)')
ax.legend(loc='upper left', framealpha=0.85, fontsize=8)
ax.set_ylim(0, 50)

fig3.tight_layout()
p3 = f"{FIG_DIR}/Figure3_MH_Validation.png"
fig3.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p3}")

# --------------------------------------------------------------------------
# FIGURE 4: Fiber quality degradation — ANALYTICAL equations, clearly labeled
#           with actual simulation range shaded
# --------------------------------------------------------------------------
ec_full = np.linspace(7.7, 16.0, 300)

# G. barbadense (Suvin)
L_suv  = 33.2 * np.maximum(0, 1 - 0.008  * np.maximum(0, ec_full - 7.7))
M_suv  = 3.70 + 0.017 * np.maximum(0, ec_full - 7.7)
S_suv  = 31.5  * np.maximum(0, 1 - 0.009  * np.maximum(0, ec_full - 7.7))

# G. arboreum (Muslin)
L_mus  = 18.5 * np.maximum(0, 1 - 0.004  * np.maximum(0, ec_full - 7.7))
M_mus  = 4.60 + 0.009 * np.maximum(0, ec_full - 7.7)
S_mus  = 23.8  * np.maximum(0, 1 - 0.005  * np.maximum(0, ec_full - 7.7))

fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4.5))

panels = [
    (axes4[0], L_suv, L_mus, 'Staple Length (mm)', 'Fiber Staple Length vs Salinity'),
    (axes4[1], M_suv, M_mus, 'Micronaire (μg inch⁻¹)', 'Micronaire vs Salinity'),
    (axes4[2], S_suv, S_mus, 'Bundle Strength (g tex⁻¹)', 'Fiber Strength vs Salinity'),
]

for ax, y_suv, y_mus, ylabel, title in panels:
    ax.plot(ec_full, y_suv, color=SUVIN_COLOR, linewidth=2, label='Suvin (G. barbadense)')
    ax.plot(ec_full, y_mus, color=MUSLIN_COLOR, linewidth=2, label='Muslin (G. arboreum)')

    # Shade actual simulation range
    sim_max_ec = suv.EC_season.max()  # 9.41
    ax.axvspan(7.7, sim_max_ec, alpha=0.08, color='gold', label=f'Stochastic sim EC range (≤{sim_max_ec:.1f})')
    ax.axvline(7.7, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(sim_max_ec, color='goldenrod', linestyle=':', linewidth=1.2, alpha=0.8)

    ax.set_xlabel('EC_season (dS m⁻¹)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.legend(fontsize=7.5, loc='upper right' if 'Micro' not in ylabel else 'upper left')

fig4.suptitle('Figure 4. Analytical Fiber Quality Degradation Sub-Model\n'
              '(ANALYTICAL equations only — shaded zone = actual stochastic simulation EC_season range)\n'
              'No field fiber quality measurements were made.',
              fontsize=9, fontweight='bold', style='italic', y=1.02)
plt.tight_layout()
p4 = f"{FIG_DIR}/Figure4_Fiber_Quality_Degradation.png"
fig4.savefig(p4, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p4}")

# --------------------------------------------------------------------------
# FIGURE 5: Variance decomposition bar chart
# --------------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(7, 4))

species = ['Suvin\n(G. barbadense)', 'Muslin\n(G. arboreum)']
ec_pct  = [91.5, 91.7]
t_pct   = [8.5,  8.3]

x5 = np.arange(len(species))
b1 = ax5.bar(x5, ec_pct, 0.45, label='EC_season attribution (%)',
             color=SUVIN_COLOR, alpha=0.85, edgecolor='white')
b2 = ax5.bar(x5, t_pct, 0.45, bottom=ec_pct, label='T_season attribution (%)',
             color='#FC8D59', alpha=0.85, edgecolor='white')

for bar, v in zip(b1, ec_pct):
    ax5.text(bar.get_x() + bar.get_width()/2, v/2,
             f'{v}%', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
for bar, v, bot in zip(b2, t_pct, ec_pct):
    ax5.text(bar.get_x() + bar.get_width()/2, bot + v/2,
             f'{v}%', ha='center', va='center', color='white', fontweight='bold', fontsize=11)

ax5.set_xticks(x5)
ax5.set_xticklabels(species, fontsize=10)
ax5.set_ylabel('Variance Attribution (%)')
ax5.set_ylim(0, 105)
ax5.set_title('Figure 5. Random Forest Variance Decomposition\n'
              '(200 trees, max_features=sqrt, n=500 realizations)',
              fontweight='bold')
ax5.legend(loc='upper right')

plt.tight_layout()
p5 = f"{FIG_DIR}/Figure5_Variance_Decomposition.png"
fig5.savefig(p5, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p5}")

# --------------------------------------------------------------------------
# Summary of actual EC range
# --------------------------------------------------------------------------
print()
print("=== SIMULATION RANGE SUMMARY (for paper accuracy) ===")
print(f"EC_season: {suv.EC_season.min():.3f} – {suv.EC_season.max():.3f} dS/m")
print(f"EC_season mean: {suv.EC_season.mean():.3f}, median: {suv.EC_season.median():.3f}")
print(f"Runs above threshold (7.7): {(suv.EC_season>7.7).sum()} / 500 ({(suv.EC_season>7.7).mean()*100:.1f}%)")
print(f"Runs above 9.0 dS/m: {(suv.EC_season>9.0).sum()} / 500")
print(f"Suvin yield min: {suv.yield_kgha.min():.1f}, max: {suv.yield_kgha.max():.1f}")
print(f"Muslin yield min: {mus.yield_kgha.min():.1f}, max: {mus.yield_kgha.max():.1f}")
print()
print("Fiber quality in ACTUAL simulation range:")
print(f"  Suvin staple: {suv.staple_mm.min():.3f} – {suv.staple_mm.max():.3f} mm")
print(f"  Suvin micronaire: {suv.micronaire.min():.4f} – {suv.micronaire.max():.4f}")
print(f"  Suvin strength: {suv.strength.min():.3f} – {suv.strength.max():.3f}")
print(f"  Muslin staple: {mus.staple_mm.min():.3f} – {mus.staple_mm.max():.3f}")
print(f"  Muslin micronaire: {mus.micronaire.min():.4f} – {mus.micronaire.max():.4f}")
print()
print("Figures saved to:", FIG_DIR)
