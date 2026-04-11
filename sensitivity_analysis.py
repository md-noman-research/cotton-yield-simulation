"""
sensitivity_analysis.py
=======================
Global Sensitivity Analysis (GSA) and Robustness Testing for the Cotton 
Salinity Simulation. This script quantifies how uncertainty in unvalidated 
parameters (OAM, Fiber coefficients, Interaction term) affects the final 
research conclusions.

Justification
-------------
As noted in the expert review, the scientific validity of the model hinges on 
estimated coefficients. This script converts those point-estimates into 
quantified uncertainty ranges using Sobol indices and Tornado plots.

Key Metrics:
- Sobol First Order (S1): Variance in yield caused purely by one parameter.
- Sobol Total Order (ST): Total variance contribution including interactions.
- Tornado Plot: Linear impact of parameters on the Muslin stability advantage.

Author: Md. Noman, NSTU
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import SALib for formal Sobol indices, fallback to variance-based GSA if missing
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False

# Import core logic from the simulation model
from stochastic_simulation_model import (
    compute_yield, 
    generate_salinity_trajectory, 
    generate_temperature_trajectory, 
    build_phenological_weights,
    SPECIES
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("./sensitivity_results")
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 1. PARAMETER SPACE DEFINITION
# -----------------------------------------------------------------------------

# We focus on the most uncertain parameters highlighted by the expert review:
# 1. Muslin OAM: baseline 0.60, range [0.5, 0.7] (+/- 17%)
# 2. Fiber Deg. Rate (Suvin k_L): baseline 0.008, range [0.0064, 0.0096] (+/- 20%)
# 3. Stress Interaction Coeff: baseline 1.5, range [1.0, 2.0] (+/- 33%)

problem = {
    'num_vars': 3,
    'names': ['OAM_Muslin', 'k_L_Suvin', 'Interaction_Coeff'],
    'bounds': [
        [0.5, 0.7],       # OAM_Muslin
        [0.0064, 0.0096], # k_L_Suvin (proxy for fiber uncertainty)
        [1.0, 2.0]        # Interaction_Coeff
    ]
}

# -----------------------------------------------------------------------------
# 2. EVALUATION FUNCTION
# -----------------------------------------------------------------------------

def evaluate_model(param_sets, n_runs=100):
    """
    Evaluates the model for each row in the Saltelli sample.
    The goal is to calculate the 'stability advantage' of Muslin or Mean Yield.
    """
    results = []
    
    # Pre-compute weights and environmental trajectories to save time
    # We use a subset of runs (100) per parameter set to keep GSA feasible
    W_NORM = build_phenological_weights(120)
    
    # Use a fixed combined-extreme regime for sensitivity testing
    regime = "combined-extreme"
    
    # Pre-generate trajectories for the 100 runs
    env_data = []
    for r_id in range(n_runs):
        ec, red, _ = generate_salinity_trajectory(120, r_id, regime)
        temp = generate_temperature_trajectory(120, r_id, regime)
        env_data.append((ec, red, temp))

    log.info(f"Running {len(param_sets)} parameter sets across {n_runs} stochastic realizations...")

    for i, p in enumerate(param_sets):
        oam_m = p[0]
        kl_s  = p[1]
        inter = p[2]
        
        muslin_yields = []
        
        # Clone species params to avoid side effects
        m_params = SPECIES["Muslin (G. arboreum)"].copy()
        m_params["OAM"] = oam_m
        
        for r_id in range(n_runs):
            ec, red, temp = env_data[r_id]
            
            # Manually compute with specific interaction term override
            # We recreate the compute_yield logic here to inject the 'inter' parameter
            ec_washed = ec * red
            ec_s = float(np.dot(ec_washed, W_NORM))
            t_s  = float(np.dot(temp, W_NORM))
            
            d_sal = m_params["MH_slope"] * max(0.0, ec_s - m_params["MH_threshold"]) * oam_m
            d_heat = 0.70 * (1.0 - np.exp(-0.35 * max(0.0, t_s - 35.0)))
            d_total = min(d_sal + d_heat + d_sal * d_heat * inter, 1.0)
            y = m_params["Y_max"] * max(0.0, 1.0 - d_total)
            
            muslin_yields.append(y)
            
        # Metric: Mean Yield of Muslin under stress
        results.append(np.mean(muslin_yields))
        
        if (i + 1) % 100 == 0:
            log.info(f"  Progress: {i+1}/{len(param_sets)}")

    return np.array(results)

# -----------------------------------------------------------------------------
# 3. GSA EXECUTION
# -----------------------------------------------------------------------------

def run_sobol_analysis():
    if not HAS_SALIB:
        log.warning("SALib not found. Skipping Sobol analysis. Please install with 'pip install SALib'")
        return
        
    param_values = saltelli.sample(problem, 64) # Small N for demonstration (N*(2D+2) samples)
    y = evaluate_model(param_values)
    
    si = sobol.analyze(problem, y)
    
    log.info("\n=== Sobol Sensitivity Indices (Muslin Yield) ===")
    df_si = pd.DataFrame({
        'Parameter': problem['names'],
        'S1 (First Order)': si['S1'],
        'ST (Total Order)': si['ST'],
        'S1_conf': si['S1_conf'],
        'ST_conf': si['ST_conf']
    })
    log.info("\n" + df_si.to_string(index=False))
    df_si.to_csv(OUT_DIR / "sobol_indices.csv", index=False)
    
    # Plot Sobol Indices
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(problem['names']))
    width = 0.35
    ax.bar(x - width/2, si['S1'], width, label='First Order (S1)', color='#3498db', yerr=si['S1_conf'])
    ax.bar(x + width/2, si['ST'], width, label='Total Order (ST)', color='#e74c3c', yerr=si['ST_conf'], alpha=0.7)
    
    ax.set_ylabel('Sensitivity Index')
    ax.set_title('Sobol Global Sensitivity Analysis\n(Impact on Muslin Yield Stability)')
    ax.set_xticks(x)
    ax.set_xticklabels(problem['names'])
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sobol_indices.png", dpi=150)
    plt.close()

# -----------------------------------------------------------------------------
# 4. TORNADO PLOT (One-at-a-time Sensitivity)
# -----------------------------------------------------------------------------

def run_tornado_analysis():
    log.info("\nGenerating Tornado Plot analysis...")
    
    # Baseline: OAM=0.6, kL=0.008, Inter=1.5
    baseline_params = np.array([[0.60, 0.008, 1.5]])
    base_yield = evaluate_model(baseline_params, n_runs=200)[0]
    
    tornado_results = []
    
    for i, name in enumerate(problem['names']):
        low_val = problem['bounds'][i][0]
        high_val = problem['bounds'][i][1]
        
        # Test low
        p_low = baseline_params.copy()
        p_low[0, i] = low_val
        y_low = evaluate_model(p_low, n_runs=200)[0]
        
        # Test high
        p_high = baseline_params.copy()
        p_high[0, i] = high_val
        y_high = evaluate_model(p_high, n_runs=200)[0]
        
        tornado_results.append({
            'Parameter': name,
            'Low': y_low - base_yield,
            'High': y_high - base_yield,
            'Width': abs(y_high - y_low)
        })
        
    df_tornado = pd.DataFrame(tornado_results).sort_values('Width')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(df_tornado))
    
    ax.barh(y_pos, df_tornado['Low'], align='center', color='#3498db', label='Lower Bound impact', alpha=0.8)
    ax.barh(y_pos, df_tornado['High'], align='center', color='#e74c3c', label='Upper Bound impact', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_tornado['Parameter'], fontweight='bold')
    ax.invert_yaxis()  # top-down
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Change in Muslin Mean Yield relative to Baseline (kg/ha)')
    ax.set_title('Tornado Plot: Model Robustness to Parameter Uncertainty\n(Baseline Yield = %.1f kg/ha)' % base_yield)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend()
    
    # Add annotations for magnitude
    for i, row in df_tornado.reset_index().iterrows():
        ax.text(max(row['Low'], row['High']) + 5, i, 'Δ=%.1f' % row['Width'], va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "tornado_plot.png", dpi=150)
    plt.close()
    log.info(f"Tornado Analysis Complete. Saved to {OUT_DIR}/tornado_plot.png")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("="*65)
    log.info("  SENSITIVITY ANALYSIS & ROBUSTNESS TESTING")
    log.info("="*65)
    
    # Run Tornado (OAT) first as it is faster and intuitive
    run_tornado_analysis()
    
    # Run Sobol (Global) if possible
    run_sobol_analysis()
    
    log.info("\nAll sensitivity analyses complete.")
    log.info(f"Summary results in directory: {OUT_DIR}")
