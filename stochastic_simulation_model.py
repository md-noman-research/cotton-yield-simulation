import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# Source Code: Stochastic Monte Carlo Simulation
# Associated Manuscript: Coastal Salinity Resilience in Gossypium barbadense 
# (Suvin) versus Gossypium arboreum (Muslin)
# ==============================================================================

# Seed strictly fixed to 42 for reproducibility as claimed in the paper
np.random.seed(42)

# --- 1. Simulation Constants ---
N_ITERATIONS = 500
DAYS = 120

# Genotype Parameters
Y_MAX_SUVIN = 1800.0  # kg/ha
Y_MAX_MUSLIN = 1200.0 # kg/ha
OAM_SUVIN = 1.00      # 100% stress transmission
OAM_MUSLIN = 0.60     # 40% intrinsic buffering (Sharif et al. 2019)

# --- 2. Phenological Weights Mapping (Step 1) ---
# Phase 1: Veg (Days 0-59) -> Weight 1x
# Phase 2: Boll Initiation (Days 60-89) -> Weight 3x
# Phase 3: Maturation (Days 90-119) -> Weight 1x
raw_weights = np.zeros(DAYS)
raw_weights[0:60] = 1.0
raw_weights[60:90] = 3.0
raw_weights[90:120] = 1.0
W_norm = raw_weights / np.sum(raw_weights)

# --- Initialization Arrays ---
yields_suvin = np.zeros(N_ITERATIONS)
yields_muslin = np.zeros(N_ITERATIONS)
ec_season_arr = np.zeros(N_ITERATIONS)
t_season_arr = np.zeros(N_ITERATIONS)

print("Initializing 500 Monte Carlo Iterations...")

# --- 3. Monte Carlo Loop ---
for i in range(N_ITERATIONS):
    # Generative arrays for this iteration
    ECe = np.zeros(DAYS)
    T = np.zeros(DAYS)
    R = np.ones(DAYS)
    Z = np.zeros(DAYS)
    
    # Initial states
    Z[0] = np.random.normal(0, 1.2)
    T[0] = 32.0 + np.random.normal(0, 1.5)
    
    for t in range(DAYS):
        # Env Process 1: Salinity (Step 2.1.1)
        mu_t = 2.0 + 13.0 / (1.0 + np.exp(-0.08 * (t - 60)))
        if t > 0:
            Z[t] = 0.80 * Z[t-1] + np.random.normal(0, 1.2)
        
        ECe[t] = np.clip(mu_t + Z[t], 0.5, 50.0)
        
        # Washout Function (R)
        if t >= 90:
            R[t] = 0.3 + 0.7 * np.exp(-0.12 * (t - 90))
            
        # Env Process 2: Temperature (Step 2.1.2)
        if t > 0:
            T[t] = 32.0 + 0.75 * (T[t-1] - 32.0) + np.random.normal(0, 1.5)
        T[t] = np.clip(T[t], 20.0, 45.0)

    # --- 4. Integrated Yield Calculation (Step 2.2) ---
    # Step 2: Weighted Seasonal Salinity Exposure
    EC_season = np.sum(ECe * R * W_norm)
    ec_season_arr[i] = EC_season
    
    # Step 3: Weighted Seasonal Thermal Exposure
    T_season = np.sum(T * W_norm)
    t_season_arr[i] = T_season
    
    # Base Damage Engines
    base_D_sal = 0.052 * max(0, EC_season - 7.7)  # Maas-Hoffman Base
    D_heat = 0.70 * (1.0 - np.exp(-0.35 * max(0, T_season - 35.0))) # Exp. Saturation
    
    # Calculate Suvin
    D_sal_suvin = base_D_sal * OAM_SUVIN
    D_total_suvin = D_sal_suvin + D_heat + (D_sal_suvin * D_heat * 1.5)
    yields_suvin[i] = Y_MAX_SUVIN * max(0.0, 1.0 - D_total_suvin)
    
    # Calculate Muslin
    D_sal_muslin = base_D_sal * OAM_MUSLIN
    D_total_muslin = D_sal_muslin + D_heat + (D_sal_muslin * D_heat * 1.5)
    yields_muslin[i] = Y_MAX_MUSLIN * max(0.0, 1.0 - D_total_muslin)

# --- 5. Output Verification & Random Forest (Section 3.1) ---
sd_suvin = np.std(yields_suvin)
sd_muslin = np.std(yields_muslin)

print("\n--- STATISTICAL VERIFICATION ---")
print(f"Suvin Standard Deviation:  {sd_suvin:.1f} kg/ha (Target: 96.5)")
print(f"Muslin Standard Deviation: {sd_muslin:.1f} kg/ha (Target: 70.8)")

# Machine Learning Verification (Random Forest decomp)
df = pd.DataFrame({'EC_season': ec_season_arr, 'T_season': t_season_arr})

rf_suvin = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
rf_suvin.fit(df, yields_suvin)
suvin_importances = rf_suvin.feature_importances_ * 100

rf_muslin = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
rf_muslin.fit(df, yields_muslin)
muslin_importances = rf_muslin.feature_importances_ * 100

print("\n--- VARIANCE DECOMPOSITION ---")
print(f"Suvin - Salinity Attribution: {suvin_importances[0]:.1f}% (Target: 60.2%)")
print(f"Suvin - Thermal Attribution:  {suvin_importances[1]:.1f}% (Target: 39.8%)")
print(f"Muslin - Salinity Attribution: {muslin_importances[0]:.1f}% (Target: 54.2%)")
print(f"Muslin - Thermal Attribution:  {muslin_importances[1]:.1f}% (Target: 45.8%)")

# --- 6. Visualization ---
plt.figure(figsize=(10, 6), dpi=300)
plt.hist(yields_suvin, bins=30, alpha=0.6, color='blue', label=f'G. barbadense (Suvin)\nSD = {sd_suvin:.1f}')
plt.hist(yields_muslin, bins=30, alpha=0.6, color='green', label=f'G. arboreum (Muslin)\nSD = {sd_muslin:.1f}')

# Means
plt.axvline(np.mean(yields_suvin), color='blue', linestyle='dashed', linewidth=2)
plt.axvline(np.mean(yields_muslin), color='green', linestyle='dashed', linewidth=2)

plt.title('Monte Carlo Simulation: Yield Volatility Asymmetry across Accreted Coastal Gradients', fontweight='bold')
plt.xlabel('Algorithmically Predicted Yield (kg/ha)')
plt.ylabel('Frequency (n=500)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('c:\\recharcs suvin\\volatility_asymmetry_plot.png')
print("\nSimulation plot saved to volatility_asymmetry_plot.png")
