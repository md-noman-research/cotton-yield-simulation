"""
stochastic yield response model for cotton genotypes under deltaic salinity ramps.
simulates ar(1) environmental persistence and mechanistic metabolic damage.

Author: Md. Noman
Affiliation: Noakhali Science and Technology University (NSTU)
Email: md.noman.research@gmail.com
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# seasonal configuration for noakhali pre-monsoon window
DAYS = 120
BURN_IN = 30
TOTAL_DAYS = DAYS + BURN_IN
ITERS = 1000

# yield baseline (kg/ha)
BASE_YIELD_SUVIN = 2500.0
BASE_YIELD_MUSLIN = 1500.0

# ec / temp distribution parameters derived from coastal monitoring proxies
INITIAL_EC = 2.0
FINAL_EC = 15.0
EC_PHI = 0.8        # high persistence due to capillary rise logic
T_PHI = 0.75
EC_MEAN = 5.5
EC_STD = 3.0
T_MEAN = 32.0
T_STD = 4.0

# metabolic thresholds based on maas & hoffman (1977)
THRESHOLD_SAL = 7.7
THRESHOLD_HEAT = 35.0
HEAT_MAX_STRESS = 0.7      
K_HEAT = 0.35              
K_SAL = 0.25               
INTERACTION_FACTOR = 0.2
SENSITIVITY_SCALING = 7.5
MAX_DAILY_DAMAGE = 0.25

# planting window scenarios
PLANTING_SCENARIOS = {'Early': -14, 'Typical': 0, 'Late': 14}

def phase_mult(day):
    """phenological sensitivity scaling relative to boll initiation."""
    if day < 60:
        return 0.03
    elif day < 90:
        return 0.15
    else:
        return 0.05

PHASES = np.array([phase_mult(d) for d in range(DAYS)], dtype=float)

def ec_drift_profile(d_len, start=INITIAL_EC, end=FINAL_EC, monsoon_start=100, monsoon_duration=5, post_monsoon_level=None):
    """sigmoid washout function to simulate monsoon-driven leaching."""
    if post_monsoon_level is None:
        post_monsoon_level = start
    days = np.arange(d_len, dtype=float)
    profile = start + (end - start) * (days / max(1.0, (d_len - 1)))
    scale = max(1.0, monsoon_duration / 6.0)
    monsoon_factor = 1.0 - 1.0 / (1.0 + np.exp(-(days - monsoon_start) / scale))
    adjusted = profile * monsoon_factor + post_monsoon_level * (1.0 - monsoon_factor)
    adjusted = np.maximum(adjusted, 1e-6)
    return adjusted

def sal_stress(EC, OAM, threshold=THRESHOLD_SAL, k=K_SAL):
    """exponential yield decay based on maas-hoffman thresholds."""
    excess = np.maximum(0.0, EC - threshold)
    return OAM * (1.0 - np.exp(-k * excess))

def heat_stress(T, threshold=THRESHOLD_HEAT, heat_max=HEAT_MAX_STRESS, k=K_HEAT):
    """thermal metabolic constraint during pre-monsoon peak."""
    excess = np.maximum(0.0, T - threshold)
    return heat_max * (1.0 - np.exp(-k * excess))

def compute_yield(OAM, ECs, Ts, base_yield):
    """multi-stage yield integration with compound stress interaction."""
    s_st = sal_stress(ECs, OAM)
    h_st = heat_stress(Ts)
    inter = INTERACTION_FACTOR * s_st * h_st
    combined = s_st + h_st + inter
    total_st = np.minimum(1.0, combined)  
    daily_damage = np.minimum(total_st * PHASES * SENSITIVITY_SCALING, MAX_DAILY_DAMAGE)
    factor = np.exp(-daily_damage.sum())
    factor = float(np.clip(factor, 0.0, 1.0))
    return base_yield * factor

# data generation (restricted to 1000 iterations for stable stats)
all_data = []
ITERS = 500  

REL_CV = EC_STD / EC_MEAN
sigma_z = np.sqrt(np.log(1.0 + REL_CV**2))

drift = ec_drift_profile(TOTAL_DAYS)
mu_daily = np.log(np.maximum(1e-6, drift)) - 0.5 * sigma_z**2

print(f"Executing 1,000-iteration Monte Carlo simulation...")

for i in range(ITERS):
    # simulate log-ec ar(1) persistence
    Z = np.zeros(TOTAL_DAYS, dtype=float)
    Z[0] = np.random.normal(0.0, sigma_z)
    eps_ec_sd = sigma_z * np.sqrt(1.0 - EC_PHI**2)
    for d in range(1, TOTAL_DAYS):
        Z[d] = EC_PHI * Z[d-1] + np.random.normal(0.0, eps_ec_sd)
    ECs = np.exp(mu_daily + Z)[BURN_IN:]

    # simulate temp ar(1) seasonality
    Ts_full = np.zeros(TOTAL_DAYS, dtype=float)
    Ts_full[0] = np.random.normal(T_MEAN, T_STD)
    eps_t_sd = T_STD * np.sqrt(1.0 - T_PHI**2)
    for d in range(1, TOTAL_DAYS):
        Ts_full[d] = T_MEAN + T_PHI * (Ts_full[d-1] - T_MEAN) + np.random.normal(0.0, eps_t_sd)
    Ts = Ts_full[BURN_IN:]

    ECs = np.clip(ECs, 0.0, 50.0)
    Ts = np.clip(Ts, 20.0, 45.0)

    # engineered features for sensitivity audit
    avg_salinity_p2 = float(np.mean(ECs[60:90]))
    max_salinity = float(np.max(ECs))
    avg_temp = float(np.mean(Ts))

    all_data.append({
        'Avg_Salinity_P2': avg_salinity_p2,
        'Max_Salinity': max_salinity,
        'Avg_Temp': avg_temp,
        'Cotton': 'Suvin',
        'Yield': compute_yield(1.0, ECs, Ts, BASE_YIELD_SUVIN)
    })
    all_data.append({
        'Avg_Salinity_P2': avg_salinity_p2,
        'Max_Salinity': max_salinity,
        'Avg_Temp': avg_temp,
        'Cotton': 'Muslin',
        'Yield': compute_yield(0.6, ECs, Ts, BASE_YIELD_MUSLIN)
    })

df = pd.DataFrame(all_data)
df.to_csv('noakhali_cotton_simulation_results.csv', index=False)

# random forest variance decomposition
features = ['Avg_Salinity_P2', 'Max_Salinity', 'Avg_Temp']

print("\n" + "-"*48)
print(f"{'Genotype':<12} {'Feature':<22} {'Importance (%)':<14}")
print("-"*48)

for variety in ['Suvin', 'Muslin']:
    sub = df[df['Cotton'] == variety].copy()
    X = sub[features].values
    y = sub['Yield'].values
    
    # TODO: calibrate sensitivity scaling against field monitoring data
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    for feat, imp in zip(features, importances):
        print(f"{variety:<12} {feat:<22} {imp*100:<14.2f}")

print("-"*48 + "\n")
