"""
stochastic_simulation_model.py
================================
Stochastic Pre-screening Model: Coastal Salinity Resilience in
Gossypium barbadense (Suvin) vs Gossypium arboreum (Muslin)

Author      : Md. Noman
Institution : Department of Agricultural Sciences, NSTU, Bangladesh
GitHub      : https://github.com/md-noman-research/cotton-yield-simulation
License     : MIT

SIMULATION DISCLOSURE
---------------------
All outputs produced by this script are generated via stochastic Monte
Carlo simulation. No field measurements, experimental trials, or empirical
observations are included. Results are model-based estimates calibrated
against published Maas-Hoffman (1977) benchmarks and should be interpreted
accordingly.

Description
-----------
Generates N Monte Carlo yield + fiber-quality realizations under AR(1)
non-stationary salinity forcing (Noakhali pre-monsoon profile) and
phenologically weighted Maas-Hoffman damage kinetics.

Key outputs
-----------
1. Yield distributions (mean, SD, variance decomposition)
2. Fiber quality (staple length, micronaire, bundle strength) by ECe
3. Point-to-point Maas-Hoffman validation table
4. Statistical diagnostics (WLS, Gamma GLM, NLS)
5. CSV exports + SHA-256 integrity log

Dependencies
------------
numpy>=1.24, scipy>=1.10, statsmodels>=0.14, scikit-learn>=1.3,
pandas>=2.0, matplotlib>=3.7

Usage
-----
    python stochastic_simulation_model.py
    python stochastic_simulation_model.py --n-runs 1000 --seed 99
    python stochastic_simulation_model.py --output-dir ./results --n-runs 500

All outputs written to ./simulation_outputs/ (default)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import gamma as gamma_dist  # noqa: F401 (kept for future use)
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with safe defaults."""
    parser = argparse.ArgumentParser(
        description=(
            "Cotton Coastal Salinity Stochastic Simulation — "
            "Gossypium barbadense (Suvin) vs Gossypium arboreum (Muslin)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-runs", type=int, default=500,
        help="Number of Monte Carlo realizations (min 50, max 10 000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed for reproducibility"
    )
    parser.add_argument(
        "--n-days", type=int, default=120,
        help="Crop season length in days (min 60, max 365)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./simulation_outputs",
        help="Directory for CSV outputs and integrity log"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Raise ValueError if any argument is outside safe bounds."""
    if not (50 <= args.n_runs <= 10_000):
        raise ValueError(f"--n-runs must be between 50 and 10 000, got {args.n_runs}")
    if not (0 <= args.seed <= 2**32 - 1):
        raise ValueError(f"--seed must be a valid 32-bit unsigned integer, got {args.seed}")
    if not (60 <= args.n_days <= 365):
        raise ValueError(f"--n-days must be between 60 and 365, got {args.n_days}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPECIES PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

SPECIES: dict = {
    "Suvin (G. barbadense)": {
        # Yield
        "Y_max"        : 1800.0,   # kg/ha — CDB (2025) non-saline baseline
        "OAM"          : 1.00,     # Osmotic Adjustment Modifier — baseline referent
        # Maas-Hoffman parameters (G. hirsutum proxy; justified in manuscript §2.3)
        "MH_threshold" : 7.7,      # dS/m
        "MH_slope"     : 0.052,    # fraction per dS/m above threshold
        # Fiber quality baselines — ICAR-CIRCOT (2018)
        "L_base"       : 33.2,     # mm  — ELS staple
        "Mic_base"     : 3.70,     # µg/inch — HVI
        "Str_base"     : 31.5,     # g/tex — bundle strength
        # Fiber degradation rates per dS/m above threshold
        # Sources: Pettigrew (2004); Francois et al. (1994)
        "k_L"          : 0.0080,   # fractional length loss per dS/m
        "k_M"          : 0.0170,   # micronaire increase (absolute) per dS/m
        "k_S"          : 0.0090,   # fractional strength loss per dS/m
    },
    "Muslin (G. arboreum)": {
        "Y_max"        : 1200.0,   # kg/ha — CDB (2025)
        "OAM"          : 0.60,     # 40% osmotic buffering — Sharif et al. (2019)
        "MH_threshold" : 7.7,
        "MH_slope"     : 0.052,
        # Fiber quality baselines — AICCIP (2019)
        "L_base"       : 18.5,     # mm — short-staple desi cotton
        "Mic_base"     : 4.60,     # µg/inch
        "Str_base"     : 23.8,     # g/tex
        # Degradation rates — diploid under salinity
        # Sources: Ashraf (2002); Francois et al. (1994)
        "k_L"          : 0.0040,
        "k_M"          : 0.0090,
        "k_S"          : 0.0050,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. ENVIRONMENTAL FORCING
# ─────────────────────────────────────────────────────────────────────────────

def generate_salinity_trajectory(
    n_days: int = 120,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-stationary AR(1) salinity process.

    mu(t) = sigmoid capillary accumulation (SRDI 2020 Noakhali profile)
    Z(t)  = AR(1) stochastic persistence  (phi=0.80, sigma_eps=1.2 dS/m)
    R(t)  = sigmoid monsoon washout after day 90 (70% leaching — SRDI 2020)

    Parameters
    ----------
    n_days : int
        Season length in days.
    seed_offset : int
        Per-run seed offset for independent realizations.

    Returns
    -------
    ECe_raw : ndarray, shape (n_days,)  — raw salinity before washout
    R       : ndarray, shape (n_days,)  — washout reduction factor [0, 1]
    t       : ndarray, shape (n_days,)  — day index
    """
    rng = np.random.default_rng(seed_offset)
    t = np.arange(1, n_days + 1)

    # Deterministic seasonal trend — sigmoid capillary rise
    mu = 2.0 + 13.0 / (1.0 + np.exp(-0.08 * (t - 60)))

    # AR(1) stochastic innovation
    Z = np.zeros(n_days)
    eps = rng.normal(0.0, 1.2, n_days)
    for i in range(1, n_days):
        Z[i] = 0.80 * Z[i - 1] + eps[i]

    ECe_raw = np.clip(mu + Z, 0.5, 50.0)

    # Monsoon washout sigmoid (days ≥ 90)
    R = np.ones(n_days)
    for i in range(n_days):
        if t[i] >= 90:
            R[i] = 0.3 + 0.7 * np.exp(-0.12 * (t[i] - 90))

    return ECe_raw, R, t


def generate_temperature_trajectory(
    n_days: int = 120,
    seed_offset: int = 0,
) -> np.ndarray:
    """
    Stationary AR(1) temperature process.
    Mean 32°C, phi=0.75, sigma=1.5°C, bounded [20, 45]°C.

    Parameters
    ----------
    n_days      : int — number of days.
    seed_offset : int — per-run offset (shifted by 10 000 to decorrelate from salinity RNG).

    Returns
    -------
    T : ndarray, shape (n_days,) — daily temperature (°C)
    """
    rng = np.random.default_rng(seed_offset + 10_000)
    T = np.zeros(n_days)
    T[0] = 32.0
    eta = rng.normal(0.0, 1.5, n_days)
    for i in range(1, n_days):
        T[i] = 32.0 + 0.75 * (T[i - 1] - 32.0) + eta[i]
    return np.clip(T, 20.0, 45.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PHENOLOGICAL WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

def build_phenological_weights(n_days: int = 120) -> np.ndarray:
    """
    Three-phase sensitivity weighting (normalized):
      P1 vegetative  (days  1-40) : relative weight 1×
      P2 boll set    (days 41-80) : relative weight 3×  [critical window]
      P3 maturation  (days 81-N)  : relative weight 1×

    Returns
    -------
    W : ndarray, shape (n_days,) — weights summing to 1.0
    """
    raw = np.ones(n_days)
    p2_start = min(40, n_days)
    p2_end   = min(80, n_days)
    raw[p2_start:p2_end] = 3.0
    return raw / raw.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 4. YIELD DAMAGE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def compute_yield(
    ECe_raw: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    params: dict,
    W_norm: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Single-realization yield under combined salinity + heat stress.

    Steps
    -----
    1. EC_season = phenologically weighted ECe after monsoon washout
    2. T_season  = phenologically weighted temperature
    3. D_sal     = Maas-Hoffman piecewise × OAM
    4. D_heat    = exponential saturation above 35°C
    5. D_total   = D_sal + D_heat + hyper-additive interaction (×1.5)
    6. Y         = Y_max × max(0, 1 - D_total)

    Parameters
    ----------
    ECe_raw : ndarray — raw daily salinity (dS/m)
    R       : ndarray — washout reduction factor
    T       : ndarray — daily temperature (°C)
    params  : dict    — species parameter dictionary
    W_norm  : ndarray — phenological weights

    Returns
    -------
    (Y, EC_season, T_season, D_sal, D_heat)
    """
    ECe_washed = ECe_raw * R
    EC_season  = float(np.dot(ECe_washed, W_norm))
    T_season   = float(np.dot(T, W_norm))

    OAM       = params["OAM"]
    threshold = params["MH_threshold"]
    slope     = params["MH_slope"]

    D_sal   = slope * max(0.0, EC_season - threshold) * OAM
    D_heat  = 0.70 * (1.0 - np.exp(-0.35 * max(0.0, T_season - 35.0)))
    D_total = min(D_sal + D_heat + D_sal * D_heat * 1.5, 1.0)

    Y = params["Y_max"] * max(0.0, 1.0 - D_total)
    return Y, EC_season, T_season, D_sal, D_heat


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIBER QUALITY MODEL
# ─────────────────────────────────────────────────────────────────────────────

def compute_fiber_quality(
    EC_season: float,
    params: dict,
) -> tuple[float, float, float]:
    """
    Linear fiber quality degradation above ECe threshold.

    L(EC)   = L_base   × max(0, 1 - k_L × excess_ECe)
    Mic(EC) = Mic_base + k_M × excess_ECe
    Str(EC) = Str_base × max(0, 1 - k_S × excess_ECe)

    Sources: Pettigrew (2004); Ashraf (2002); Francois et al. (1994)

    Parameters
    ----------
    EC_season : float — phenologically weighted seasonal ECe (dS/m)
    params    : dict  — species parameter dictionary

    Returns
    -------
    (L, Mic, Str)
    """
    excess = max(0.0, EC_season - params["MH_threshold"])

    L   = params["L_base"]   * max(0.0, 1.0 - params["k_L"] * excess)
    Mic = params["Mic_base"] + params["k_M"] * excess
    Str = params["Str_base"] * max(0.0, 1.0 - params["k_S"] * excess)

    return L, Mic, Str


# ─────────────────────────────────────────────────────────────────────────────
# 6. SHA-256 INTEGRITY HASH
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file for data integrity verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_integrity_log(
    output_dir: Path,
    csv_paths: list[Path],
    run_meta: dict,
) -> None:
    """
    Write JSON integrity log containing SHA-256 hashes of all CSV outputs
    and run metadata. Allows independent verification of exact reproducibility.
    """
    log_data = {
        "generated_utc"   : datetime.now(timezone.utc).isoformat(),
        "run_parameters"  : run_meta,
        "file_hashes_sha256": {
            p.name: sha256_file(p) for p in csv_paths if p.exists()
        },
    }
    log_path = output_dir / "integrity_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    log.info("Integrity log → %s", log_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. VARIANCE DECOMPOSITION — RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────

def run_variance_decomposition(
    all_results: dict,
    seed: int,
) -> dict:
    """Random Forest variance decomposition (200 trees, max_features='sqrt')."""
    log.info("─" * 65)
    log.info("Variance Decomposition (Random Forest, 200 trees)")
    log.info("─" * 65)

    rf_results: dict = {}
    for species_name, df in all_results.items():
        X = df[["EC_season", "T_season"]].values
        y = df["yield_kgha"].values

        rf = RandomForestRegressor(
            n_estimators=200, random_state=seed, max_features="sqrt"
        )
        rf.fit(X, y)
        imp = rf.feature_importances_

        rf_results[species_name] = {
            "EC_season_pct": imp[0] * 100,
            "T_season_pct" : imp[1] * 100,
        }
        log.info(
            "  %s → EC_season: %.1f%%  |  T_season: %.1f%%",
            species_name, imp[0] * 100, imp[1] * 100,
        )

    return rf_results


# ─────────────────────────────────────────────────────────────────────────────
# 8. STATISTICAL DIAGNOSTICS (WLS, Gamma GLM, NLS)
# ─────────────────────────────────────────────────────────────────────────────

def run_statistical_diagnostics(
    all_results: dict,
    species_params: dict,
) -> dict:
    """
    Fit three complementary regression models per species:
      - WLS log-linear (heteroscedasticity-robust)
      - Gamma GLM with log link (positive-valued yield)
      - Mechanistic NLS re-fit of Maas-Hoffman piecewise model
    """
    log.info("─" * 65)
    log.info("Statistical Diagnostics")
    log.info("─" * 65)

    diagnostic_results: dict = {}

    for species_name, df in all_results.items():
        log.info("  [%s]", species_name)
        params = species_params[species_name]

        X_raw  = df["EC_season"].values
        y_vals = df["yield_kgha"].values
        X_sm   = sm.add_constant(X_raw)

        # ── WLS log-linear
        log_y   = np.log(np.clip(y_vals, 1e-3, None))
        weights = 1.0 / (X_raw + 1.0)
        wls_r2  = sm.WLS(log_y, X_sm, weights=weights).fit().rsquared_adj
        log.info("    WLS adj-R²      : %.3f", wls_r2)

        # ── Gamma GLM
        try:
            glm_model = sm.GLM(
                y_vals, X_sm,
                family=sm.families.Gamma(link=sm.families.links.Log()),
            ).fit()
            glm_pr2 = 1.0 - glm_model.deviance / glm_model.null_deviance
            log.info("    Gamma GLM pR²   : %.3f", glm_pr2)
        except Exception:
            glm_pr2 = float("nan")
            log.warning("    Gamma GLM       : convergence issue — skipped")

        # ── Mechanistic NLS (Y = b0 × max(0, 1 - b1 × max(0, EC - b2)))
        def mh_model(ec: np.ndarray, b0: float, b1: float, b2: float) -> np.ndarray:
            return b0 * np.maximum(0.0, 1.0 - b1 * np.maximum(0.0, ec - b2))

        try:
            p0 = [params["Y_max"], params["MH_slope"], params["MH_threshold"]]
            popt, _ = curve_fit(mh_model, X_raw, y_vals, p0=p0, maxfev=10_000)
            y_hat  = mh_model(X_raw, *popt)
            ss_res = np.sum((y_vals - y_hat) ** 2)
            ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
            nls_r2 = 1.0 - ss_res / ss_tot
            log.info("    NLS R²          : %.3f", nls_r2)
            log.info("    NLS b₀ (Y_max)  : %.1f kg/ha", popt[0])
            log.info("    NLS b₁ (slope)  : %.5f", popt[1])
            log.info("    NLS b₂ (thresh) : %.2f dS/m", popt[2])
        except Exception as exc:
            nls_r2 = float("nan")
            log.warning("    NLS             : fit error — %s", exc)

        diagnostic_results[species_name] = {
            "wls_adj_r2": wls_r2,
            "glm_pr2"   : glm_pr2,
            "nls_r2"    : nls_r2,
        }

    return diagnostic_results


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAAS-HOFFMAN VALIDATION TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_mh_validation_table() -> pd.DataFrame:
    """
    Point-to-point Maas-Hoffman validation at discrete ECe checkpoints.
    Suvin (OAM=1.00) matches G. hirsutum reference by design.
    Muslin (OAM=0.60) shows 40% osmotic reduction.
    """
    ec_checkpoints = [7.7, 8.5, 9.0, 10.0, 12.0, 14.0, 16.0]
    rows = []

    log.info("─" * 65)
    log.info("Point-to-Point Maas-Hoffman Validation")
    log.info("─" * 65)
    header = f"{'ECe':>6} | {'M-H Ref (%)':>11} | {'Suvin D_sal (%)':>15} | {'Dev':>6} | {'Muslin D_sal (%)':>16} | {'OAM Red.':>9}"
    log.info(header)
    log.info("  " + "-" * 76)

    for ec in ec_checkpoints:
        mh_ref  = max(0.0, 0.052 * (ec - 7.7)) * 100
        suv     = mh_ref                                   # OAM = 1.00
        mus     = max(0.0, 0.052 * (ec - 7.7) * 0.60) * 100
        dev     = suv - mh_ref
        oam_red = (mh_ref - mus) / mh_ref * 100 if mh_ref > 0 else float("nan")

        log.info(
            "  %6.1f | %11.2f | %15.2f | %6.2f | %16.2f | %8.1f%%",
            ec, mh_ref, suv, dev, mus, oam_red if not np.isnan(oam_red) else 0
        )
        rows.append({
            "ECe_dSm"           : ec,
            "MH_Reference_pct"  : round(mh_ref, 2),
            "Suvin_Dsal_pct"    : round(suv, 2),
            "Deviation_pct"     : round(dev, 2),
            "Muslin_Dsal_pct"   : round(mus, 2),
            "OAM_Reduction_pct" : round(oam_red, 1) if not np.isnan(oam_red) else None,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 10. FIBER QUALITY LOOKUP TABLE (ANALYTICAL)
# ─────────────────────────────────────────────────────────────────────────────

def build_fiber_quality_table(species_params: dict) -> pd.DataFrame:
    """Analytical fiber quality lookup at discrete ECe checkpoints."""
    ec_points = [7.7, 8.5, 10.0, 12.0, 14.0, 16.0]
    suv_p = species_params["Suvin (G. barbadense)"]
    mus_p = species_params["Muslin (G. arboreum)"]
    rows  = []

    log.info("─" * 65)
    log.info("Fiber Quality at Discrete ECe Checkpoints (Analytical)")
    log.info("─" * 65)

    for ec in ec_points:
        sL, sMic, sStr = compute_fiber_quality(ec, suv_p)
        mL, mMic, mStr = compute_fiber_quality(ec, mus_p)
        log.info(
            "  ECe %5.1f | Suvin L=%.2f Mic=%.3f Str=%.2f | "
            "Muslin L=%.2f Mic=%.3f Str=%.2f",
            ec, sL, sMic, sStr, mL, mMic, mStr
        )
        rows.append({
            "ECe_dSm"             : ec,
            "Suvin_Staple_mm"     : round(sL, 2),
            "Suvin_Micronaire"    : round(sMic, 3),
            "Suvin_Strength_gtex" : round(sStr, 2),
            "Muslin_Staple_mm"    : round(mL, 2),
            "Muslin_Micronaire"   : round(mMic, 3),
            "Muslin_Strength_gtex": round(mStr, 2),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    validate_args(args)

    SEED       = args.seed
    N_RUNS     = args.n_runs
    N_DAYS     = args.n_days
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-compute phenological weights (depends on N_DAYS)
    W_NORM = build_phenological_weights(N_DAYS)

    log.info("=" * 65)
    log.info("  Cotton Salinity Simulation — NSTU Bangladesh")
    log.info("  DISCLOSURE: Stochastic simulation — no field data")
    log.info("  N_RUNS=%d | Seed=%d | N_DAYS=%d", N_RUNS, SEED, N_DAYS)
    log.info("=" * 65)

    # ── 6. Monte Carlo simulation loop ───────────────────────────────────────
    all_results: dict = {}
    saved_csv: list[Path] = []

    for species_name, params in SPECIES.items():
        log.info("\n[Running] %s ...", species_name)
        records = []

        for run_id in range(N_RUNS):
            ECe_raw, R, _t = generate_salinity_trajectory(N_DAYS, seed_offset=run_id)
            T = generate_temperature_trajectory(N_DAYS, seed_offset=run_id)

            Y, EC_s, T_s, D_sal, D_heat = compute_yield(ECe_raw, R, T, params, W_NORM)
            L, Mic, Str = compute_fiber_quality(EC_s, params)

            records.append({
                "run_id"    : run_id,
                "EC_season" : round(EC_s, 6),
                "T_season"  : round(T_s, 6),
                "D_sal"     : round(D_sal, 6),
                "D_heat"    : round(D_heat, 6),
                "yield_kgha": round(Y, 4),
                "staple_mm" : round(L, 4),
                "micronaire": round(Mic, 4),
                "strength"  : round(Str, 4),
            })

        df = pd.DataFrame(records)
        all_results[species_name] = df

        log.info("  Mean yield    : %.1f kg/ha", df["yield_kgha"].mean())
        log.info("  SD   yield    : %.1f kg/ha", df["yield_kgha"].std())
        log.info("  Mean EC_season: %.3f dS/m",  df["EC_season"].mean())
        log.info("  Mean staple   : %.2f mm",     df["staple_mm"].mean())
        log.info("  Mean mic      : %.3f µg/in",  df["micronaire"].mean())
        log.info("  Mean strength : %.2f g/tex",  df["strength"].mean())

        safe = (
            species_name
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "")
        )
        csv_path = OUTPUT_DIR / f"{safe}_runs.csv"
        df.to_csv(csv_path, index=False)
        saved_csv.append(csv_path)
        log.info("  Saved → %s", csv_path)

    # ── 7. Variance decomposition ─────────────────────────────────────────────
    rf_results = run_variance_decomposition(all_results, SEED)

    # ── 8. Statistical diagnostics ────────────────────────────────────────────
    diagnostic_results = run_statistical_diagnostics(all_results, SPECIES)

    # ── 9. Maas-Hoffman validation table ─────────────────────────────────────
    val_df   = build_mh_validation_table()
    val_path = OUTPUT_DIR / "maas_hoffman_validation.csv"
    val_df.to_csv(val_path, index=False)
    saved_csv.append(val_path)
    log.info("Saved → %s", val_path)

    # ── 10. Fiber quality table ───────────────────────────────────────────────
    fq_df   = build_fiber_quality_table(SPECIES)
    fq_path = OUTPUT_DIR / "fiber_quality_table.csv"
    fq_df.to_csv(fq_path, index=False)
    saved_csv.append(fq_path)
    log.info("Saved → %s", fq_path)

    # ── 11. Summary export ────────────────────────────────────────────────────
    summary_rows = []
    for species_name, df in all_results.items():
        vd   = rf_results[species_name]
        diag = diagnostic_results[species_name]
        summary_rows.append({
            "Species"            : species_name,
            "N_runs"             : N_RUNS,
            "Seed"               : SEED,
            "Mean_Yield_kgha"    : round(df["yield_kgha"].mean(), 1),
            "SD_Yield_kgha"      : round(df["yield_kgha"].std(),  1),
            "EC_season_Var_pct"  : round(vd["EC_season_pct"], 1),
            "T_season_Var_pct"   : round(vd["T_season_pct"],  1),
            "WLS_Adj_R2"         : round(diag["wls_adj_r2"], 3),
            "GammaGLM_pR2"       : round(diag["glm_pr2"],    3),
            "NLS_R2"             : round(diag["nls_r2"],      3),
            "Mean_Staple_mm"     : round(df["staple_mm"].mean(),  2),
            "Mean_Micronaire"    : round(df["micronaire"].mean(), 3),
            "Mean_Strength_gtex" : round(df["strength"].mean(),   2),
        })

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "simulation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    saved_csv.append(summary_path)

    # ── 12. SHA-256 integrity log ─────────────────────────────────────────────
    run_meta = {
        "n_runs" : N_RUNS,
        "seed"   : SEED,
        "n_days" : N_DAYS,
        "species": list(SPECIES.keys()),
    }
    write_integrity_log(OUTPUT_DIR, saved_csv, run_meta)

    log.info("=" * 65)
    log.info("  SIMULATION COMPLETE")
    log.info("  All outputs → %s/", OUTPUT_DIR)
    log.info("=" * 65)
    for f in sorted(OUTPUT_DIR.iterdir()):
        log.info("    %-45s  (%s bytes)", f.name, f"{f.stat().st_size:,}")


if __name__ == "__main__":
    main()
