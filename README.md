# Cotton Yield Simulation — Coastal Salinity Resilience

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Type-Stochastic_Simulation-orange)]()
[![Status](https://img.shields.io/badge/Status-Peer_Review_Ready-brightgreen)]()

> **⚠️ Simulation Disclosure:** All data produced by this repository are generated via stochastic Monte Carlo simulation. No field measurements, experimental trials, or empirical observations are included. All outputs must be interpreted as model-based estimates calibrated against published Maas–Hoffman benchmarks.

---

## Overview

This repository contains the complete stochastic pre-screening simulation codebase for the research paper:

**"Coastal Salinity Resilience of *Gossypium barbadense* (Suvin) vs *Gossypium arboreum* (Muslin): A Dual-Output Stochastic Simulation Under Non-Stationary AR(1) Salinity Forcing"**

- **Author:** Md. Noman, Department of Agricultural Sciences, NSTU, Bangladesh  
- **Institution:** Noakhali Science and Technology University (NSTU)  
- **Target Journal:** Under review

---

## Repository Structure

```
cotton-yield-simulation/
│
├── stochastic_simulation_model.py   # Core Monte Carlo engine (500 runs, AR(1))
├── generate_figures.py              # Publication-ready figure generation
├── build_docx.py                    # Markdown → DOCX manuscript converter
│
├── simulation_outputs/              # Auto-generated CSV outputs (git-ignored)
├── figures/                         # Auto-generated figures (git-ignored)
│
├── requirements.txt                 # Exact dependency versions
├── .gitignore                       # Excludes outputs, cache, env files
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## Scientific Model Summary

### Environmental Forcing
- **Salinity:** Non-stationary AR(1) process — sigmoid capillary accumulation (SRDI 2020 Noakhali profile), φ = 0.80, σ_ε = 1.2 dS/m, with monsoon washout sigmoid (70% reduction after day 90)
- **Temperature:** Stationary AR(1), mean 32°C, φ = 0.75, bounded [20, 45]°C

### Yield Damage Kinetics
- **Maas–Hoffman (1977):** Piecewise linear salinity damage, threshold 7.7 dS/m, slope 5.2%/(dS/m)
- **OAM (Osmotic Adjustment Modifier):** Suvin = 1.00 (baseline); Muslin = 0.60 (40% osmotic buffering, Sharif et al. 2019)
- **Phenological Weighting:** Boll-set phase (days 41–80) weighted 3× vegetative/maturation
- **Heat co-stress:** Exponential saturation above 35°C with hyper-additive interaction term (1.5×)

### Species Parameters

| Parameter | Suvin (*G. barbadense*) | Muslin (*G. arboreum*) |
|---|---|---|
| Y_max (kg/ha) | 1800 | 1200 |
| OAM | 1.00 | 0.60 |
| MH threshold (dS/m) | 7.7 | 7.7 |
| MH slope (%/dS/m) | 5.2 | 5.2 |
| Staple baseline (mm) | 33.2 | 18.5 |
| Micronaire baseline (µg/inch) | 3.70 | 4.60 |
| Strength baseline (g/tex) | 31.5 | 23.8 |

### Statistical Diagnostics
- WLS log-linear regression
- Gamma GLM (log link, pseudo-R²)
- Mechanistic NLS (Maas–Hoffman re-fit)
- Random Forest variance decomposition (200 trees)

---

## Installation

```bash
git clone https://github.com/md-noman-research/cotton-yield-simulation.git
cd cotton-yield-simulation
pip install -r requirements.txt
```

**Python 3.10+ required.**

---

## Usage

### Step 1 — Run the simulation engine
```bash
python stochastic_simulation_model.py
```
Outputs written to `./simulation_outputs/`:
- `Suvin_G_barbadense_runs.csv` — 500 Monte Carlo realizations (Suvin)
- `Muslin_G_arboreum_runs.csv` — 500 Monte Carlo realizations (Muslin)
- `maas_hoffman_validation.csv` — Point-to-point M-H benchmark table
- `fiber_quality_table.csv` — Analytical fiber quality at discrete ECe checkpoints
- `simulation_summary.csv` — Aggregated statistics + diagnostics

### Step 2 — Generate publication figures
```bash
python generate_figures.py
```
Figures written to `./figures/`:
- `Figure1_Yield_Distributions.png`
- `Figure2_ECseason_vs_Yield.png`
- `Figure3_MH_Validation.png`
- `Figure4_Fiber_Quality_Degradation.png`
- `Figure5_Variance_Decomposition.png`

### CLI Options (simulation engine)
```bash
python stochastic_simulation_model.py --n-runs 1000 --seed 99 --output-dir ./my_outputs
```

---

## Reproducibility

The simulation uses **NumPy's `default_rng` Generator** (PCG64 algorithm) with per-run seed offsets for full reproducibility:

```
Seed = 42 (default)
N_RUNS = 500
N_DAYS = 120
```

A **SHA-256 integrity hash** of all CSV outputs is logged at the end of each run. Any modification to parameters will produce a different, detectable hash.

---

## Key References

| Reference | Usage |
|---|---|
| Maas & Hoffman (1977) *Irrig. Sci.* 2:57–68 | Threshold-slope damage kinetics |
| Pettigrew (2004) *Agron. J.* 96:1062–1069 | Fiber quality degradation rates |
| Ashraf (2002) *Crit. Rev. Plant Sci.* 21:1–30 | G. arboreum salinity tolerance |
| Francois et al. (1994) *Agron. J.* 86:292–298 | Fiber quality under salt stress |
| Sharif et al. (2019) *J. Plant Growth Regul.* | OAM osmotic buffering (Muslin) |
| SRDI (2020) *Salinity Report, Bangladesh* | Noakhali coastal salinity profile |

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this simulation framework in your research, please cite:

```
Noman, Md. (2026). Cotton Salinity Resilience Simulation Framework 
[Computer software]. GitHub. 
https://github.com/md-noman-research/cotton-yield-simulation
```

---

## Contact

For scientific correspondence regarding the model formulation, please open a [GitHub Issue](https://github.com/md-noman-research/cotton-yield-simulation/issues).
