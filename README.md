# Stochastic Yield Response of Cotton (Suvin vs Muslin) under Coastal Salinity Stress

## Overview
This repository contains a mechanistic simulation framework designed to evaluate the differential yield resilience of two cotton genotypes—*Gossypium barbadense* L. (Suvin) and *Gossypium arboreum* L. (Muslin)—under compound salinity and thermal stress. By utilizing a 1,000-iteration Monte Carlo engine, the simulation models the stochastic nature of pre-monsoon environmental persistence in deltaic environments.

## Research Context
The study is situated in the accreted char lands of Noakhali, Bangladesh, an environment characterized by sustained ionic loading (ECe > 12.0 dS/m) and high capillary salt persistence. The primary objective is to investigate whether the hypothesized osmotic adjustment potential of diploid landraces (Muslin) provides a superior risk-stabilization profile compared to high-value tetraploid cultivars (Suvin) for smallholder farmers in vulnerable coastal zones.

## Methodology
The analytical pipeline utilizes a three-tier statistical audit to account for the heteroscedasticity and non-linearity inherent in exponential yield decay models:
1.  **Weighted Least Squares (WLS)**: Applied to log-transformed yield data to stabilize multiplicative variance.
2.  **Gamma Generalized Linear Model (GLM)**: Implemented with a log-link function for skewed yield distributions.
3.  **Nonlinear Least Squares (NLS)**: Direct mechanistic fitting against the generative exponential damage function for structural validation.

## Requirements
*   Python 3.8+
*   NumPy & Pandas
*   SciPy
*   Statsmodels
*   Matplotlib & Seaborn
*   Scikit-learn (RandomForestRegressor)

## Author
**Md. Noman**  
Noakhali Science and Technology University (NSTU)  
Email: md.noman.research@gmail.com
