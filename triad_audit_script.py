"""
statistical cross-validation of cotton yields under compound stress using 
multiple modeling frameworks (wls, gamma glm, and mechanistic nls).

Author: Md. Noman
Affiliation: Noakhali Science and Technology University (NSTU)
Email: md.noman.research@gmail.com
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# load sim data generated from ar(1) environmental trajectories
df_all = pd.read_csv('noakhali_cotton_simulation_results.csv')

def run_triad_analysis(df, name):
    """comparative robustness check across linear and non-linear domains."""
    df = df.copy()
    df['log_yield'] = np.log(df['Yield'] + 1e-6)
    df['yield_pos'] = df['Yield'].clip(lower=1e-6)
    
    features = ['Avg_Salinity_P2', 'Max_Salinity', 'Avg_Temp']
    X = sm.add_constant(df[features])
    
    # log-linear wls to stabilize variance under multiplicative damage
    ols_res = sm.OLS(df['log_yield'], X).fit()
    weights = 1.0 / np.exp(ols_res.fittedvalues.clip(lower=-20, upper=20))
    wls_model = sm.WLS(df['log_yield'], X, weights=weights).fit()
    wls_r2 = wls_model.rsquared_adj
    
    # gamma glm handles the heavy-tailed, right-skewed yield distribution
    gamma_model = smf.glm('yield_pos ~ Avg_Salinity_P2 + Max_Salinity + Avg_Temp', 
                         data=df, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    pseudo_r2 = 1 - (gamma_model.deviance / gamma_model.null_deviance)
    
    # fit against generative mechanistic exponential decay function (maas & hoffman, 1977)
    def mechanistic_exp(X_tuple, b0, b1, b2, b3):
        s, ms, t = X_tuple
        return b0 * np.exp(-(b1*s + b2*ms + b3*t))
    
    x_data = (df['Avg_Salinity_P2'].values, df['Max_Salinity'].values, df['Avg_Temp'].values)
    try:
        p0 = [500, 0.1, 0.05, 0.1] 
        popt, _ = curve_fit(mechanistic_exp, x_data, df['Yield'].values, p0=p0, maxfev=5000)
        y_pred_nls = mechanistic_exp(x_data, *popt)
        nls_r2 = 1 - (np.sum((df['Yield'] - y_pred_nls)**2) / np.sum((df['Yield'] - df['Yield'].mean())**2))
    except Exception:
        nls_r2 = np.nan

    # visual confirmation of log-linearization (supplementary fig s1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(x='Avg_Salinity_P2', y='Yield', data=df, ax=ax[0], 
                scatter_kws={'alpha':0.4, 's':20, 'color':'#1f77b4'},
                line_kws={'color':'red', 'lw':1.5})
    ax[0].set_xlabel('Avg Salinity P2 (dS/m)')
    ax[0].set_ylabel('Raw Yield (kg/ha)') # response under salinity
    ax[0].set_title(f'{name.upper()}:\nRaw Yield vs Salinity\n(Heteroscedastic)')
    
    sns.regplot(x='Avg_Salinity_P2', y='log_yield', data=df, ax=ax[1], 
                scatter_kws={'alpha':0.4, 's':20, 'color':'#2ca02c'},
                line_kws={'color':'red', 'lw':1.5})
    ax[1].set_xlabel('Avg Salinity P2 (dS/m)')
    ax[1].set_ylabel('Log(Yield)')
    ax[1].set_title(f'{name.upper()}:\nLog-Transformed\n(Linearized)')
    
    plt.tight_layout()
    plt.savefig(f'{name}_yield_transform.png', dpi=300)
    plt.close()
    
    # TODO: evaluate model selection using aic/bic criteria
    return wls_r2, pseudo_r2, nls_r2

suvin_data = df_all[df_all['Cotton'] == 'Suvin']
muslin_data = df_all[df_all['Cotton'] == 'Muslin']

r_s = run_triad_analysis(suvin_data, 'suvin')
r_m = run_triad_analysis(muslin_data, 'muslin')

# summary of model performance (r-squared)
print("\n" + "-"*48)
print(f"{'Statistical Method':<22} {'Suvin R2':<12} {'Muslin R2':<12}")
print("-"*48)
print(f"{'WLS Log-Linear':<22} {r_s[0]:<12.3f} {r_m[0]:<12.3f}")
print(f"{'Gamma GLM':<22} {r_s[1]:<12.3f} {r_m[1]:<12.3f}")
print(f"{'NLS (Mechanistic)':<22} {r_s[2]:<12.3f} {r_m[2]:<12.3f}")
print("-"*48 + "\n")
