"""

HCP-A - age corrected using GAMLSS models

Are different brain features (cortical) affected by metabolic dysfunction in the same way?
This is investigated for males and females separately!

-------------------------------------------------------------------------------
lv = 0
-------------------------------------------------------------------------------

=== Male (LV 0): Named correlations with FDR-corrected spin p-values ===
Perfusion–FC          r = +0.215   p_fdr = 0.042
FA–MD                 r = -0.442   p_fdr = 0.00699
FA–SC                 r = +0.229   p_fdr = 0.00699
Arrival–FC            r = -0.331   p_fdr = 0.00699
Myelin–SC             r = -0.180   p_fdr = 0.00699
Thickness–Myelin      r = -0.133   p_fdr = 0.042

=== Female (LV 0): Named correlations with FDR-corrected spin p-values ===
Perfusion–FC          r = +0.305   p_fdr = 0.0186
FA–MD                 r = -0.340   p_fdr = 0.014
FA–SC                 r = +0.234   p_fdr = 0.014
Thickness–FA          r = +0.238   p_fdr = 0.042

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from globals import path_results, path_figures
from functions import vasa_null_Schaefer, pval_cal
from statsmodels.stats.multitest import multipletests

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------

lv = 0

nnodes = 400
nspins = 1000
features = ["Perfusion", "Arrival", "Thickness", "Myelin", "FA", "MD", "FC", "SC"]
spins = vasa_null_Schaefer(nspins)

#------------------------------------------------------------------------------
# Load PLS Loadings
#------------------------------------------------------------------------------

loadings_female = np.zeros((len(features), nnodes))
loadings_male = np.zeros((len(features), nnodes))

for i in range(len(features)):
    loadings_female[i,:] = np.load(path_results + 'GAM_loadings_CORTEX_' + features[i] + '_Female_lv_' + str(lv) + '.npy')
    loadings_male[i,:] = np.load(path_results + 'GAM_loadings_CORTEX_' + features[i] + '_Male_lv_' + str(lv) + '.npy')
loadings_female = loadings_female.T
loadings_male = loadings_male.T

#------------------------------------------------------------------------------
# Utility: Spin-corrected Pearson correlation
#------------------------------------------------------------------------------

def corr_spin(x, y, spins, nspins):
    rho, _ = pearsonr(x, y)
    null = np.array([pearsonr(x, y[spins[:, i]])[0] for i in range(nspins)])
    return rho, null

#------------------------------------------------------------------------------
# Plot heatmap
#------------------------------------------------------------------------------

def plot_similarity_heatmap_with_spin_fdr(data, feature_labels, group, spins, nspins = 1000):
    num_features = data.shape[1]
    sim_matrix = np.zeros((num_features, num_features))
    pval_matrix = np.ones((num_features, num_features))

    # upper triangle indices (no diagonal)
    iu = np.triu_indices(num_features, k = 1)

    # store raw p-values for upper triangle
    upp_p = np.empty_like(iu[0], dtype = float)

    for idx, (i, j) in enumerate(zip(*iu)):
        r, nulls = corr_spin(data[:, i], data[:, j], spins, nspins)
        sim_matrix[i, j] = r
        p = pval_cal(r, nulls, nspins)
        pval_matrix[i, j] = pval_matrix[j, i] = p
        upp_p[idx] = p

    _, upp_q, _, _ = multipletests(upp_p, alpha = 0.05, method = 'fdr_bh')

    # fill corrected q-values into a matrix
    pval_matrix_fdr = np.ones_like(pval_matrix)
    for (i, j), q in zip(zip(*iu), upp_q):
        pval_matrix_fdr[i, j] = pval_matrix_fdr[j, i] = q

    annot = np.empty_like(sim_matrix, dtype = object)
    for i in range(num_features):
        for j in range(num_features):
            r_val = sim_matrix[i, j]
            p_val = pval_matrix_fdr[i, j]
            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            annot[i, j] = f'{r_val:.2f}{stars}'

    mask = np.tril(np.ones_like(sim_matrix, dtype = bool)) # mask lower triangle

    plt.figure(figsize = (7, 6))
    sns.heatmap(sim_matrix,
                vmin = -1, vmax = 1,
                cmap = 'coolwarm', square = True,
                xticklabels = feature_labels, yticklabels = feature_labels,
                annot = annot, fmt = '',
                mask = mask,
                cbar_kws = {'label': 'Pearson r'})
    plt.title(f'Feature Similarity Heatmap ({group.capitalize()}, LV {lv}) - FDR Corrected')
    plt.tight_layout()
    plt.savefig(f"{path_figures}GAM_similarity_heatmap_{group}_lv{lv}_spin_fdr.svg", dpi = 300)
    plt.show()
    return sim_matrix, pval_matrix_fdr

#------------------------------------------------------------------------------
# Plot loadings similarity
#------------------------------------------------------------------------------

sim_loadings_female, p_female = plot_similarity_heatmap_with_spin_fdr(loadings_female, features, 'Female', spins, nspins)
sim_loadings_male, p_male = plot_similarity_heatmap_with_spin_fdr(loadings_male, features, 'Male', spins, nspins)

#------------------------------------------------------------------------------
# Print correlations with FDR-corrected p
#------------------------------------------------------------------------------

def tidy_pairs(sim, p, labels):
    rows = []
    n = len(labels)
    for i in range(n):
        for j in range(i+1, n):
            rows.append({
                "feature_1": labels[i],
                "feature_2": labels[j],
                "pair": f"{labels[i]}–{labels[j]}",
                "r": sim[i, j],
                "p_fdr": p[i, j]
            })
    # Sort by absolute correlation (desc), then by p-value (asc)
    df = pd.DataFrame(rows).sort_values(["r"], key=lambda s: s.abs(), ascending = False).reset_index(drop = True)
    return df

def print_pairs(df, group_label):
    print(f"\n=== {group_label}: Named correlations with FDR-corrected spin p-values ===")
    for _, row in df.iterrows():
        print(f"{row['pair']:<20s}  r = {row['r']:+.3f}   p_fdr = {row['p_fdr']:.3g}")

# Build tidy tables
df_female = tidy_pairs(sim_loadings_female, p_female, features)
df_male   = tidy_pairs(sim_loadings_male,   p_male,   features)

# Print to console
print_pairs(df_female, f"Female (LV {lv})")
print_pairs(df_male,   f"Male (LV {lv})")

#------------------------------------------------------------------------------
# END