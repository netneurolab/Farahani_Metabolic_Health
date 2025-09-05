"""

HCP-A - age corrected using GAMLSS models

################################## lv = 0 #####################################

Post-hoc Resampled Mann–Whitney U (Male):
WMH vs Perfusion: p = 0.0001, p_fdr = 0.0002
WMH vs Myelin: p = 0.0010, p_fdr = 0.0015
WMH vs FA: p = 0.0025, p_fdr = 0.0037
WMH vs SC: p = 0.0036, p_fdr = 0.0052
WMH vs Perfusion_: p = 0.0001, p_fdr = 0.0002
WMH vs FA_: p = 0.0111, p_fdr = 0.0155
Perfusion vs Arrival: p = 0.0001, p_fdr = 0.0002
Perfusion vs Thickness: p = 0.0001, p_fdr = 0.0002
Perfusion vs Myelin: p = 0.0001, p_fdr = 0.0002
Perfusion vs FA: p = 0.0001, p_fdr = 0.0002
Perfusion vs MD: p = 0.0001, p_fdr = 0.0002
Perfusion vs FC: p = 0.0001, p_fdr = 0.0002
Perfusion vs SC: p = 0.0001, p_fdr = 0.0002
Perfusion vs Perfusion_: p = 0.0002, p_fdr = 0.0003
Perfusion vs Arrival_: p = 0.0001, p_fdr = 0.0002
Perfusion vs FA_: p = 0.0001, p_fdr = 0.0002
Perfusion vs MD_: p = 0.0001, p_fdr = 0.0002
Arrival vs Thickness: p = 0.0001, p_fdr = 0.0002
Arrival vs Myelin: p = 0.0001, p_fdr = 0.0002
Arrival vs FA: p = 0.0001, p_fdr = 0.0002
Arrival vs MD: p = 0.0251, p_fdr = 0.0338
Arrival vs FC: p = 0.0003, p_fdr = 0.0005
Arrival vs SC: p = 0.0001, p_fdr = 0.0002
Arrival vs Perfusion_: p = 0.0001, p_fdr = 0.0002
Arrival vs FA_: p = 0.0136, p_fdr = 0.0186
Thickness vs Myelin: p = 0.0001, p_fdr = 0.0002
Thickness vs FA: p = 0.0001, p_fdr = 0.0002
Thickness vs MD: p = 0.0001, p_fdr = 0.0002
Thickness vs FC: p = 0.0001, p_fdr = 0.0002
Thickness vs SC: p = 0.0004, p_fdr = 0.0006
Thickness vs Perfusion_: p = 0.0001, p_fdr = 0.0002
Thickness vs Arrival_: p = 0.0010, p_fdr = 0.0015
Thickness vs MD_: p = 0.0001, p_fdr = 0.0002
Myelin vs MD: p = 0.0001, p_fdr = 0.0002
Myelin vs FC: p = 0.0001, p_fdr = 0.0002
Myelin vs Perfusion_: p = 0.0001, p_fdr = 0.0002
Myelin vs Arrival_: p = 0.0001, p_fdr = 0.0002
Myelin vs MD_: p = 0.0001, p_fdr = 0.0002
FA vs MD: p = 0.0001, p_fdr = 0.0002
FA vs FC: p = 0.0001, p_fdr = 0.0002
FA vs SC: p = 0.0070, p_fdr = 0.0099
FA vs Perfusion_: p = 0.0001, p_fdr = 0.0002
FA vs Arrival_: p = 0.0001, p_fdr = 0.0002
FA vs MD_: p = 0.0001, p_fdr = 0.0002
MD vs SC: p = 0.0001, p_fdr = 0.0002
MD vs Perfusion_: p = 0.0001, p_fdr = 0.0002
MD vs FA_: p = 0.0001, p_fdr = 0.0002
FC vs SC: p = 0.0001, p_fdr = 0.0002
FC vs Perfusion_: p = 0.0001, p_fdr = 0.0002
FC vs FA_: p = 0.0001, p_fdr = 0.0002
SC vs Perfusion_: p = 0.0001, p_fdr = 0.0002
SC vs Arrival_: p = 0.0001, p_fdr = 0.0002
SC vs MD_: p = 0.0001, p_fdr = 0.0002
Perfusion_ vs Arrival_: p = 0.0001, p_fdr = 0.0002
Perfusion_ vs FA_: p = 0.0001, p_fdr = 0.0002
Perfusion_ vs MD_: p = 0.0001, p_fdr = 0.0002
Arrival_ vs FA_: p = 0.0023, p_fdr = 0.0034
FA_ vs MD_: p = 0.0006, p_fdr = 0.0010

Post-hoc Resampled Mann–Whitney U (Female):
WMH vs Perfusion: p = 0.0001, p_fdr = 0.0002
WMH vs MD: p = 0.0055, p_fdr = 0.0082
WMH vs FC: p = 0.0001, p_fdr = 0.0002
WMH vs Perfusion_: p = 0.0001, p_fdr = 0.0002
Perfusion vs Arrival: p = 0.0001, p_fdr = 0.0002
Perfusion vs Thickness: p = 0.0001, p_fdr = 0.0002
Perfusion vs Myelin: p = 0.0001, p_fdr = 0.0002
Perfusion vs FA: p = 0.0001, p_fdr = 0.0002
Perfusion vs MD: p = 0.0001, p_fdr = 0.0002
Perfusion vs FC: p = 0.0001, p_fdr = 0.0002
Perfusion vs SC: p = 0.0001, p_fdr = 0.0002
Perfusion vs Perfusion_: p = 0.0001, p_fdr = 0.0002
Perfusion vs Arrival_: p = 0.0001, p_fdr = 0.0002
Perfusion vs FA_: p = 0.0001, p_fdr = 0.0002
Perfusion vs MD_: p = 0.0001, p_fdr = 0.0002
Arrival vs Thickness: p = 0.0001, p_fdr = 0.0002
Arrival vs Myelin: p = 0.0001, p_fdr = 0.0002
Arrival vs FA: p = 0.0001, p_fdr = 0.0002
Arrival vs MD: p = 0.0001, p_fdr = 0.0002
Arrival vs FC: p = 0.0001, p_fdr = 0.0002
Arrival vs SC: p = 0.0001, p_fdr = 0.0002
Arrival vs Perfusion_: p = 0.0001, p_fdr = 0.0002
Arrival vs MD_: p = 0.0128, p_fdr = 0.0185
Thickness vs MD: p = 0.0001, p_fdr = 0.0002
Thickness vs FC: p = 0.0001, p_fdr = 0.0002
Thickness vs Perfusion_: p = 0.0001, p_fdr = 0.0002
Thickness vs Arrival_: p = 0.0269, p_fdr = 0.0368
Thickness vs FA_: p = 0.0156, p_fdr = 0.0221
Myelin vs MD: p = 0.0001, p_fdr = 0.0002
Myelin vs FC: p = 0.0001, p_fdr = 0.0002
Myelin vs SC: p = 0.0069, p_fdr = 0.0102
Myelin vs Perfusion_: p = 0.0001, p_fdr = 0.0002
Myelin vs Arrival_: p = 0.0350, p_fdr = 0.0471
Myelin vs FA_: p = 0.0189, p_fdr = 0.0263
FA vs MD: p = 0.0001, p_fdr = 0.0002
FA vs FC: p = 0.0001, p_fdr = 0.0002
FA vs Perfusion_: p = 0.0001, p_fdr = 0.0002
FA vs Arrival_: p = 0.0022, p_fdr = 0.0036
FA vs FA_: p = 0.0012, p_fdr = 0.0021
MD vs FC: p = 0.0001, p_fdr = 0.0002
MD vs SC: p = 0.0001, p_fdr = 0.0002
MD vs Perfusion_: p = 0.0001, p_fdr = 0.0002
MD vs Arrival_: p = 0.0036, p_fdr = 0.0055
MD vs FA_: p = 0.0028, p_fdr = 0.0044
MD vs MD_: p = 0.0001, p_fdr = 0.0002
FC vs SC: p = 0.0001, p_fdr = 0.0002
FC vs Perfusion_: p = 0.0001, p_fdr = 0.0002
FC vs Arrival_: p = 0.0001, p_fdr = 0.0002
FC vs FA_: p = 0.0001, p_fdr = 0.0002
FC vs MD_: p = 0.0001, p_fdr = 0.0002
SC vs Perfusion_: p = 0.0001, p_fdr = 0.0002
SC vs Arrival_: p = 0.0023, p_fdr = 0.0037
SC vs FA_: p = 0.0009, p_fdr = 0.0016
Perfusion_ vs Arrival_: p = 0.0001, p_fdr = 0.0002
Perfusion_ vs FA_: p = 0.0001, p_fdr = 0.0002
Perfusion_ vs MD_: p = 0.0001, p_fdr = 0.0002
Arrival_ vs MD_: p = 0.0015, p_fdr = 0.0025
FA_ vs MD_: p = 0.0013, p_fdr = 0.0022

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import combinations
from globals import path_results, path_figures
from statsmodels.stats.multitest import multipletests

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

lv = 0

nnodes = 400
features_cortex = ["Perfusion", "Arrival", "Thickness", "Myelin", "FA", "MD", "FC", "SC"]
features_wm = ["Perfusion", "Arrival", "FA", "MD"]
features = ["WMH"] + features_cortex + features_wm

def pad_to_400(x):
    """Pads arrays shorter than 400 with NaN"""
    return np.pad(x, (0, nnodes - len(x)), constant_values = np.nan) if len(x) < nnodes else x

variances_male = np.zeros(len(features))
variances_female = np.zeros(len(features))
data_male = np.zeros((nnodes, len(features)))
data_female = np.zeros((nnodes, len(features)))

for m in range(len(features)):
    if m == 0:
        x_male = np.load(f"{path_results}GAM_loadings_WMH_Male_lv_{lv}.npy")
        x_female = np.load(f"{path_results}GAM_loadings_WMH_Female_lv_{lv}.npy")
    elif 1 <= m < 9:
        fname = features_cortex[m - 1]
        x_male = np.load(f"{path_results}GAM_loadings_CORTEX_{fname}_Male_lv_{lv}.npy")
        x_female = np.load(f"{path_results}GAM_loadings_CORTEX_{fname}_Female_lv_{lv}.npy")
    else:
        fname = features_wm[m - 9]
        x_male = np.load(f"{path_results}GAM_loadings_WM_{fname}_Male_lv_{lv}.npy")
        x_female = np.load(f"{path_results}GAM_loadings_WM_{fname}_Female_lv_{lv}.npy")

    x_male = pad_to_400(x_male)
    x_female = pad_to_400(x_female)

    variances_male[m] = np.nanmean(np.abs(x_male))
    variances_female[m] = np.nanmean(np.abs(x_female))

    data_male[:, m] = x_male
    data_female[:, m] = x_female

#------------------------------------------------------------------------------
# Do statistics
#------------------------------------------------------------------------------

features_cortex = ["Perfusion", "Arrival", "Thickness", "Myelin", "FA", "MD", "FC", "SC"]
features_wm = ["Perfusion_", "Arrival_", "FA_", "MD_"]
features = ["WMH"] + features_cortex + features_wm 

def mannwhitney_permutation(x, y, n_perm = 10000, alternative = 'two-sided', seed = None):
    if seed is not None:
        np.random.seed(seed)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    obs_diff = np.median(x) - np.median(y)

    combined = np.concatenate([x, y])
    n_x = len(x)

    null_dist = np.zeros(n_perm)
    for i in range(n_perm):
        np.random.shuffle(combined)
        null_x = combined[:n_x]
        null_y = combined[n_x:]
        null_dist[i] = np.median(null_x) - np.median(null_y)

    if alternative == 'two-sided':
        p = (1 + np.count_nonzero(abs((null_dist - np.mean(null_dist))) > abs((obs_diff - np.mean(null_dist))))) / (n_perm + 1)
    elif alternative == 'greater':
        p = np.mean(null_dist >= obs_diff)
    else:
        p = np.mean(null_dist <= obs_diff)
    return p

def pairwise_median_resample(data, features, n_perm = 10000):
    pvals = []
    pairs = []

    for i, j in combinations(range(len(features)), 2):
        x = data[:, i]
        y = data[:, j]

        p = mannwhitney_permutation(x, y, n_perm = n_perm, alternative = 'two-sided')
        pvals.append(p)
        pairs.append((features[i], features[j]))

    pvals = np.array(pvals)
    pvals_fdr = np.full_like(pvals, np.nan)

    valid = ~np.isnan(pvals)
    if np.any(valid):
        _, corrected, _, _ = multipletests(pvals[valid], method = 'fdr_bh')
        pvals_fdr[valid] = corrected

    return pairs, pvals.tolist(), pvals_fdr.tolist()

pairs_m, raw_p_m, fdr_p_m = pairwise_median_resample(np.abs(data_male), features)
pairs_f, raw_p_f, fdr_p_f = pairwise_median_resample(np.abs(data_female), features)

print("\nPost-hoc Resampled Mann–Whitney U (Male):")
for i, (f1, f2) in enumerate(pairs_m):
    if fdr_p_m[i] < 0.05:
        print(f"{f1} vs {f2}: p = {raw_p_m[i]:.4f}, p_fdr = {fdr_p_m[i]:.4f}")

print("\nPost-hoc Resampled Mann–Whitney U (Female):")
for i, (f1, f2) in enumerate(pairs_f):
    if fdr_p_f[i] < 0.05:
        print(f"{f1} vs {f2}: p = {raw_p_f[i]:.4f}, p_fdr = {fdr_p_f[i]:.4f}")

#------------------------------------------------------------------------------
# Plot the results
#------------------------------------------------------------------------------

mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.dpi': 300
})

def annotate_significance(ax, features, pairs, pvals_fdr, alpha_levels = [0.05, 0.01, 0.001]):
    feature_idx = {feat: i for i, feat in enumerate(features)}
    heights = [bar.get_height() for bar in ax.patches]
    used_y = {}  # To stack lines

    for i, (f1, f2) in enumerate(pairs):
        if pvals_fdr[i] < alpha_levels[0]:
            idx1, idx2 = feature_idx[f1], feature_idx[f2]
            x1, x2 = min(idx1, idx2), max(idx1, idx2)
            h1, h2 = heights[idx1], heights[idx2]
            base_y = max(h1, h2) + 0.01
            y = base_y

            # Keep raising the line if it overlaps with an existing one
            while y in used_y:
                y += 0.01
            used_y[y] = True

            # Determine significance level
            if pvals_fdr[i] < alpha_levels[2]:
                stars = '***'
            elif pvals_fdr[i] < alpha_levels[1]:
                stars = '**'
            else:
                stars = '*' 
            # Draw line and star
            ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw = 1.5, color = 'black')
            ax.text((x1 + x2) / 2, y + 0.015, stars, ha = 'center', va = 'bottom', fontsize = 14)

def plot_with_significance(variances, pairs, pvals_fdr, sex_label, lv):
    fig, ax = plt.subplots(figsize = (12, 10))
    bars = ax.bar(features, variances, color = 'silver', edgecolor = 'black', linewidth = 1.2)
    ax.set_title(f'Mean Absolute Loadings - {sex_label}')
    ax.set_ylabel('Mean |Loading|')
    ax.set_ylim(0, max(variances) + 0.2)
    annotate_significance(ax, features, pairs, pvals_fdr)
    plt.xticks(rotation = 45)
    plt.tight_layout()
    fig.savefig(path_figures + f'GAM_loadings_{sex_label.lower()}_lv_{lv}.svg', format = 'svg')
    plt.show()

# Plot male
plot_with_significance(variances_male, pairs_m, fdr_p_m, 'Male', lv)

# Plot female
plot_with_significance(variances_female, pairs_f, fdr_p_f, 'Female', lv)

#------------------------------------------------------------------------------
# Another way of visualization of the results and statistics
#------------------------------------------------------------------------------

def annotate_significance_dotplot(ax, features, pairs, pvals_fdr, data, alpha_levels = [0.05, 0.01, 0.001]):
    feature_idx = {feat: i for i, feat in enumerate(features)}
    mean_heights = [np.nanmean(np.abs(data[:, i])) for i in range(len(features))]
    used_y = {}

    for i, (f1, f2) in enumerate(pairs):
        if pvals_fdr[i] < alpha_levels[0]:
            idx1, idx2 = feature_idx[f1], feature_idx[f2]
            x1, x2 = min(idx1, idx2), max(idx1, idx2)
            h1, h2 = mean_heights[idx1], mean_heights[idx2]
            base_y = max(h1, h2) + 0.01
            y = base_y

            # Raise line if overlapping
            while y in used_y:
                y += 0.01
            used_y[y] = True

            # Significance stars
            if pvals_fdr[i] < alpha_levels[2]:
                stars = '***'
            elif pvals_fdr[i] < alpha_levels[1]:
                stars = '**'
            else:
                stars = '*'

            # Line and annotation
            ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw = 1.5, color = 'black')
            ax.text((x1 + x2) / 2, y + 0.015, stars, ha = 'center', va = 'bottom', fontsize = 14)

def plot_with_dots_and_significance(data, pairs, pvals_fdr, sex_label, lv):
    fig, ax = plt.subplots(figsize=(12, 10))

    for i, feature in enumerate(features):
        values = np.abs(data[:, i])
        values = values[~np.isnan(values)]
        x = np.random.normal(i, 0.08, size = len(values)) # jitter for dot separation
        ax.scatter(x, values, alpha = 0.5, color = 'gray', edgecolor = 'black', linewidth = 0.3)

        # Plot mean
        mean_val = np.nanmean(values)
        ax.plot(i, mean_val, marker = 'o', color = 'black', markersize = 7, zorder = 3)

    ax.set_title(f'Mean Absolute Loadings - {sex_label}')
    ax.set_ylabel('Mean |Loading|')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation = 45, ha = 'right')
    ax.set_ylim(0, np.nanmax(np.abs(data)) + 0.2)

    annotate_significance_dotplot(ax, features, pairs, pvals_fdr, data)
    plt.tight_layout()
    fig.savefig(path_figures + f'GAM_loadings_dotplot_{sex_label.lower()}_lv_{lv}.svg', format = 'svg')
    plt.show()

plot_with_dots_and_significance(data_male, pairs_m, fdr_p_m, 'Male', lv)
plot_with_dots_and_significance(data_female, pairs_f, fdr_p_f, 'Female', lv)

#------------------------------------------------------------------------------
# END
