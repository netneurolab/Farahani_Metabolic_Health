"""

UK Biobank - GAMLSS models are used to correct age-effects.

################################## lv = 0 #####################################

Post-hoc Resampled Mann–Whitney U (Male):
Area - DK vs FA - JHU: p = 0.0004, p_fdr = 0.0010
Area - DK vs MD - JHU: p = 0.0009, p_fdr = 0.0020
Area - DK vs ISOVF - JHU: p = 0.0001, p_fdr = 0.0003
Area - DK vs Volume - subcortex: p = 0.0073, p_fdr = 0.0138
Area - DK vs WMH: p = 0.0043, p_fdr = 0.0084
Area - DK vs ATT: p = 0.0001, p_fdr = 0.0003
Area - DK vs CBF: p = 0.0001, p_fdr = 0.0003
Volume - DK vs FA - JHU: p = 0.0001, p_fdr = 0.0003
Volume - DK vs MD - JHU: p = 0.0001, p_fdr = 0.0003
Volume - DK vs ISOVF - JHU: p = 0.0001, p_fdr = 0.0003
Volume - DK vs Volume - subcortex: p = 0.0004, p_fdr = 0.0010
Volume - DK vs WMH: p = 0.0034, p_fdr = 0.0069
Volume - DK vs ATT: p = 0.0001, p_fdr = 0.0003
Volume - DK vs CBF: p = 0.0001, p_fdr = 0.0003
Thickness - DK vs FA - JHU: p = 0.0154, p_fdr = 0.0282
Thickness - DK vs MD - JHU: p = 0.0188, p_fdr = 0.0334
Thickness - DK vs ISOVF - JHU: p = 0.0030, p_fdr = 0.0063
Thickness - DK vs ATT: p = 0.0001, p_fdr = 0.0003
Thickness - DK vs CBF: p = 0.0001, p_fdr = 0.0003
FA - JHU vs ATT: p = 0.0001, p_fdr = 0.0003
FA - JHU vs CBF: p = 0.0001, p_fdr = 0.0003
MD - JHU vs ATT: p = 0.0001, p_fdr = 0.0003
MD - JHU vs CBF: p = 0.0001, p_fdr = 0.0003
ISOVF - JHU vs ATT: p = 0.0005, p_fdr = 0.0011
ISOVF - JHU vs CBF: p = 0.0001, p_fdr = 0.0003
ICVF - JHU vs ATT: p = 0.0001, p_fdr = 0.0003
ICVF - JHU vs CBF: p = 0.0001, p_fdr = 0.0003
Volume - subcortex vs ATT: p = 0.0001, p_fdr = 0.0003
Volume - subcortex vs CBF: p = 0.0001, p_fdr = 0.0003
WMH vs CBF: p = 0.0001, p_fdr = 0.0003
ATT vs CBF: p = 0.0001, p_fdr = 0.0003

Post-hoc Resampled Mann–Whitney U (Female):
Area - DK vs Thickness - DK: p = 0.0027, p_fdr = 0.0059
Area - DK vs FA - JHU: p = 0.0001, p_fdr = 0.0003
Area - DK vs MD - JHU: p = 0.0005, p_fdr = 0.0012
Area - DK vs ISOVF - JHU: p = 0.0001, p_fdr = 0.0003
Area - DK vs ICVF - JHU: p = 0.0069, p_fdr = 0.0131
Area - DK vs Volume - subcortex: p = 0.0047, p_fdr = 0.0092
Area - DK vs WMH: p = 0.0109, p_fdr = 0.0193
Area - DK vs ATT: p = 0.0001, p_fdr = 0.0003
Area - DK vs CBF: p = 0.0001, p_fdr = 0.0003
Volume - DK vs Thickness - DK: p = 0.0007, p_fdr = 0.0017
Volume - DK vs FA - JHU: p = 0.0001, p_fdr = 0.0003
Volume - DK vs MD - JHU: p = 0.0001, p_fdr = 0.0003
Volume - DK vs ISOVF - JHU: p = 0.0001, p_fdr = 0.0003
Volume - DK vs ICVF - JHU: p = 0.0028, p_fdr = 0.0059
Volume - DK vs Volume - subcortex: p = 0.0009, p_fdr = 0.0021
Volume - DK vs WMH: p = 0.0103, p_fdr = 0.0189
Volume - DK vs ATT: p = 0.0001, p_fdr = 0.0003
Volume - DK vs CBF: p = 0.0001, p_fdr = 0.0003
Thickness - DK vs ATT: p = 0.0001, p_fdr = 0.0003
Thickness - DK vs CBF: p = 0.0001, p_fdr = 0.0003
FA - JHU vs ATT: p = 0.0001, p_fdr = 0.0003
FA - JHU vs CBF: p = 0.0001, p_fdr = 0.0003
MD - JHU vs ATT: p = 0.0029, p_fdr = 0.0059
MD - JHU vs CBF: p = 0.0001, p_fdr = 0.0003
ISOVF - JHU vs Volume - subcortex: p = 0.0205, p_fdr = 0.0342
ISOVF - JHU vs ATT: p = 0.0143, p_fdr = 0.0246
ISOVF - JHU vs CBF: p = 0.0001, p_fdr = 0.0003
ICVF - JHU vs ATT: p = 0.0001, p_fdr = 0.0003
ICVF - JHU vs CBF: p = 0.0001, p_fdr = 0.0003
Volume - subcortex vs ATT: p = 0.0001, p_fdr = 0.0003
Volume - subcortex vs CBF: p = 0.0001, p_fdr = 0.0003
WMH vs CBF: p = 0.0001, p_fdr = 0.0003
ATT vs CBF: p = 0.0001, p_fdr = 0.0003

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

lv = 0 # this can be 0 or 1!

nnodes = 400

feature_slices = {
    "Area - DK": slice(0, 62),
    "Volume - DK": slice(62, 124),
    "Thickness - DK": slice(124, 186),

    "FA - JHU": slice(186, 234),
    "MD - JHU": slice(234, 282),
    "ISOVF - JHU": slice(282, 330),
    "ICVF - JHU": slice(330, 378),
    
    "Volume - subcortex": slice(378, 438),
    "WMH": slice(438, 440),
    "ATT": slice(440, 465),
    "CBF": slice(465, 490)
}

x_male = np.load(path_results + f'GAM_ukbb_Male_brain_loadings_lv_{lv}.npy')
x_female = np.load(path_results + f'GAM_ukbb_Female_brain_loadings_lv_{lv}.npy')
  
features = [
         "Area - DK",
         "Volume - DK",
         "Thickness - DK",
         "FA - JHU",
         "MD - JHU",
         "ISOVF - JHU",
         "ICVF - JHU",
         "Volume - subcortex",
         "WMH",
         "ATT",
         "CBF"]

def calculate_feature_data(data, feature_slices, features):
    """Return per-feature arrays without collapsing to mean."""
    feature_arrays = []
    for feature in features:
        slice_obj = feature_slices[feature]
        feature_values = np.abs(data[slice_obj])  # keep array
        feature_arrays.append(feature_values)
    return feature_arrays

# Keep full arrays for stats
data_male_full = calculate_feature_data(x_male, feature_slices, features)
data_female_full = calculate_feature_data(x_female, feature_slices, features)

# Still compute variances for plotting (optional)
variances_male = [np.mean(vals) for vals in data_male_full]
variances_female = [np.mean(vals) for vals in data_female_full]

#------------------------------------------------------------------------------
# Do statistics
#------------------------------------------------------------------------------

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
        x = data[ i]
        y = data[ j]

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

pairs_m, raw_p_m, fdr_p_m = pairwise_median_resample(data_male_full, features)
pairs_f, raw_p_f, fdr_p_f = pairwise_median_resample(data_female_full, features)

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
    fig.savefig(path_figures + f'GAM_ukbb_loadings_{sex_label.lower()}_lv_{lv}.svg', format = 'svg')
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
    mean_heights = [np.nanmean(np.abs(data[ i])) for i in range(len(features))]
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
    fig, ax = plt.subplots(figsize = (12, 10))

    for i, feature in enumerate(features):
        values = np.abs(data[ i])
        values = values[~np.isnan(values)]
        x = np.random.normal(i, 0.08, size = len(values))  # jitter for dot separation
        ax.scatter(x, values, alpha = 0.5, color = 'gray', edgecolor = 'black', linewidth = 0.3)

        # Plot mean
        mean_val = np.nanmean(values)
        ax.plot(i, mean_val, marker = 'o', color = 'black', markersize = 7, zorder = 3)

    ax.set_title(f'Mean Absolute Loadings - {sex_label}')
    ax.set_ylabel('Mean |Loading|')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation = 45, ha = 'right')

    # Flatten all data to find max for y-limit
    all_vals = np.concatenate([np.ravel(vals) for vals in data])
    ax.set_ylim(0, np.nanmax(np.abs(all_vals)) + 0.1)

    annotate_significance_dotplot(ax, features, pairs, pvals_fdr, data)
    plt.tight_layout()
    fig.savefig(path_figures + f'GAM_ukbb_loadings_dotplot_{sex_label.lower()}_lv_{lv}.svg', format = 'svg')
    plt.show()

plot_with_dots_and_significance(data_male_full, pairs_m, fdr_p_m, 'Male', lv)
plot_with_dots_and_significance(data_female_full, pairs_f, fdr_p_f, 'Female', lv)

#------------------------------------------------------------------------------
# END
