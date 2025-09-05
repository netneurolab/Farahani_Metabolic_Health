"""

UK Biobank - GAMLSS models are used to correct age-effects.

################################## Males ######################################

number of included subjects:    1431
number of brain features:       490
number of clinical measures:    33

    Singular values for latent variable 0: 0.6531
    x-score and y-score Spearman correlation for latent variable 0:           0.3694
    x-score and y-score Pearson correlation for latent variable 0:           0.4026
    p-value = 9.99000999e-04

################################# Females #####################################

number of included subjects:    1582
number of brain features:       490
number of clinical measures:    33

    Singular values for latent variable 0: 0.6037
    x-score and y-score Spearman correlation for latent variable 0:           0.3911
    x-score and y-score Pearson correlation for latent variable 0:           0.3824
    p-value = 9.99000999e-04

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import seaborn as sns
from pyls import behavioral_pls
import matplotlib.pyplot as plt
from scipy.stats import zscore, spearmanr, pearsonr
from globals import path_results, path_figures

#------------------------------------------------------------------------------
# Load data to run PLS model - GAM corrected data is used here.
#------------------------------------------------------------------------------

sex_label = 'Female' # This can be Female or Male!

data_bio_array =  np.load(path_results + f'biobank_biomarkers_{sex_label}_clean_centercorrected.npy') # (1582, 32)
data_brain_array = np.load(path_results + f'biobank_brain_data_{sex_label}_clean_centercorrected.npy') # (1582, 490)

brain_names = np.load(path_results + f'biobank_names_brain_data_{sex_label}_clean_centercorrected.npy', allow_pickle = True) # 490
bio_names = np.load(path_results + f'biobank_names_biomarkers_{sex_label}_clean_centercorrected.npy', allow_pickle = True)

age_0 = np.load(path_results + f'biobank_age_{sex_label}_0_0.npy').reshape(-1, 1)
age_2 = np.load(path_results + f'biobank_age_{sex_label}_2_0.npy').reshape(-1, 1)

data_bio_array = np.concatenate((age_0, age_2, data_bio_array), axis = 1)
bio_names = np.concatenate((np.array(['age_0']), np.array(['age_2']), bio_names), axis = 0)

# Z-score
X = zscore(data_brain_array, axis = 0)
Y = zscore(data_bio_array, axis = 0)

#------------------------------------------------------------------------------
# PLS
#------------------------------------------------------------------------------

nspins = 1000
num_subjects = len(X)
spins = np.zeros((num_subjects, nspins))
for spin_ind in range(nspins):
    spins[:,spin_ind] = np.random.permutation(range(0, num_subjects))
spins = spins.astype(int)

pls_result = behavioral_pls(X,
                            Y,
                            n_boot = nspins,
                            n_perm = nspins,
                            permsamples = spins,
                            test_split = 0,
                            seed = 0)

#------------------------------------------------------------------------------
# Save the results - PLS scores
#------------------------------------------------------------------------------

for lv in range(3):
    np.save(path_results + f'GAM_biobank_{sex_label}_score_x_lv_' + str(lv),
            pls_result['x_scores'][:, lv])
    np.save(path_results + f'GAM_biobank_{sex_label}_score_y_lv_' + str(lv),
            pls_result['y_scores'][:, lv])
    np.save(path_results + f'GAM_biobank_{sex_label}_y_loadings_lv_' + str(lv),
            pls_result['y_loadings'][:, lv])

#------------------------------------------------------------------------------
# Plot the association between brain and bodily scores
#------------------------------------------------------------------------------

def plot_scores_and_correlations_unicolor(lv,
                                          pls_result,
                                          title,
                                          clinical_scores,
                                          path_fig,
                                          column_name):

    plt.figure(figsize = (5,5))
    plt.scatter(range(1, len(pls_result.varexp) + 1),
                pls_result.varexp,
                color = 'gray')
    plt.savefig(path_figures  + f'GAM_biobank_scatter_PLS_{sex_label}_lv_' + str(lv) + '.svg',
                format = 'svg')
    plt.title(title)
    plt.xlabel('Latent variables')
    plt.ylabel('Variance Explained')

    # Calculate and print singular values
    singvals = pls_result["singvals"] ** 2 / np.sum(pls_result["singvals"] ** 2)
    print(f'Singular values for latent variable {lv}: {singvals[lv]:.4f}')

    # Plot score correlation
    plt.figure(figsize = (5, 5))
    plt.title(title)

    sns.regplot(x = pls_result['x_scores'][:, lv],
                y = pls_result['y_scores'][:, lv],
                scatter = False)
    sns.scatterplot(x = pls_result['x_scores'][:, lv],
                    y = pls_result['y_scores'][:, lv],
                    c = clinical_scores,
                    s = 30,
                    cmap = 'coolwarm',
                    vmin = 0,
                    vmax = 1,
                    edgecolor='black',
                    linewidth = 0.5)

    plt.xlabel('X scores')
    plt.ylabel('Y scores')
    plt.savefig(path_figures + 'GAM_biobank_' + title + f'_{sex_label}_lv_' + str(lv) +'.svg',
                format = 'svg')
    plt.tight_layout()

    # Calculate and print score correlations
    score_correlation_spearmanr = spearmanr(pls_result['x_scores'][:, lv],
                                            pls_result['y_scores'][:, lv])
    score_correlation_pearsonr = pearsonr(pls_result['x_scores'][:, lv],
                                          pls_result['y_scores'][:, lv])

    print(f'x-score and y-score Spearman correlation for latent variable {lv}: \
          {score_correlation_spearmanr.correlation:.4f}')
    print(f'x-score and y-score Pearson correlation for latent variable {lv}: \
          {score_correlation_pearsonr[0]:.4f}')

for behavior_ind in range(np.size(Y, axis = 1)):
    for lv in range(1):
        title = f'Latent Variable {lv + 1}'
        column_name = (bio_names[behavior_ind])
        colors = (Y[:,behavior_ind] - min(Y[:,behavior_ind])) / (max(Y[:,behavior_ind]) - min(Y[:,behavior_ind]))
        plot_scores_and_correlations_unicolor(lv,
                                              pls_result,
                                              bio_names[behavior_ind],
                                              colors,
                                              path_results,
                                              bio_names)
        
#------------------------------------------------------------------------------
# Plot loadings
#------------------------------------------------------------------------------

cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))

def plot_loading_bar(lv, pls_result, combined_columns, vmin_val, vmax_val):
    """
    Create a horizontal bar plot of loadings, ordered by magnitude, and mark significance.
    Significance is determined based on the confidence interval crossing zero.
    """
    err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1] -
           pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
    values = pls_result.y_loadings[:, lv]

    # Determine significance: if CI crosses zero, it's non-significant
    significance = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1] * \
                    pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) > 0

    # Sort values, errors, and significance by loading magnitude
    sorted_indices = np.argsort(values)
    values_sorted = values[sorted_indices]
    err_sorted = err[sorted_indices]
    labels_sorted = np.array(combined_columns)[sorted_indices]
    significance_sorted = significance[sorted_indices]

    # Plot the loadings
    plt.figure(figsize = (10, 10))
    bars = plt.barh(labels_sorted,
                    values_sorted,
                    xerr = err_sorted,
                    color=[colors[90] if sig else 'gray' for sig in significance_sorted])

    plt.xlabel('x-loading')
    plt.ylabel('Behavioral Measure')
    plt.title(f'Latent Variable {lv + 1} Loadings')

    # Highlight significant loadings by making them bold
    for bar, sig in zip(bars, significance_sorted):
        if sig:
            bar.set_linewidth(1.5)
            bar.set_edgecolor('black')
    ax = plt.gca()
    x_ticks = np.linspace(vmin_val, vmax_val, num = 5)
    ax.set_xticks(x_ticks)
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.savefig(path_figures + f'GAM_biobank_PLS_bars_{sex_label}_lv_' + str(lv) + '.svg',
                format = 'svg')
    plt.show()

plot_loading_bar(0, pls_result, bio_names, -0.5, 0.5)


#------------------------------------------------------------------------------
# What is happening on the brain side ?
#------------------------------------------------------------------------------

xload = behavioral_pls(Y,
                       X,
                       n_boot = 1000,
                       n_perm = 0,
                       test_split = 0,
                       seed = 0)

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

brain_names = np.array(brain_names)

for lv in range(3):
    loadings = xload.y_loadings[:, lv]
    np.save(path_results + f'GAM_ukbb_{sex_label}_brain_loadings_lv_{lv}.npy', loadings)
    variances = []
    for label, sl in feature_slices.items():
        block_values = loadings[sl]
        if len(block_values) > 0:
            avg_abs_loading = np.mean(np.abs(block_values))
        else:
            avg_abs_loading = np.nan
        variances.append(avg_abs_loading)

    # Plot
    plt.figure(figsize = (8, 5))
    plt.bar(feature_slices.keys(), variances)
    plt.xlabel("Feature Type")
    plt.ylabel("Mean Absolute Loading")
    plt.title(f"LV-{lv + 1} | Loadings by Feature Type")
    plt.xticks(rotation = 45)
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
    plt.tight_layout()
    plt.savefig(path_figures + f'GAM_biobank_brain_loadings_{sex_label}_by_feature_block_lv_{lv}.svg')
    plt.show()

feature_labels = list(feature_slices.keys())
x_positions = np.arange(len(feature_labels))
bar_width = 0.6

for lv in range(3):
    loadings = xload.y_loadings[:, lv]

    plt.figure(figsize = (10, 6))
    for i, label in enumerate(feature_labels):
        block_values = loadings[feature_slices[label]]

        # Bar: average loading
        plt.bar(x_positions[i], np.mean(block_values), width = bar_width, alpha = 0.2,
                color = 'gray', zorder = 1)

        # Dots: individual region loadings
        for val in block_values:
            plt.scatter(x_positions[i] + np.random.uniform(-bar_width/4, bar_width/4),
                        val,
                        color = 'gray',
                        s = 15,
                        edgecolor = 'k',
                        linewidth = 0.2,
                        zorder = 2)

    plt.xticks(x_positions, feature_labels, rotation = 45, ha = 'right')
    plt.ylabel(f"PLS Loading (LV{lv + 1})")
    plt.title(f"Block-wise Loadings with Region Dots (LV{lv + 1})")
    plt.tight_layout()
    plt.savefig(f"{path_figures}GAM_PLS_lv_{lv}_biobank_feature_block_dots_{sex_label}_brain_loadings.svg",
                format='svg')
    plt.show()

#------------------------------------------------------------------------------
# END
