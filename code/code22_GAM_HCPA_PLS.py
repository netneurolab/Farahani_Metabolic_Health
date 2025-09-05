"""

HCP-A - age corrected using GAMLSS models

################################## Males ######################################

number of included subjects:    268
number of brain features:       3289
number of clinical measures:    28

PLS results:

    Singular values for latent variable 0: 0.3491
    x-score and y-score Spearman correlation for latent variable 0:           0.5127
    x-score and y-score Pearson correlation for latent variable 0:           0.5184
    p-value = 9.99000999e-04
    
    Singular values for latent variable 1: 0.1320
    x-score and y-score Spearman correlation for latent variable 1:           0.3968
    x-score and y-score Pearson correlation for latent variable 1:           0.4301
    p-value = 9.99000999e-04
    
    Singular values for latent variable 2: 0.0981
    x-score and y-score Spearman correlation for latent variable 2:           0.3836
    x-score and y-score Pearson correlation for latent variable 2:           0.5009
    p-value = 1.99800200e-03

    lv = 3 - p-value = 2.69730270e-02

################################# Females #####################################

number of included subjects:    329
number of brain features:       3289
number of clinical measures:    28

PLS results:

    Singular values for latent variable 0: 0.4615
    x-score and y-score Spearman correlation for latent variable 0:           0.4589
    x-score and y-score Pearson correlation for latent variable 0:           0.5099
    p-value = 9.99000999e-04

    Singular values for latent variable 1: 0.1202
    x-score and y-score Spearman correlation for latent variable 1:           0.3330
    x-score and y-score Pearson correlation for latent variable 1:           0.3889
    p-value = 9.99000999e-04

    Singular values for latent variable 2: 0.0556
    x-score and y-score Spearman correlation for latent variable 2:           0.3856
    x-score and y-score Pearson correlation for latent variable 2:           0.3788
    p-value = 6.09390609e-02

    lv = 3 - p-value = 8.09190809e-02

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import seaborn as sns
from pyls import behavioral_pls
import matplotlib.pyplot as plt
from globals import path_results, path_figures
from scipy.stats import zscore, pearsonr, spearmanr
from functions import save_parcellated_data_in_SchaeferTian_forVis

#------------------------------------------------------------------------------
# load cleaned data (using GAMLSS models) - and also add "age" to the biomarker side
#------------------------------------------------------------------------------

sex_label = 'Female'

Y1 = np.load(path_results + f'bio_data_PLS_{sex_label}_clean.npy').T
Y2 = np.load(path_results + f'bio_data_PLS_{sex_label}.npy').T[:1, :]/12
Y = np.concatenate((Y2.T, Y1), axis = 1)
X = np.load(path_results + f'brain_data_PLS_{sex_label}_clean.npy').T

#names = np.load(path_results + f'names_bio_data_PLS_{sex_label}.npy', allow_pickle = True)
name_write = [
    "Age","LH", "FSH", "Estradiol", "Testosterone", "ALT", "Cholesterol", "Triglycerides",
    "Urea", "Protein", "CO2", "Calcium", "Bilirubin", "Albumin", "HDL", "Liver_ALK", "HbA1c",
    "Insulin", "LDL", "Vitamin_D", "AST", "Glucose", "Chloride", "Creatinine", "Potassium",
    "Sodium", "MAP", "BMI"
]

#------------------------------------------------------------------------------
# Handle nan values and then run the PLS model
#------------------------------------------------------------------------------

# Fill NaNs
nan_mask = np.isnan(X)
col_means = np.nanmean(X, axis = 0)
X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

# Standardize
X = zscore(X, axis = 0)
Y = zscore(Y, axis = 0)

nspins = 1000
num_subjects = (len(X))

spins = np.zeros((num_subjects, nspins))
for spin_ind in range(nspins):
    spins[:,spin_ind] = np.random.permutation(range(0, num_subjects))

spins = spins.astype(int)
np.save(path_results + f'GAM_spin_PLS_{sex_label}.npy', spins)

# Use the already created spin (for the main PLS)
spins = np.load(path_results + f'GAM_spin_PLS_{sex_label}.npy')

pls_result = behavioral_pls(X,
                            Y,
                            n_boot = nspins,
                            n_perm = nspins,
                            permsamples = spins,
                            test_split = 0,
                            seed = 0)
for lv in range(3):
    np.save(path_results + f'GAM_{sex_label}_score_x_lv_' + str(lv), pls_result['x_scores'][:, lv])
    np.save(path_results + f'GAM_{sex_label}_score_y_lv_' + str(lv), pls_result['y_scores'][:, lv])

#------------------------------------------------------------------------------
# Plot scatter plot (scores)
#------------------------------------------------------------------------------

def plot_scores_and_correlations_unicolor(lv,
                                          pls_result,
                                          title,
                                          clinical_scores,
                                          path_fig,
                                          column_name):

    plt.figure(figsize = (5, 5))
    plt.scatter(range(1, len(pls_result.varexp) + 1),
                pls_result.varexp,
                color = 'gray')
    plt.savefig(path_figures  + f'GAM_scatter_PLS_{sex_label}_lv_' + str(lv) + '.svg', format = 'svg')
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
    plt.savefig(path_figures + 'GAM_' + title + f'_{sex_label}_lv_' + str(lv) +'.svg', format = 'svg')
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
    for lv in range(3):
        title = f'Latent Variable {lv + 1}'
        column_name = (name_write[behavior_ind])
        colors = (Y[:,behavior_ind] - min(Y[:,behavior_ind])) / (max(Y[:,behavior_ind]) - min(Y[:,behavior_ind]))
        plot_scores_and_correlations_unicolor(lv,
                                              pls_result,
                                              name_write[behavior_ind],
                                              colors,
                                              path_results,
                                              name_write)

#------------------------------------------------------------------------------
# Loading plots
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
    plt.savefig(path_figures + 'GAM_PLS_bio_bars_lv_' + str(lv) + f'_{sex_label}.svg',
                format = 'svg')
    plt.show()

plot_loading_bar(0, pls_result, name_write, -0.5, 0.5)
plot_loading_bar(1, pls_result, name_write, -0.5, 0.5)
plot_loading_bar(2, pls_result, name_write, -0.5, 0.5)

#------------------------------------------------------------------------------
# Flip X and Y in PLS to get x-loading confidence intervals
#------------------------------------------------------------------------------
nnodes = 400
xload = behavioral_pls(Y,
                       X,
                       n_boot = 1000,
                       n_perm = 0,
                       test_split = 0,
                       seed = 0)

# Save WMH loadings
for lv in range(3):
    loadings_cortex = xload.y_loadings[:, lv]
    loadings_wmh = loadings_cortex[:9]
    np.save(path_results + f'GAM_loadings_WMH_{sex_label}_lv_' + str(lv) + '.npy', loadings_wmh)

# Save cortical loading maps
cortical_features = ["Perfusion", "Arrival", "Thickness", "Myelin", "FA", "MD", "FC", "SC"]
for lv in range(3):
    loadings_cortex = xload.y_loadings[:, lv]
    for i, feature in enumerate(cortical_features):
        start = 9 + i * nnodes
        end = 9 + (i + 1) * nnodes
        save_parcellated_data_in_SchaeferTian_forVis(
            loadings_cortex[start:end],
            'cortex', 'X',
            path_results,
            f'GAM_jhu_loadings_{feature.lower()}_lv{lv}_{sex_label}')
        np.save(path_results + 'GAM_loadings_CORTEX_' + cortical_features[i] + f'_{sex_label}_lv_' + str(lv) + '.npy', loadings_cortex[start:end])

# Save JHU loadings
names_jhu = ["Perfusion", "Arrival", "FA", "MD"]
for lv in range(3):
    for i in range(4):
        start = 3209 + i * 20
        end = 3209 + (i + 1) * 20
        loadings_jhu = xload.y_loadings[start: end, lv]
        np.save(path_results + 'GAM_loadings_WM_' + names_jhu[i] + f'_{sex_label}_lv_' + str(lv) + '.npy', loadings_jhu)

#------------------------------------------------------------------------------

# Parameters
n_cortical = 8
n_wm = 4
nnodes_cort = 400
nnodes_wm = 20
features_cortex = ["Perfusion", "Arrival", "Thickness", "Myelin", "FA", "MD", "FC", "SC"]
features_wm = ["Perfusion_WM", "Arrival_WM", "FA_WM", "MD_WM"]
all_features = ["WMH"] + features_cortex + features_wm 
bar_width = 0.6

for lv in range(2):
    # Loadings (cortical and JHU WM tracts)
    loadings_cortex = xload.y_loadings[9:, lv]

    # WMH loadings
    loadings_wmh =  xload.y_loadings[:9, lv]

    # Extract cortical loadings (shape: 400x8)
    loadings_mat_cortex = np.zeros((nnodes_cort, n_cortical))
    for i in range(n_cortical):
        loadings_mat_cortex[:, i] = loadings_cortex[i * nnodes_cort:(i + 1) * nnodes_cort]

    # Extract WM loadings (shape: 20x4)
    loadings_mat_wm = np.zeros((nnodes_wm, n_wm))
    for i in range(n_wm):
        loadings_mat_wm[:, i] = loadings_cortex[3200 + i * nnodes_wm: 3200 + (i + 1) * nnodes_wm]

    # Compute mean absolute loadings for bars
    mean_abs_loadings = np.concatenate([
        [np.mean(loadings_wmh)],
        np.mean((loadings_mat_cortex), axis = 0),
        np.mean((loadings_mat_wm), axis = 0)
    ])

    plt.figure(figsize = (10, 6))
    x_positions = np.arange(len(all_features))
    plt.bar(x_positions, mean_abs_loadings,
            width=bar_width, alpha = 0.3, color = 'gray', label = "Mean |Loading|")
    for val in loadings_wmh:
        plt.scatter(x_positions[0] + np.random.uniform(-bar_width/4, bar_width/4), val,
                    s = 12, color = 'black', alpha = 0.6, zorder = 2)
    for i in range(n_cortical):
        for val in loadings_mat_cortex[:, i]:
            plt.scatter(x_positions[i + 1] + np.random.uniform(-bar_width/4, bar_width/4), val,
                        s = 12, color = 'black', alpha = 0.6, zorder = 2)
    for i in range(n_wm):
        for val in loadings_mat_wm[:, i]:
            plt.scatter(x_positions[n_cortical + i + 1] + np.random.uniform(-bar_width/4, bar_width/4), val,
                        s = 12, color = 'black', alpha = 0.6, zorder = 2)
    plt.xticks(x_positions, all_features, rotation = 45, ha = 'right')
    plt.title(f"Loadings Distribution for LV {lv}")
    plt.ylabel("PLS Loading")
    plt.tight_layout()
    plt.savefig(path_figures + 'GAM_PLS_brain_bars_lv_' + str(lv) + f'_{sex_label}.svg',
                format = 'svg')
    plt.show()

#------------------------------------------------------------------------------
# END
