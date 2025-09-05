"""

UK Biobank

----------------------------------- MALES -------------------------------------

number of subjects:          1431
number of clinical measures: 33
number of brain measures:    490

PLS results:

    Singular values for latent variable 0: 0.7571
    x-score and y-score Spearman correlation for latent variable 0:           0.6373
    x-score and y-score Pearson correlation for latent variable 0:           0.6262
    p = 9.99000999e-04

    Singular values for latent variable 1: 0.1533
    x-score and y-score Spearman correlation for latent variable 1:           0.3363
    x-score and y-score Pearson correlation for latent variable 1:           0.3504
    p = 9.99000999e-04

    Singular values for latent variable 2: 0.0364
    x-score and y-score Spearman correlation for latent variable 2:           0.2261
    x-score and y-score Pearson correlation for latent variable 2:           0.2460
    p = 9.99000999e-04

----------------------------------- FEMALES -----------------------------------

number of subjects:          1582
number of clinical measures: 33
number of brain measures:    490

PLS results:

    Singular values for latent variable 0: 0.7811
    x-score and y-score Spearman correlation for latent variable 0:           0.6067
    x-score and y-score Pearson correlation for latent variable 0:           0.5949

    Singular values for latent variable 1: 0.1493
    x-score and y-score Spearman correlation for latent variable 1:           0.3712
    x-score and y-score Pearson correlation for latent variable 1:           0.3636

    Singular values for latent variable 2: 0.0212
    x-score and y-score Spearman correlation for latent variable 2:           0.1857
    x-score and y-score Pearson correlation for latent variable 2:           0.1927

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import re
import numpy as np
import pandas as pd
import seaborn as sns
from pyls import behavioral_pls
import matplotlib.pyplot as plt
from netneurotools import datasets
from scipy.stats import zscore, spearmanr, pearsonr
from globals import path_results, path_figures, path_ukbiobank
from functions import save_parcellated_data_in_cammun033_forVis

#------------------------------------------------------------------------------
# Load the biomarker + mri imaging data --> clean the columns
#------------------------------------------------------------------------------

sex_label = 'Female' # This can be Female or Male!

if sex_label == 'Female':
    sex_num = 0
elif sex_label == 'Male':
    sex_num = 1

main_data = pd.read_csv(path_ukbiobank + 'merged_biodata_with_cbf.csv')

# Drop all columns ending in "-3.0"
main_data = main_data[[col for col in main_data.columns if not col.endswith('-3.0')]]

#------------------------------------------------------------------------------
# First filter by sex (female/male only)
#------------------------------------------------------------------------------

# Extract sex column and map to binary
main_data['sex_31-0.0'] = main_data['sex_31-0.0'].map({'Female': 0, 'Male': 1})
main_data = main_data[main_data['sex_31-0.0'] == sex_num].reset_index(drop = True)

#------------------------------------------------------------------------------
# Define interesting labels to KEEP for the PLS model
#------------------------------------------------------------------------------

interesting_labels = [
    'waist_circumference_48-2.0',
    'hip_circumference_49-2.0',
    'body_fat_percentage_23099-2.0',
    'potassium_in_urine_30520-0.0',
    'sodium_in_urine_30530-0.0',
    'alkaline_phosphatase_30610-0.0',
    'alanine_aminotransferase_30620-0.0',
    'aspartate_aminotransferase_30650-0.0',
    'urea_30670-0.0',
    'calcium_30680-0.0',
    'cholesterol_30690-0.0',
    'creatinine_30700-0.0',
    'c-reactive_protein_30710-0.0',
    'glucose_30740-0.0',
    'glycated_haemoglobin_hba1c_30750-0.0',
    'hdl_cholesterol_30760-0.0',
    'ldl_direct_30780-0.0',
    'oestradiol_30800-0.0',
    'total_bilirubin_30840-0.0',
    'testosterone_30850-0.0',
    'total_protein_30860-0.0',
    'triglycerides_30870-0.0',
    'vitamin_d_30890-0.0',
    'body_mass_index_bmi_21001-2.0',
    'diastolic_blood_pressure_automated_reading_4079-0.1',
    'systolic_blood_pressure_automated_reading_4080-0.1',
    'age_when_attended_assessment_centre_21003-0.0',
    'age_when_attended_assessment_centre_21003-2.0',
    'body_fat_percentage_23099-0.0',
    'waist_circumference_48-0.0',
    'hip_circumference_49-0.0',
    'body_mass_index_bmi_21001-0.0',
    'diastolic_blood_pressure_automated_reading_4079-0.0',
    'systolic_blood_pressure_automated_reading_4080-0.0']

# Add subject ID and sex columns to keep
columns_to_keep = [col for col in interesting_labels if col in main_data.columns]
main_data = main_data[columns_to_keep + list(main_data.columns[main_data.columns.get_loc(columns_to_keep[-1]) + 1:])].copy()
main_data.reset_index(drop = True, inplace = True)

#------------------------------------------------------------------------------
# Remove usbjects with lots of missing values
# first, romove the subjects who has more than 20 missing values on brain or clinical side 
# second, remove columns with lots of missing values more than 500)
#------------------------------------------------------------------------------

# Split into two parts
first_cols = columns_to_keep
rest_cols = main_data.columns.difference(first_cols)

# Create mask for rows to keep
mask_first = main_data[first_cols].isnull().sum(axis = 1) <= 40
mask_rest = main_data[rest_cols].isnull().sum(axis = 1) <= 40

# Keep only rows that satisfy both conditions
main_data = main_data[mask_first & mask_rest]

# Count NaNs in each column
nan_counts = main_data.isna().sum()

# Filter columns with more than 500 NaNs
columns_with_many_nans = nan_counts[nan_counts > 500]
main_data = main_data.drop(columns = columns_with_many_nans.index)
main_data.reset_index(drop = True, inplace = True)
columns_to_keep = [col for col in interesting_labels if col in main_data.columns]
   
# Devide the data into brain and biomarker sections
data_bio = main_data[columns_to_keep].copy()
data_brain = main_data.drop(columns=columns_to_keep).copy()

data_bio.reset_index(drop = True, inplace = True)
data_brain.reset_index(drop = True, inplace = True)

save_data_bio = np.array(data_bio)
np.save(path_results + f'data_bio_array_{sex_label}_biobank_with_nan.npy', save_data_bio)

#------------------------------------------------------------------------------
# Handle nan values, remove subject id column, convert to numbers, and so on
#------------------------------------------------------------------------------

# Mode imputation - missing values
data_brain = data_brain.fillna(data_brain.mean())
data_bio = data_bio.fillna(data_bio.mean())

data_brain_array = (np.array(data_brain))
brain_names = data_brain.columns # 485 columns


np.save(path_results + f'brain_names_{sex_label}_biobank.npy', brain_names)
np.save(path_results + f'data_brain_array_{sex_label}_biobank.npy', data_brain_array)

data_bio_array = (np.array(data_bio))
bio_names = data_bio.columns # 33 columns

np.save(path_results + f'bio_names_{sex_label}_biobank.npy', bio_names)
np.save(path_results + f'data_bio_array_{sex_label}_biobank.npy', data_bio_array)

#------------------------------------------------------------------------------
# zscore X and Y
#------------------------------------------------------------------------------

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
    np.save(path_results + f'biobank_{sex_label}_score_x_lv_' + str(lv),
            pls_result['x_scores'][:, lv])
    np.save(path_results + f'biobank_{sex_label}_score_y_lv_' + str(lv),
            pls_result['y_scores'][:, lv])
    np.save(path_results + f'biobank_{sex_label}_y_loadings_lv_' + str(lv),
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

    plt.figure(figsize = (5, 5))
    plt.scatter(range(1, len(pls_result.varexp) + 1),
                pls_result.varexp,
                color = 'gray')
    plt.savefig(path_figures  + 'biobank_scatter_PLS_{sex_label}_lv_' + str(lv) + '.svg', format = 'svg')
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
    plt.savefig(path_figures + 'biobank_' + title + f'_{sex_label}_lv_' + str(lv) + '.svg', format = 'svg')
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
        column_name = (interesting_labels[behavior_ind])
        colors = (Y[:,behavior_ind] - min(Y[:,behavior_ind])) / (max(Y[:,behavior_ind]) - min(Y[:,behavior_ind]))
        plot_scores_and_correlations_unicolor(lv,
                                              pls_result,
                                              interesting_labels[behavior_ind],
                                              colors,
                                              path_results,
                                              interesting_labels)

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
    plt.savefig(path_figures + f'biobank_PLS_bars_{sex_label}_lv_' + str(lv) + '.svg',
                format = 'svg')
    plt.show()

plot_loading_bar(0, pls_result, bio_names, -0.5, 0.5)
plot_loading_bar(1, pls_result, bio_names, -0.5, 0.5)
plot_loading_bar(2, pls_result, bio_names, -0.5, 0.5)

#------------------------------------------------------------------------------
# What is happening on the brain side?
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
    np.save(path_results + f'ukbb_{sex_label}_brain_loadings_lv_{lv}.npy', loadings)
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
    plt.savefig(path_figures + f'biobank_brain_loadings_{sex_label}_by_feature_block_lv_{lv}.svg')
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
    plt.savefig(f"{path_figures}PLS_lv_{lv}_feature_block_dots_{sex_label}_brain_loadings.svg",
                format = 'svg')
    plt.show()

#------------------------------------------------------------------------------
# Load Cammoun atlas info & coordinates
#------------------------------------------------------------------------------

# Load and prepare atlas cortex info
scale = "scale033"
cammoun = datasets.fetch_cammoun2012()
info = pd.read_csv(cammoun['info'])
info_cortex = info.query('scale == @scale & structure == "cortex"').copy()

# ✅ FIX: Define merge_key now
info_cortex['merge_key'] = info_cortex['hemisphere'].str[0].str.lower() + '_' + info_cortex['label'].str.lower()

feature_slices = {
    "area_of": slice(0, 62),
    "volume_of": slice(62, 124),
    "mean_thickness_of": slice(124, 186),
}

lv = 1
loadings = xload.y_loadings[:, lv]
brain_names = np.array(brain_names)

for prefix, sl in feature_slices.items():
    matched_data = []

    for i in range(*sl.indices(len(brain_names))):
        name = brain_names[i]
        loading = loadings[i]

        # Remove prefix
        stripped = name.replace(f'{prefix}_', '')

        # Get hemisphere from tail
        if '_left_hemisphere' in stripped:
            region = stripped.replace('_left_hemisphere', '')
            hemisphere = 'l'
        elif '_right_hemisphere' in stripped:
            region = stripped.replace('_right_hemisphere', '')
            hemisphere = 'r'
        else:
            print(f"Skipping {name} — no hemisphere found")
            continue

        # REMOVE trailing numeric suffixes (e.g., _27296-2.0)
        region = re.sub(r'_\d+[-.0-9]*$', '', region)

        merge_key = f"{hemisphere}_{region.lower()}"
        # Find match in cortex info
        match = info_cortex[info_cortex['merge_key'] == merge_key]
        if match.empty:
            print(f"⚠️ No match for: {merge_key}")
            continue

        parcel_id = match['id'].values[0]
        matched_data.append((parcel_id, loading))

    # Initialize empty vector for all cortical regions
    values = np.full(len(info_cortex), np.nan)
    for pid, val in matched_data:
        values[info_cortex['id'].values - 1 == pid - 1] = val

    print(f"\n{prefix} → valid values: {np.count_nonzero(~np.isnan(values))}/{len(values)}")

    # Save for visualization
    save_parcellated_data_in_cammun033_forVis(
        values,
        path_results,
        f'{sex_label}_{prefix.strip().lower()}_lv{lv}')

#------------------------------------------------------------------------------
# END