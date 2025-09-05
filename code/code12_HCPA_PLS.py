"""

HCP-A

################################## Males ######################################

number of included subjects:    268
number of brain features:       3289
number of clinical measures:    28

Number of missing values replaced with median per column:
    lh (LH blood test result)                              2
    fsh (FSH blood test result)                            2
    festrs (Hormonal Measures Female Estradiol Results)    2
    rsptest_no (Blood value - Testosterone (ng/dL))        2
    a1crs (HbA1c Results)                                  1
    MAP (nan)                                              2

Number of missing values replaced with median per column - brain:
    WMH_1    63
    WMH_2    63
    WMH_3    63
    WMH_4    63
    WMH_5    63
    WMH_6    63
    WMH_7    63
    WMH_8    63
    WMH_9    63

PLS results:

    Singular values for latent variable 0: 0.6674
    x-score and y-score Spearman correlation for latent variable 0:           0.7663
    x-score and y-score Pearson correlation for latent variable 0:           0.7461
    p-value = 9.99000999e-04

    Singular values for latent variable 1: 0.1283
    x-score and y-score Spearman correlation for latent variable 1:           0.5137
    x-score and y-score Pearson correlation for latent variable 1:           0.5097
    p-value = 9.99000999e-04

    Singular values for latent variable 2: 0.0472
    x-score and y-score Spearman correlation for latent variable 2:           0.3777
    x-score and y-score Pearson correlation for latent variable 2:           0.4203
    p-value = 9.99000999e-04

    lv = 3 - p-value = 2.21778222e-01

################################# Females #####################################

number of included subjects:    329
number of brain features:       3289
number of clinical measures:    28

Number of missing values replaced with median per column:
    lh (LH blood test result)                              1
    fsh (FSH blood test result)                            1
    festrs (Hormonal Measures Female Estradiol Results)    1
    rsptest_no (Blood value - Testosterone (ng/dL))        1
    a1crs (HbA1c Results)                                  3
    MAP (nan)                                              4
    bmi                                                    2

Number of missing values replaced with median per column - brain:
    WMH_1     62
    WMH_2     62
    WMH_3     62
    WMH_4     62
    WMH_5     62
    WMH_6     62
    WMH_7     62
    WMH_8     62
    WMH_9     62
    FA_277     1
    MD_277     1


PLS results:

    Singular values for latent variable 0: 0.7190
    x-score and y-score Spearman correlation for latent variable 0:           0.6879
    x-score and y-score Pearson correlation for latent variable 0:           0.6954
    p-value = 9.99000999e-04

    Singular values for latent variable 1: 0.1414
    x-score and y-score Spearman correlation for latent variable 1:           0.4545
    x-score and y-score Pearson correlation for latent variable 1:           0.5139
    p-value = 9.99000999e-04

    Singular values for latent variable 2: 0.0255
    x-score and y-score Spearman correlation for latent variable 2:           0.3484
    x-score and y-score Pearson correlation for latent variable 2:           0.3673
    p-value = 6.99300699e-03

    lv = 3 - p-value =  2.75724276e-01

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
from pyls import behavioral_pls
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr, spearmanr
from globals import path_results, path_figures, path_info_sub
from functions import save_parcellated_data_in_SchaeferTian_forVis

#------------------------------------------------------------------------------
# Do you want to perform PLS for female or male group?
#------------------------------------------------------------------------------

sex_label = 'Male' # This can be Female or Male!

if sex_label == 'Female':
    sex_num = 0
elif sex_label == 'Male':
    sex_num = 1
    
#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

cortex_data = np.load(path_results + 'data_merged.npy')     # WMH and cortical features
jhu_data = np.load(path_results    + 'data_merged_JHU.npy') # JHU WM tracts

brain_data = np.concatenate((cortex_data, jhu_data), axis = 0)
num_subjects = brain_data.shape[1]

#------------------------------------------------------------------------------
# Load blood test + demographics data and clean them
#------------------------------------------------------------------------------

finalDF = pd.read_csv(path_results + 'finalDF.csv')
finalDF['bmi'] = (finalDF["weight_std (Weight - Standard Unit)"] * 0.453592)/ \
    (finalDF["vtl007 (Height in inches)"]*(finalDF["vtl007 (Height in inches)"])*0.0254*0.0254)

# Filter the column to exclude values equal to 99999/0999
valid_values = finalDF.loc[(finalDF['friedewald_ldl (Friedewald LDL Cholesterol: )'] != 99999) &
                           (finalDF['friedewald_ldl (Friedewald LDL Cholesterol: )'] != 9999),
                           'friedewald_ldl (Friedewald LDL Cholesterol: )']

# Calculate the median of the valid values
median_value = valid_values.median()

# Replace the 99999/9999 values with the calculated median
finalDF.loc[(finalDF['friedewald_ldl (Friedewald LDL Cholesterol: )'] == 99999) |
            (finalDF['friedewald_ldl (Friedewald LDL Cholesterol: )'] == 9999),
            'friedewald_ldl (Friedewald LDL Cholesterol: )'] = median_value

# Filter the column to exclude values equal to 9999
valid_values_2 = finalDF.loc[finalDF['a1crs (HbA1c Results)'] != 9999,
                           'a1crs (HbA1c Results)']

# Calculate the median of the valid values
median_value_2 = valid_values_2.median()

# Replace the 9999 values with the calculated median
finalDF.loc[finalDF['a1crs (HbA1c Results)'] == 9999,
            'a1crs (HbA1c Results)'] = median_value_2

columns_of_interest = [
    "interview_age", "lh (LH blood test result)", "fsh (FSH blood test result)",
    "festrs (Hormonal Measures Female Estradiol Results)",
    "rsptest_no (Blood value - Testosterone (ng/dL))",
    "ls_alt (16. ALT(SGPT)(U/L))", "rsptc_no (Blood value - Total Cholesterol (mg/dL))",
    "rsptrig_no (Blood value - Triglycerides (mg/dL))", "ls_ureanitrogen (10. Urea Nitrogen(mg/dL))",
    "ls_totprotein (11. Total Protein(g/dL))", "ls_co2 (7. CO2 Content(mmol/L))",
    "ls_calcium (17. Calcium(mg/dL))", "ls_bilirubin (13. Bilirubin, Total(mg/dL))",
    "ls_albumin (12. Albumin(g/dL))", "laba5 (Hdl Cholesterol (Mg/Dl))",
    "bp1_alk (Liver ALK Phos)", "a1crs (HbA1c Results)", "insomm (Insulin: Comments)",
    "friedewald_ldl (Friedewald LDL Cholesterol: )", "vitdlev (25- vitamin D level (ng/mL))",
    "ls_ast (15. AST(SGOT)(U/L))", "glucose (Glucose)", "chloride (Chloride)",
    "creatinine (Creatinine)", "potassium (Potassium)", "sodium (Sodium)",
    "MAP (nan)", "bmi"
]
name_write = [
    "Age", "LH", "FSH", "Estradiol", "Testosterone", "ALT", "Cholesterol", "Triglycerides",
    "Urea", "Protein", "CO2", "Calcium", "Bilirubin", "Albumin", "HDL", "Liver_ALK", "HbA1c",
    "Insulin", "LDL", "Vitamin_D", "AST", "Glucose", "Chloride", "Creatinine", "Potassium",
    "Sodium", "MAP", "BMI"
]

# Add sex as a variable of interest
df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
age = np.array(df['interview_age'])/12
df['sex'] = df['sex'].map({'F': 0, 'M': 1})
sex = np.array(df['sex'])

# Filter to include only female/male subjects first
sex_mask = (sex == sex_num)
beh_data = finalDF[sex_mask].reset_index(drop = True)
beh_data = beh_data[columns_of_interest]

sex = sex[sex_mask]
brain_data = brain_data[:, sex_mask]

# Then count NaNs per female subject
nan_counts = beh_data.isna().sum(axis = 1)

# Create a mask to exclude subjects with >20 missing values
mask_nan = nan_counts <= 20

# Apply missing value filter
beh_data = beh_data[mask_nan].reset_index(drop = True)
sex = sex[mask_nan]
age = age[sex_mask][mask_nan]
brain_data = brain_data[:, mask_nan]

# Print number of missing values per column (before imputation)
nan_per_column = beh_data.isna().sum()
print("Number of missing values replaced with median per column:")
print(nan_per_column[nan_per_column > 0])

# Median imputation - missing values for clinical measures
beh_data = beh_data.apply(lambda x: x.fillna(np.nanmedian(x)), axis = 0)

# Convert to NumPy array
behaviour_data_array = beh_data.to_numpy()

names = beh_data.columns
data = brain_data
data = data.T

# Provide brain feature names
brain_names = (
    [f'WMH_{i+1}' for i in range(9)] +
    [f'Perfusion_{i+1}' for i in range(400)] +
    [f'Arrival_{i+1}' for i in range(400)] +
    [f'Thickness_{i+1}' for i in range(400)] +
    [f'Myelin_{i+1}' for i in range(400)] +
    [f'FA_{i+1}' for i in range(400)] +
    [f'MD_{i+1}' for i in range(400)] +
    [f'FC_{i+1}' for i in range(400)] +
    [f'SC_{i+1}' for i in range(400)] +
    [f'Perfusion_WM_{i+1}' for i in range(20)] +
    [f'Arrival_WM_{i+1}' for i in range(20)] +
    [f'FA_WM_{i+1}' for i in range(20)] +
    [f'MD_WM_{i+1}' for i in range(20)]
    )

# Convert to DataFrame to inspect NaNs per column
data_df = pd.DataFrame(data, columns = brain_names[:data.shape[1]])

# Count NaNs per column BEFORE imputation
nan_per_column_brain = data_df.isna().sum()

# Print number of NaNs that will be replaced
print("Number of missing values replaced with median per column - brain:")
print(nan_per_column_brain[nan_per_column_brain > 0])

# Median imputation - missing values for brain measures
data = np.where(np.isnan(data), np.nanmedian(data, axis = 0), data)

#------------------------------------------------------------------------------
# Save data - for further analysis later on
#------------------------------------------------------------------------------

np.save(path_results + f'brain_data_PLS_{sex_label}.npy', data)
np.save(path_results + f'bio_data_PLS_{sex_label}.npy', behaviour_data_array)
np.save(path_results + f'names_brain_data_PLS_{sex_label}.npy', brain_names)
np.save(path_results + f'names_bio_data_PLS_{sex_label}.npy', names)

#------------------------------------------------------------------------------
#                            PLS analysis - main
#------------------------------------------------------------------------------

X = zscore(data, axis = 0)
Y = zscore(behaviour_data_array, axis = 0)

nspins = 1000
num_subjects = len(X)

spins = np.zeros((num_subjects, nspins))
for spin_ind in range(nspins):
    spins[:,spin_ind] = np.random.permutation(range(0, num_subjects))

spins = spins.astype(int)
np.save(path_results + f'spin_PLS_{sex_label}.npy', spins)

# Use the already created spin (for the main PLS)
spins = np.load(path_results + f'spin_PLS_{sex_label}.npy')

pls_result = behavioral_pls(X,
                            Y,
                            n_boot = nspins,
                            n_perm = nspins,
                            permsamples = spins,
                            test_split = 0,
                            seed = 0)
for lv in range(3):
    np.save(path_results + f'{sex_label}_score_x_lv_' + str(lv), pls_result['x_scores'][:, lv])
    np.save(path_results + f'{sex_label}_score_y_lv_' + str(lv), pls_result['y_scores'][:, lv])

#------------------------------------------------------------------------------
# plot scatter plot (scores)
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
    plt.savefig(path_figures + f'scatter_PLS_{sex_label}_lv_' + str(lv) + '.svg', format = 'svg')
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
    plt.savefig(path_figures + title + f'_{sex_label}_lv_' + str(lv) + '.svg', format = 'svg')
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
        column_name = (names[behavior_ind])
        colors = (Y[:,behavior_ind] - min(Y[:,behavior_ind])) / (max(Y[:,behavior_ind]) - min(Y[:,behavior_ind]))
        plot_scores_and_correlations_unicolor(lv,
                                              pls_result,
                                              name_write[behavior_ind],
                                              colors,
                                              path_results,
                                              names)
        
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
                    color = [colors[90] if sig else 'gray' for sig in significance_sorted])

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
    plt.savefig(path_figures + 'PLS_bio_bars_lv_' + str(lv) + f'_{sex_label}.svg',
                format = 'svg')
    plt.show()

plot_loading_bar(0, pls_result, names, -0.5, 0.5)
plot_loading_bar(1, pls_result, names, -0.5, 0.5)
plot_loading_bar(2, pls_result, names, -0.5, 0.5)

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
    np.save(path_results + f'loadings_WMH_{sex_label}_lv_' + str(lv) + '.npy', loadings_wmh)
    
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
            f'loadings_{feature}_lv_{lv}_{sex_label}')
        np.save(path_results + 'loadings_CORTEX_' + cortical_features[i] + f'_{sex_label}_lv_' + str(lv) + '.npy', loadings_cortex[start:end])
        
# Save JHU loadings
names_jhu = ["Perfusion", "Arrival", "FA", "MD"]
for lv in range(3):
    for i in range(4):
        start = 3209 + i * 20
        end = 3209 + (i + 1) * 20
        loadings_jhu = xload.y_loadings[start: end, lv]
        np.save(path_results + 'loadings_WM_' + names_jhu[i] + f'_{sex_label}_lv_' + str(lv) + '.npy', loadings_jhu)

#------------------------------------------------------------------------------
# Plot all 13 next to each other (WM and GM data + WMH data)
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

    # Extract cortical loadings (shape: 400 x 8)
    loadings_mat_cortex = np.zeros((nnodes_cort, n_cortical))
    for i in range(n_cortical):
        loadings_mat_cortex[:, i] = loadings_cortex[i * nnodes_cort:(i + 1) * nnodes_cort]

    # Extract WM tract loadings (shape: 20x4)
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
            width=bar_width,
            alpha = 0.3,
            color = 'gray',
            label = "Mean |Loading|")

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
    plt.savefig(path_figures + 'PLS_brain_bars_lv_' + str(lv) + f'_{sex_label}.svg',
                format = 'svg')
    plt.show()

#------------------------------------------------------------------------------
# END
