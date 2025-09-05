"""

UK Biobank - GAMLSS models are used to correct age-effects.

lv = 0 - metabolic axis!

n_permutation  = 1,000

Female:
behavior                    spearman_r    p_perm       num_missing
fluid_intelligence_score   -0.101643      0.000999     159

Male:
behavior                    spearman_r    p_perm       num_missing
fluid_intelligence_score    0.067517      0.016983     121

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from globals import path_results, path_figures

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

lv = 0             # Under study latent variable
sex_label = 'Male' # This can be Female or Male!
cleaned = 1        # Use corrected behavioral data:1; use behavioral data as is: 0

if sex_label == 'Female':
    sex_num = 0
elif sex_label == 'Male':
    sex_num = 1

x_score =  np.load(path_results + f'GAM_biobank_{sex_label}_score_x_lv_' + str(lv) + '.npy')
y_score = np.load(path_results + f'GAM_biobank_{sex_label}_score_y_lv_' + str(lv) + '.npy')

if cleaned == 1:
    behavior = np.load(path_results + f'biobank_behavior_{sex_label}_clean_centercorrected.npy')
    behavior_name = np.load(path_results + f'biobank_names_behavior_{sex_label}_clean_centercorrected.npy',
                            allow_pickle = True)
if cleaned == 0:
    behavior = np.load(path_results + f'biobank_behavior_{sex_label}_not_clean.npy')
    behavior_name = np.load(path_results + f'biobank_names_behavior_{sex_label}_not_clean.npy',
                            allow_pickle = True)

behavior = behavior[:, behavior_name == 'fluid_intelligence_score_20016-2.0']
behavior_name = 'fluid_intelligence_score'

# IQR-based outlier removal
def remove_outliers_iqr(x, thresh = 3.0):
    """Mask outliers using IQR rule; returns mask of valid values."""
    x = np.asarray(x)
    if np.all(np.isnan(x)):
        return np.full_like(x, False, dtype = bool)
    q1 = np.nanpercentile(x, 5)
    q3 = np.nanpercentile(x, 95)
    iqr = q3 - q1
    lower = q1 - thresh * iqr
    upper = q3 + thresh * iqr
    return (x >= lower) & (x <= upper)

# Apply outlier mask to each variable in the behavior matrix
for i in range(behavior.shape[1]):
    mask = remove_outliers_iqr(behavior[:, i])
    behavior[~mask, i] = np.nan  # Mark outliers as NaN

###############################################################################
############################## RELATE TO BEHAVIOR #############################
###############################################################################

num_missing = np.sum(np.isnan(behavior), axis = 0)

corr = np.full(len(behavior.T), np.nan)
for i in range(len(behavior.T)):
    corr[i] = spearmanr(behavior[:, i], x_score, nan_policy = 'omit')[0]
    
n_perm = 1000
perm_corrs = np.full((len(behavior.T), n_perm), np.nan)
for p in range(n_perm):
    permuted_scores = np.random.permutation(x_score)
    for i in range(len(behavior.T)):
        r = spearmanr(behavior[:, i], permuted_scores, nan_policy = 'omit')[0]
        perm_corrs[i, p] = r
    print(p)

#------------------------------------------------------------------------------
# Compute empirical p-values     
#------------------------------------------------------------------------------

def get_perm_p(emp, null):
    return (1 + sum(abs(null - np.mean(null))
                    > abs(emp - np.mean(null)))) / (len(null) + 1)

pvals = np.full(len(behavior.T), np.nan)
for i in range(len(behavior.T)):
    real_r = corr[i]
    null_dist = perm_corrs[i, :]
    pvals[i] = get_perm_p(real_r,perm_corrs[i, :].flatten() )

#------------------------------------------------------------------------------
# Report or save
#------------------------------------------------------------------------------

df_results = pd.DataFrame({
    'behavior': behavior_name,
    'spearman_r': corr,
    'p_perm': pvals,
    'num_missing': num_missing
})

if cleaned == 1:
    df_results.to_csv(path_results + f'GAM_biobank_{sex_label}_behavior_pls{lv}_correlation_results.csv', index = False)
if cleaned == 0:
    df_results.to_csv(path_results + f'GAM_biobank_{sex_label}_behavior_pls{lv}_correlation_results_not_clean.csv', index = False)

# Print significant results (e.g., p < 0.05)
significant_results = df_results[df_results['p_perm'] < 0.05]
print(significant_results)

#------------------------------------------------------------------------------
# Visualize the scatter plots
#------------------------------------------------------------------------------

for idx, row in significant_results.iterrows():
    biomarker_name_ind = row['behavior']
    rho = row['spearman_r']
    missing = int(row['num_missing'])

    n_idx = np.where(behavior_name == biomarker_name_ind)[0][0]
    x = behavior[:, n_idx]
    y = x_score

    if np.all(np.isnan(x)):
        print(f"⚠️ Skipped {biomarker_name_ind}: all values are NaN.")
        continue
    # Skip if more than half the values are missing
    if missing > behavior.shape[0] / 2:
        print(f"⚠️ Skipped {biomarker_name_ind}: more than 50% missing.")
        continue

    mask_valid = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[mask_valid]
    y_valid = y[mask_valid]

    # Fit linear regression line
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    reg_line = slope * x_valid + intercept

    plt.figure(figsize = (5, 5))
    plt.scatter(x_valid, y_valid, s = 10, color = 'silver')
    plt.plot(x_valid, reg_line, color = 'firebrick', linewidth = 2)

    plt.title(f"{biomarker_name_ind}\nρ = {rho:.2f}, Missing = {missing}")
    plt.xlabel(biomarker_name_ind)
    plt.ylabel("x_score")
    plt.tight_layout()

    # filename - not to produce any error on my system
    safe_name = re.sub(r'[^\w\-_\.]', '_', biomarker_name_ind)[:100]
    
    if cleaned == 1:
        save_path = path_figures + f'GAM_UKBB_{sex_label}_scatter_{safe_name}_lv_{lv}.svg'
    if cleaned == 0:
        save_path = path_figures + f'GAM_UKBB_{sex_label}_scatter_{safe_name}_lv_{lv}_not_clean.svg'
    plt.savefig(save_path, format = 'svg')
    plt.show()

#------------------------------------------------------------------------------
# END
