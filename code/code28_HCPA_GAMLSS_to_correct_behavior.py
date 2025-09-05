"""

HCP-A - age corrected using GAMLSS models

n_permutation = 1,000

lv = 0

Male:
Nothing is significant

Female:
	name	observed_corr	p_value_notcorrected	num_missing
1	nih_fluidcogcomp_unadjusted (Fluid Cognition Composite Score unadjusted)	0.17837064530500032	0.002997002997002997	36
2	nih_totalcogcomp_unadjusted (Total Cognition Composite Score unadjusted)	0.13871474784325008	0.016983016983016984	36

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import re
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from globals import path_results, path_info_sub

#------------------------------------------------------------------------------
# Load and merge data
#------------------------------------------------------------------------------

lv = 0 
sex_label = 'Female' # This can be Female or Male!

if sex_label == 'Female':
    sex_num = 0
elif sex_label == 'Male':
    sex_num = 1
    
df_basic = pd.read_csv(path_info_sub + 'clean_data_info.csv')
df_behavior = pd.read_csv(path_results + 'no_plasma_finalDF_all.csv')

# Merge dataframes based on subject ID
df_all = pd.merge(df_behavior, df_basic, on = 'subjectkey', how = 'inner')

#------------------------------------------------------------------------------
# Same filtering as what is applied to the PLS analysis
#------------------------------------------------------------------------------

# Reload original finalDF to get full variable table
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

finalDF = finalDF[
    [   "subjectkey", 
        "interview_age",
        "lh (LH blood test result)",
        "fsh (FSH blood test result)",
        "festrs (Hormonal Measures Female Estradiol Results)",
        "rsptest_no (Blood value - Testosterone (ng/dL))",
        "ls_alt (16. ALT(SGPT)(U/L))",
        "rsptc_no (Blood value - Total Cholesterol (mg/dL))",
        "rsptrig_no (Blood value - Triglycerides (mg/dL))",
        "ls_ureanitrogen (10. Urea Nitrogen(mg/dL))",
        "ls_totprotein (11. Total Protein(g/dL))",
        "ls_co2 (7. CO2 Content(mmol/L))",
        "ls_calcium (17. Calcium(mg/dL))",
        "ls_bilirubin (13. Bilirubin, Total(mg/dL))",
        "ls_albumin (12. Albumin(g/dL))",
        "laba5 (Hdl Cholesterol (Mg/Dl))",
        "bp1_alk (Liver ALK Phos)",
        "a1crs (HbA1c Results)",
        "insomm (Insulin: Comments)",
        "friedewald_ldl (Friedewald LDL Cholesterol: )",
        "vitdlev (25- vitamin D level (ng/mL))",
        "ls_ast (15. AST(SGOT)(U/L))",
        "glucose (Glucose)",
        "chloride (Chloride)",
        "creatinine (Creatinine)",
        "potassium (Potassium)",
        "sodium (Sodium)",
        "MAP (nan)",
        "bmi"
    ]]

df_basic['sex'] = df_basic['sex'].map({'F': 0, 'M': 1})
sex = np.array(df_basic['sex'])

# Filter to include only female/male subjects first
sex_mask = (sex == sex_num)
beh_data = finalDF[sex_mask].reset_index(drop = True)
sex = sex[sex_mask]

# Then count NaNs per female/male subject
nan_counts = beh_data.isna().sum(axis = 1)

# Create a mask to exclude subjects with >20 missing values
mask_nan = nan_counts <= 20

# Apply missing value filter
beh_data = beh_data[mask_nan].reset_index(drop = True)

#------------------------------------------------------------------------------
# Merge the two dataframes
#------------------------------------------------------------------------------

# Both dataframes have subjectkey as string for matching
beh_data['subjectkey'] = beh_data['subjectkey'].astype(str)
df_all['subjectkey'] = df_all['subjectkey'].astype(str)

# Subset and reorder df_all based on beh_data subjectkey order
merged_data_aligned = df_all.set_index('subjectkey').loc[beh_data['subjectkey']].reset_index()
age_ = merged_data_aligned['interview_age_x']
#------------------------------------------------------------------------------
# Remove columns with all constant values or NaN + constant
#------------------------------------------------------------------------------

composite_scores = [
    'nih_crycogcomp_unadjusted (Crystal Cognition Composite Score unadjusted)',
    'nih_fluidcogcomp_unadjusted (Fluid Cognition Composite Score unadjusted)',
    'nih_totalcogcomp_unadjusted (Total Cognition Composite Score unadjusted)',
    ]
df_cleaned = merged_data_aligned[composite_scores]


df_cleaned.loc[df_cleaned["nih_fluidcogcomp_unadjusted (Fluid Cognition Composite Score unadjusted)"] == 999,
               "nih_fluidcogcomp_unadjusted (Fluid Cognition Composite Score unadjusted)"] = np.nan

# Define extreme/outlier values to check for
extreme_values = [999, 9999, 99999]
# Loop through numeric columns
for col in df_cleaned.columns:
    if df_cleaned[col].dtype.kind in 'iuf':  # numeric types only
        mask = df_cleaned[col].isin(extreme_values)

        if mask.any():
            print(f"⚠️ Column '{col}' contains outlier values at positions: {np.where(mask)[0].tolist()}")

            # Plot with dots (scatter)
            plt.figure(figsize = (6, 2))
            plt.scatter(range(len(df_cleaned[col])), df_cleaned[col],
                        s = 10, color = 'gray', label = 'All')
            plt.scatter(np.where(mask)[0], df_cleaned.loc[mask, col],
                        s = 25, color = 'red', label = 'Outliers')
            plt.title(f"Outliers in column: {col}")
            plt.xlabel('Index')
            plt.ylabel(col)
            plt.legend()
            plt.grid(True)
            plt.show()

            # Ask user whether to replace outliers with NaN
            user_input = input(f"❓ Replace outliers in column '{col}' with NaN? (y/n): ").strip().lower()
            if user_input == 'y':
                df_cleaned.loc[mask, col] = np.nan

biomarkers = df_cleaned.to_numpy()
biomarkers = (np.array(biomarkers))
names_biomarkers = df_cleaned.columns

Y = (biomarkers) # 268 by 167

#------------------------------------------------------------------------------
# Do GAM to correct behavioral measures for age
#------------------------------------------------------------------------------

pandas2ri.activate()
gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')

n_features = Y.shape[1]
n_subjects = Y.shape[0]
residuals_Y = np.full((n_subjects, n_features), np.nan)
x_all = age_/12
for i in range(n_features):
    try:
        y = Y[:, i]
        x = x_all

        # Remove rows with NaN in x or y
        mask_valid = ~np.isnan(x) & ~np.isnan(y)

        # Prepare valid data and transfer to R
        df_r = pd.DataFrame({'x': x[mask_valid], 'y': y[mask_valid]})
        ro.globalenv['df'] = pandas2ri.py2rpy(df_r)

        # Fit model in R
        ro.r('model <- gamlss(y ~ fp(x), sigma.fo = ~ fp(x), data = df, family = NO())')

        # Extract fitted values and compute residuals
        mu_fitted = np.array(ro.r('fitted(model, what = "mu")')).flatten()
        residuals = y[mask_valid] - mu_fitted

        # Store residuals
        residuals_Y[mask_valid, i] = residuals
        print(f"{i}: success")

    except Exception as e:
        print(f"{i}: Failed to fit model – {e}")
        residuals_Y[:, i] = np.nan  # Optionally assign NaN to whole column

# Remove columns with all NaNs (i.e., failed GAMLSS fitting)
valid_columns_mask = ~np.all(np.isnan(residuals_Y), axis = 0)
residuals_Y = residuals_Y[:, valid_columns_mask]
names_biomarkers = names_biomarkers[valid_columns_mask]
np.save(path_results + f'{sex_label}_residuals_Y.npy', residuals_Y)
np.save(path_results + f'{sex_label}_biomarkers.npy', names_biomarkers)

# Replace original Y with residuals
Y = residuals_Y
num_missing = np.sum(np.isnan(Y), axis = 0)

#------------------------------------------------------------------------------
# Compute similarity of loadings with this
#------------------------------------------------------------------------------

x_score = np.load(path_results + f'GAM_{sex_label}_score_x_lv_' + str(lv) + '.npy')
y_score = np.load(path_results + f'GAM_{sex_label}_score_y_lv_' + str(lv) + '.npy')

num_measures = len(names_biomarkers)
correlation_values = np.zeros((num_measures, 1))

for n in range(num_measures):
    x = Y[:, n]
    y = x_score

    # Mask out NaNs in either variable
    mask_valid = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask_valid) > 2:
        rho, _ = spearmanr(x[mask_valid], y[mask_valid])
    else:
        rho = np.nan
    correlation_values[n, 0] = rho

n_permutations = 1000
perm_distribution = np.zeros((num_measures, n_permutations))

# Perform permutation testing
for perm in range(n_permutations):
    permuted_x = np.random.permutation(x_score)
    for n in range(num_measures):
        x = Y[:, n]
        y = permuted_x
        mask_valid = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask_valid) > 2:
            perm_distribution[n, perm] = spearmanr(x[mask_valid], y[mask_valid])[0]
        else:
            perm_distribution[n, perm] = np.nan
    print(perm)

def get_perm_p(emp, null):
    return (1 + sum(abs(null - np.mean(null))
                    > abs(emp - np.mean(null)))) / (len(null) + 1)

# Calculate p-values
p_values = np.zeros((num_measures, 1))
for n in range(num_measures):
    p_values[n] = get_perm_p(correlation_values[n],perm_distribution[n,:].flatten() )
    
# Create a DataFrame for results
df_result = pd.DataFrame({
    'name': names_biomarkers,
    'observed_corr': correlation_values.flatten(),
    'p_value_notcorrected': p_values.flatten(),
    'num_missing': num_missing
})

# Save the results
df_result.to_csv(path_results + f'GAM_{sex_label}_spearman_permutation_results_x_score_lv_' + str(lv) + '.csv',
                 index = False)

# Print significant results (e.g., p < 0.05)
significant_results = df_result[df_result['p_value_notcorrected'] < 0.05]
print(significant_results)

#------------------------------------------------------------------------------
# Visualize the scatter plots
#------------------------------------------------------------------------------

for idx, row in significant_results.iterrows():
    biomarker_name = row['name']
    rho = row['observed_corr']
    missing = int(row['num_missing'])

    n_idx = np.where(names_biomarkers == biomarker_name)[0][0]
    x = Y[:, n_idx]
    y = x_score

    mask_valid = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[mask_valid]
    y_valid = y[mask_valid]

    # Fit linear regression line (least squares)
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    reg_line = slope * x_valid + intercept

    plt.figure(figsize = (5, 5))
    plt.scatter(x_valid, y_valid, s = 10, color = 'silver')
    plt.plot(x_valid, reg_line, color = 'firebrick', linewidth = 2)

    plt.title(f"{biomarker_name}\nρ = {rho:.2f}, Missing = {missing}")
    plt.xlabel(biomarker_name)
    plt.ylabel("x_score")
    plt.tight_layout()

    # filename
    safe_name = re.sub(r'[^\w\-_\.]', '_', biomarker_name)[:100]
    save_path = path_results + f'GAM_{sex_label}_scatter_{safe_name}_lv_{lv}.svg'
    plt.savefig(save_path, format = 'svg')
    plt.show()

#------------------------------------------------------------------------------
# END
