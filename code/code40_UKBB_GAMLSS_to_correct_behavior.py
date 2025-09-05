"""

UK Biobank - GAMLSS models are used to correct age-effects.

Because the analyses target fluid intelligence, the age-2 variable effect was extracted.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.embedded import RRuntimeError
from globals import path_results, path_ukbiobank, path_figures

#------------------------------------------------------------------------------
# Indicate biological sex of interest
#------------------------------------------------------------------------------

sex_label = 'Male' # This can be Female or Male!

if sex_label == 'Female':
    sex_num = 0
elif sex_label == 'Male':
    sex_num = 1

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

main_data = pd.read_csv(path_ukbiobank + 'merged_biodata_with_cbf.csv') # 4606 by 580
behavior = pd.read_csv(path_ukbiobank + 'merged_behavior_with_cbf.csv') # 4606 by 635

# Drop all columns ending in "-3.0"
main_data = main_data[[col for col in main_data.columns if not col.endswith('-3.0')]]
behavior = behavior[[col for col in behavior.columns if not col.endswith('-3.0')]]  # 4606 by 585

# reorder both so that they have the same order of subjects
main_data = main_data.sort_values('eid').reset_index(drop = True) # 4606 by 530
behavior = behavior.sort_values('eid').reset_index(drop = True)   # 4606 by 585

# only keep the last 30 columns of behavior dataframe
behavior = behavior.iloc[:, 535:] # 4606 by 30

interesting_labels = [
    'sleeplessness__insomnia_1200-0.0',
    'sleeplessness__insomnia_1200-2.0',
    'weight_change_compared_with_1_year_ago_2306-2.0',
    'prospective_memory_result_20018-2.0',
    'smoking_status_20116-0.0',
    'smoking_status_20116-2.0',
    'frequency_of_drinking_alcohol_20414-0.0']

for label in interesting_labels:
    if label in behavior.columns:
        unique_vals = behavior[label].unique()
        print(f"{label}: {sorted(unique_vals[~pd.isnull(unique_vals)])}")
    else:
        print(f"⚠️ Column '{label}' not found in behavior DataFrame.")

# Define mapping dictionaries
mappings = {
    'sleeplessness__insomnia_1200-0.0': {
        'Never/rarely': 0,
        'Sometimes': 1,
        'Usually': 2,
        'Prefer not to answer': np.nan
    },
    'sleeplessness__insomnia_1200-2.0': {
        'Never/rarely': 0,
        'Sometimes': 1,
        'Usually': 2,
        'Prefer not to answer': np.nan
    },
    'weight_change_compared_with_1_year_ago_2306-2.0': {
        'Yes - lost weight': 0,
        'No - weigh about the same': 1,
        'Yes - gained weight': 2,
        'Do not know': np.nan,
        'Prefer not to answer': np.nan
    },
    'prospective_memory_result_20018-2.0': {
        'Correct recall on first attempt': 2,
        'Correct recall on second attempt': 1,
        'Instruction not recalled, either skipped or incorrect': 0,
        'Prefer not to answer': np.nan
    },
    'smoking_status_20116-0.0': {
        'Never': 0,
        'Previous': 1,
        'Current': 2,
        'Prefer not to answer': np.nan
    },
    'smoking_status_20116-2.0': {
        'Never': 0,
        'Previous': 1,
        'Current': 2,
        'Prefer not to answer': np.nan
    },
    'frequency_of_drinking_alcohol_20414-0.0': {
        'Never': 0,
        'Monthly or less': 1,
        '2 to 3 times a week': 2,
        '2 to 4 times a month': 3,
        '4 or more times a week': 4,
        'Prefer not to answer': np.nan
    }
}

# Apply mappings
for col, mapping in mappings.items():
    if col in behavior.columns:
        behavior[col] = behavior[col].map(mapping)
    else:
        print(f"⚠️ Column '{col}' not found.")

behavior_numeric = behavior.apply(pd.to_numeric, errors='coerce')

# Drop columns that are entirely NaN (i.e., conversion failed completely)
behavior = behavior_numeric.dropna(axis = 1, how = 'all')

# load dataframe with a column which reports the data acquisition sites
site_data = pd.read_csv(path_ukbiobank + 'data_062025_withimaging.csv')

# Define prefixes to keep (assessment site and subject IDs)
prefixes = ('uk_biobank_assessment_centre_54-2.0',
            'eid')

filtered_site_data = site_data[[col for col in site_data.columns 
                              if col == 'eid' or col.startswith(prefixes)]]
filtered_site_data = filtered_site_data[filtered_site_data['eid'].isin(main_data['eid'])].copy()
filtered_site_data = filtered_site_data.set_index('eid').loc[main_data['eid'].values].reset_index()

#------------------------------------------------------------------------------
# First filter by sex (female only)
#------------------------------------------------------------------------------

# Extract sex column and map to binary
main_data['sex_31-0.0'] = main_data['sex_31-0.0'].map({'Female': 0, 'Male': 1})
female_mask = main_data['sex_31-0.0'] == sex_num

main_data = main_data[female_mask].reset_index(drop = True)
behavior = behavior[female_mask].reset_index(drop = True)
filtered_site_data = filtered_site_data[female_mask].reset_index(drop = True)

#------------------------------------------------------------------------------
# Define interesting labels to KEEP for the PLS model
#------------------------------------------------------------------------------

interesting_labels = [
    'waist_circumference_48-2.0','hip_circumference_49-2.0',
    'body_fat_percentage_23099-2.0','potassium_in_urine_30520-0.0',
    'sodium_in_urine_30530-0.0','alkaline_phosphatase_30610-0.0',
    'alanine_aminotransferase_30620-0.0','aspartate_aminotransferase_30650-0.0',
    'urea_30670-0.0','calcium_30680-0.0','cholesterol_30690-0.0','creatinine_30700-0.0',
    'c-reactive_protein_30710-0.0','glucose_30740-0.0','glycated_haemoglobin_hba1c_30750-0.0',
    'hdl_cholesterol_30760-0.0','ldl_direct_30780-0.0','oestradiol_30800-0.0',
    'total_bilirubin_30840-0.0','testosterone_30850-0.0','total_protein_30860-0.0',
    'triglycerides_30870-0.0','vitamin_d_30890-0.0', 'body_mass_index_bmi_21001-2.0',
    'diastolic_blood_pressure_automated_reading_4079-0.1',
    'systolic_blood_pressure_automated_reading_4080-0.1',
    'age_when_attended_assessment_centre_21003-0.0',
    'age_when_attended_assessment_centre_21003-2.0',
    'body_fat_percentage_23099-0.0','waist_circumference_48-0.0',
    'hip_circumference_49-0.0','body_mass_index_bmi_21001-0.0',
    'diastolic_blood_pressure_automated_reading_4079-0.0',
    'systolic_blood_pressure_automated_reading_4080-0.0']

# Add subject ID and sex columns to keep
columns_to_keep = [col for col in interesting_labels if col in main_data.columns]
main_data = main_data[columns_to_keep + list(main_data.columns[main_data.columns.get_loc(columns_to_keep[-1]) + 1:])].copy()

main_data.reset_index(drop = True, inplace = True)
behavior.reset_index(drop = True, inplace = True)
filtered_site_data.reset_index(drop = True, inplace = True)

#------------------------------------------------------------------------------
# Remove usbjects with lots of missing values
# first, romove the subjects who has more than 20 missing values on brain or clinical side
# second, remove columns with lots of missing values more than 500)
#------------------------------------------------------------------------------

# Split into two parts (clinical measures and brain measures)
first_cols = columns_to_keep # This includes the interesting columns that were present in main_data
rest_cols = main_data.columns.difference(first_cols) # finds the brain data

# Create mask for rows to keep
mask_first = main_data[first_cols].isnull().sum(axis = 1) <= 40
mask_rest = main_data[rest_cols].isnull().sum(axis = 1) <= 40

# Keep only rows that satisfy both conditions
main_data = main_data[mask_first & mask_rest]
filtered_site_data = filtered_site_data[mask_first & mask_rest]
behavior = behavior[mask_first & mask_rest]

# Count NaNs in each column
nan_counts = main_data.isna().sum()

# Filter columns with more than 500 NaNs
columns_with_many_nans = nan_counts[nan_counts > 500]
main_data = main_data.drop(columns = columns_with_many_nans.index)
main_data.reset_index(drop = True, inplace = True)
columns_to_keep = [col for col in interesting_labels if col in main_data.columns]
   
# Devide the data into brain and biomarker(clinical) sections
data_bio = main_data[columns_to_keep].copy()
data_bio.reset_index(drop = True, inplace = True)
behavior.reset_index(drop = True, inplace = True)
filtered_site_data.reset_index(drop = True, inplace = True)

center_df = filtered_site_data[['eid', 'uk_biobank_assessment_centre_54-2.0']].copy()
center_df.columns = ['eid', 'center'] # rename
centername = center_df['center']

#------------------------------------------------------------------------------
# Remove age from measures that need to be corrected - using GAM models
#------------------------------------------------------------------------------

data_bio = data_bio.fillna(data_bio.mean())
age = data_bio['age_when_attended_assessment_centre_21003-2.0']

data_behavior_array = behavior.to_numpy()
behavior_names = behavior.columns.tolist()

np.save(path_results + f'biobank_behavior_{sex_label}_not_clean.npy', data_behavior_array)
np.save(path_results + f'biobank_names_behavior_{sex_label}_not_clean.npy', behavior_names)

###############################################################################
############################# Correct behavior ################################
###############################################################################

pandas2ri.activate()
gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')

residuals_Y = np.full_like(data_behavior_array, np.nan, dtype=np.float64)
center = center_df['center'].astype(str).values # Convert center to string

cmap = get_cmap("coolwarm")

if sex_label == 'Female':
    sex_color = cmap(0.9)  # red end
elif sex_label == 'Male':
    sex_color = cmap(0.1)  # blue end

# BEHAVIOR DATA
fig, axs = plt.subplots(9, 6, figsize = (20, 32))
axs = axs.flatten()  # Convert to 1D list so you can use axs[i]
for i in range(len(residuals_Y.T)):
    x = age
    y = data_behavior_array[:, i]

    # Remove rows with NaN in x or y
    mask_valid = ~np.isnan(x) & ~np.isnan(y)
    c = center # Site labels
    # Prepare valid data and transfer to R
    # df_r = pd.DataFrame({'x': x[mask_valid], 'y': y[mask_valid], 'center': c[mask_valid]})
    df_r = pd.DataFrame({'x': x[mask_valid], 'y': y[mask_valid]})
    ro.globalenv['df'] = pandas2ri.py2rpy(df_r)
    ro.r('library(gamlss)')
    # ro.r('df$center <- as.factor(df$center)') # Ensure center is treated as factor
    try:
        #ro.r('model <- gamlss(y ~ fp(x) + center, sigma.fo = ~fp(x), nu.fo = ~fp(x), data = df, family = NO())')
        ro.r('model <- gamlss(y ~ fp(x), sigma.fo = ~fp(x), data = df, family = NO())')
        mu_fitted = np.array(ro.r('fitted(model, what = "mu")'))

        ax = axs[i]
        ax.scatter(x[mask_valid], y[mask_valid], s = 10, color = 'silver', alpha = 0.8)
        ax.scatter(x[mask_valid], mu_fitted, color = sex_color, s = 10, label = 'GAMLSS fit', alpha = 0.7)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(behavior_names[i])
        ax.set_title(f'{behavior_names[i]} vs. Age')
        ax.legend(fontsize=8)
        residuals = y[mask_valid] - mu_fitted

        # Store residuals
        residuals_Y[mask_valid, i] = residuals
        print(i)
    except RRuntimeError as e:
        print(f"⚠️ Skipped variable {i} ({behavior_names[i]}): GAMLSS did not converge.")
        residuals_Y[:, i] = np.nan


plt.tight_layout()
plt.savefig(path_figures + f'gamlss_fit_behavior_{sex_label}.png', dpi = 300)
plt.show()

np.save(path_results + f'biobank_behavior_{sex_label}_clean_centercorrected.npy', residuals_Y)
np.save(path_results + f'biobank_names_behavior_{sex_label}_clean_centercorrected.npy', behavior_names)

#------------------------------------------------------------------------------
# END
