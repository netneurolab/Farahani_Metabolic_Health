"""

UK biobank

Use GAMLSS models to correct age-effects

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
from globals import path_results, path_ukbiobank, path_figures

#------------------------------------------------------------------------------
# Indicate the biological sex of interest
#------------------------------------------------------------------------------

sex_label = 'Female' # This can be Female or Male!

if sex_label == 'Female':
    sex_num = 0
elif sex_label == 'Male':
    sex_num = 1

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

main_data = pd.read_csv(path_ukbiobank + 'merged_biodata_with_cbf.csv')

# Drop all columns ending in "-3.0"
main_data = main_data[[col for col in main_data.columns if not col.endswith('-3.0')]]

# load dataframe with a column which reports the data acquisition sites
site_data = pd.read_csv(path_ukbiobank + 'data_082025_withimaging.csv')

# Define prefixes to keep (assessment site and subject IDs)
prefixes = ('uk_biobank_assessment_centre_54-2.0',
            'age_when_attended_assessment_centre_21003-0.0',
            'age_when_attended_assessment_centre_21003-2.0',
            'eid')

filtered_site_data = site_data[[col for col in site_data.columns 
                              if col == 'eid' or col.startswith(prefixes)]]
filtered_site_data = filtered_site_data[filtered_site_data['eid'].isin(main_data['eid'])].copy()
filtered_site_data = filtered_site_data.set_index('eid').loc[main_data['eid'].values].reset_index()

main_data['sex_31-0.0'] = main_data['sex_31-0.0'].map({'Female': 0, 'Male': 1})
female_mask = main_data['sex_31-0.0'] == sex_num

# reorder both so that they have the same order of subjects
main_data = main_data[female_mask].reset_index(drop = True)
main_data = main_data.sort_values('eid').reset_index(drop = True)

filtered_site_data = filtered_site_data[female_mask].reset_index(drop = True)
filtered_site_data = filtered_site_data.sort_values('eid').reset_index(drop = True)

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

#------------------------------------------------------------------------------
# Define interesting labels to KEEP for the PLS model and hence should be corrected
#------------------------------------------------------------------------------

# Only retain the interesting columns from the clinical side and keep also the rest of brain data
columns_to_keep = [col for col in interesting_labels if col in main_data.columns]
main_data = main_data[columns_to_keep + list(main_data.columns[main_data.columns.get_loc(columns_to_keep[-1]) + 1:])].copy()

main_data.reset_index(drop = True, inplace = True)
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

# Count NaNs in each column
nan_counts = main_data.isna().sum()

# Filter columns with more than 500 NaNs
columns_with_many_nans = nan_counts[nan_counts > 500]
main_data = main_data.drop(columns = columns_with_many_nans.index)
main_data.reset_index(drop = True, inplace = True)
columns_to_keep = [col for col in interesting_labels if col in main_data.columns]
   
# Devide the data into brain and biomarker(clinical) sections
data_bio = main_data[columns_to_keep].copy()
data_brain = main_data.drop(columns=columns_to_keep).copy()

data_bio.reset_index(drop = True, inplace = True)
data_brain.reset_index(drop = True, inplace = True)

#------------------------------------------------------------------------------
# Get the center information for subjects who have ASL data
#------------------------------------------------------------------------------

center_df = filtered_site_data[['eid', 'uk_biobank_assessment_centre_54-2.0']].copy()
center_df.columns = ['eid', 'center'] # Rename for clarity
centername = center_df['center']

# Save as a .npy file - this is the information with regard to the imaging site
np.save(path_results + f'{sex_label}_centers.npy', center_df)

age_0  = filtered_site_data['age_when_attended_assessment_centre_21003-0.0']
age_2  = filtered_site_data['age_when_attended_assessment_centre_21003-2.0']

np.save(path_results + f'biobank_age_{sex_label}_0_0.npy', age_0)
np.save(path_results + f'biobank_age_{sex_label}_2_0.npy', age_2)

#------------------------------------------------------------------------------
# Remove age frmpm measures that need to be corrected - using GAM models
#------------------------------------------------------------------------------

# Mean imputation to fill missing values
data_brain = data_brain.fillna(data_brain.mean())
data_brain_array = (np.array(data_brain))
brain_names = data_brain.columns

data_bio = data_bio.fillna(data_bio.mean())

age = data_bio['age_when_attended_assessment_centre_21003-2.0']
data_bio = data_bio.drop(columns=['age_when_attended_assessment_centre_21003-2.0'])
data_bio = data_bio.drop(columns=['age_when_attended_assessment_centre_21003-0.0'])

data_bio_array = data_bio.to_numpy()
bio_names = data_bio.columns.tolist()

np.save(path_results + f'biobank_names_biomarkers_{sex_label}_clean_centercorrected.npy', data_bio.columns)
np.save(path_results + f'biobank_names_brain_data_{sex_label}_clean_centercorrected.npy', brain_names)
np.save(path_results + f'biobank_age_{sex_label}.npy', age)

###############################################################################
############################ Correct Brain data ###############################
###############################################################################

pandas2ri.activate()
gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')

residuals_Y = np.full_like(data_brain_array, np.nan, dtype=np.float64)
unique_centers = np.unique(centername)

center = center_df['center'].astype(str).values # Convert center to string

for i in range(490):
    y = data_brain_array[:, i]
    x = age_2
    c = center # Site labels

    df_r = pd.DataFrame({'x': x, 'y': y, 'center': c})
    ro.globalenv['df'] = pandas2ri.py2rpy(df_r)

    try:
        ro.r('library(gamlss)')
        ro.r('df$center <- as.factor(df$center)') # Ensure center is treated as factor
        ro.r('model <- gamlss(y ~ fp(x) + center, sigma.fo = ~fp(x), data = df, family = NO())')
        mu_fitted = np.array(ro.r('fitted(model, what = "mu")'))

        # Plot
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, s=10, label='Observed')
        plt.scatter(x, mu_fitted, color='red', label='GAMLSS fit')
        plt.xlabel('Age (years)')
        plt.ylabel(f'{brain_names[i]}')
        plt.title(f'{brain_names[i]} vs. Age (center-corrected)')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Residuals
        residuals_Y[:, i] = y - mu_fitted
        print(f"{i}: {brain_names[i]} done")

    except Exception as e:
        print(f"⚠️ Feature {i} — {brain_names[i]}: GAMLSS error — {e}")

np.save(path_results + f'biobank_brain_data_{sex_label}_clean_centercorrected.npy', residuals_Y)

###############################################################################
############################ Correct biomarkers ###############################
###############################################################################

pandas2ri.activate()
gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')

residuals_Y = np.full_like(data_bio_array, np.nan, dtype=np.float64)

cmap = get_cmap("coolwarm")

if sex_label == 'Female':
    sex_color = cmap(0.9)  # red end
elif sex_label == 'Male':
    sex_color = cmap(0.1)  # blue end

use_age0 = [name.endswith(('-0.0', '-0.1')) for name in bio_names]

# BIOMARKER DATA
fig, axs = plt.subplots(9, 4, figsize = (20, 32))
axs = axs.flatten() # Convert to 1D list so you can use axs[i]
for i in range(len(residuals_Y.T)):
    x = age_0 if use_age0[i] else age_2
    y = data_bio_array[:, i]
    c = center # Site labels
    df_r = pd.DataFrame({'x': x, 'y': y, 'center': c})
    ro.globalenv['df'] = pandas2ri.py2rpy(df_r)
    ro.r('library(gamlss)')
    ro.r('df$center <- as.factor(df$center)') # Ensure center is treated as factor
    ro.r('model <- gamlss(y ~ fp(x) + center, sigma.fo = ~fp(x), nu.fo = ~fp(x), data = df, family = NO())')
    mu_fitted = np.array(ro.r('fitted(model, what = "mu")'))

    ax = axs[i]
    ax.scatter(x, y, s = 10, color = 'silver', alpha = 0.8)
    ax.scatter(x, mu_fitted, color=sex_color, s=10, label='GAMLSS fit', alpha = 0.7)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel(bio_names[i])
    which_age = 'age_0' if use_age0[i] else 'age_2'
    ax.set_title(f'{bio_names[i]} vs. Age ({which_age})')
    ax.legend(fontsize = 8)
    residuals_Y[:, i] = y - mu_fitted
    print(i)

plt.tight_layout()
plt.savefig(path_figures + f'gamlss_fit_biomarkers_{sex_label}.png', dpi = 300)
plt.show()

np.save(path_results + f'biobank_biomarkers_{sex_label}_clean_centercorrected.npy', residuals_Y)

#------------------------------------------------------------------------------
# END
