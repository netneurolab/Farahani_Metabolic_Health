"""

HCP-A

Plot the biomakers(peripheral) vs. age

# female
Number of missing values replaced with median per column:
lh (LH blood test result)                              1
fsh (FSH blood test result)                            1
festrs (Hormonal Measures Female Estradiol Results)    1
rsptest_no (Blood value - Testosterone (ng/dL))        1
a1crs (HbA1c Results)                                  3
MAP (nan)                                              4
bmi                                                    2

# male
Number of missing values replaced with median per column:
lh (LH blood test result)                              2
fsh (FSH blood test result)                            2
festrs (Hormonal Measures Female Estradiol Results)    2
rsptest_no (Blood value - Testosterone (ng/dL))        2
a1crs (HbA1c Results)                                  1
MAP (nan)                                              2

Note: total number female: 329 - male: 268

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from globals import path_results, path_figures, path_info_sub

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

brain_data = np.load(path_results +'data_merged.npy')
num_subjects = brain_data.shape[1]

#------------------------------------------------------------------------------
# Load blood test + demographics data, clean them, and add BMI as one of the measures here
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

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
age_total = np.array(df['interview_age']) / 12
df['interview_age'] = df['interview_age'] / 12
df['sex'] = df['sex'].map({'F': 0, 'M': 1}) # 0: Female, 1: Male
sex = np.array(df['sex'])

finalDF = finalDF[columns_of_interest]

#------------------------------------------------------------------------------
# Group data based on biological sex
#------------------------------------------------------------------------------

def preprocess_by_sex(df_behav, sex_mask, age, sex_label):
    df_sub = df_behav[sex_mask].reset_index(drop = True)
    age_sub = age[sex_mask]

    # Drop subjects with >20 NaNs
    mask_valid = df_sub.isna().sum(axis = 1) <= 20
    df_sub = df_sub[mask_valid].reset_index(drop = True)
    age_sub = age_sub[mask_valid]

    # Show NaNs before imputation
    nan_summary = df_sub.isna().sum()
    print(f"\n# {sex_label}\nNumber of missing values replaced with median per column:")
    print(nan_summary[nan_summary > 0])

    # Median imputation
    df_sub = df_sub.apply(lambda col: col.fillna(np.nanmedian(col)), axis=0)
    return df_sub.to_numpy(), age_sub

behaviour_data_female, age_female = preprocess_by_sex(finalDF, sex == 0, age_total, "female")
behaviour_data_male, age_male     = preprocess_by_sex(finalDF, sex == 1, age_total, "male")

#------------------------------------------------------------------------------
# Plot the biomarkers versus age - HCP-A data
#------------------------------------------------------------------------------

cmap = get_cmap("coolwarm")
color_female = cmap(0.9)  # red end
color_male = cmap(0.1)    # blue end

plt.figure(figsize = (20, 28))
for i, name in enumerate(name_write[1:]): # skip the 'age' column when plotting
    ax = plt.subplot(7, 4, i + 1)

    ax.scatter(age_female, behaviour_data_female[:, i + 1],
               color = color_female, label = 'Female',
               alpha = 0.6, s = 20)

    ax.scatter(age_male, behaviour_data_male[:, i + 1],
               color = color_male, label = 'Male',
               alpha = 0.6, s = 20)

    ax.set_title(f"{name} vs. Age")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.savefig(path_figures + "behavioral_vs_age_scatterplots_female_male.png", dpi = 300)
plt.show()

cmap = get_cmap("coolwarm")
color_female = cmap(0.9)  # red end
color_male = cmap(0.1)    # blue end

plt.figure(figsize = (20, 28))
for i, name in enumerate(name_write[1:]): # skip the 'age' column when plotting
    ax = plt.subplot(7, 4, i + 1)

    ax.scatter(age_female, behaviour_data_female[:, i + 1],
               color = color_female, label = 'Female',
               alpha = 0.6, s = 20)

    ax.scatter(age_male, behaviour_data_male[:, i + 1],
               color = color_male, label = 'Male',
               alpha = 0.6, s = 20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis = 'both', length = 4, width = 1)

plt.savefig(path_figures + "behavioral_vs_age_scatterplots_female_male_notitle.png", dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END
