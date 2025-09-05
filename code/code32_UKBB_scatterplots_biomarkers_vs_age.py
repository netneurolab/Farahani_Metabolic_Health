"""

UK biobank

Plot thebiomarkers versus age[at imaging session]

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from globals import path_results, path_figures

#------------------------------------------------------------------------------
# Load the biomarker data
#------------------------------------------------------------------------------

bio_names = np.load(path_results + 'bio_names_Female_biobank.npy', allow_pickle = True)
data_bio_array_female = np.load(path_results + 'data_bio_array_Female_biobank_with_nan.npy')
data_bio_array_male = np.load(path_results + 'data_bio_array_Male_biobank_with_nan.npy')

# Get index of the age variable
age_index = np.where(bio_names == 'age_when_attended_assessment_centre_21003-2.0')[0][0]
age_female = data_bio_array_female[:,age_index]
age_male = data_bio_array_male[:,age_index]

# Remove the age column from data arrays
data_bio_array_female = np.delete(data_bio_array_female, age_index, axis = 1)
data_bio_array_male = np.delete(data_bio_array_male, age_index, axis = 1)

# Remove the age name from bio_names
bio_names = np.delete(bio_names, age_index)

cmap = cm.coolwarm
norm = colors.Normalize(vmin = 0, vmax = 100)

# Plot
plt.figure(figsize = (20, 32))
for i, name in enumerate(bio_names):
    plt.subplot(9, 4, i + 1)
    # Females as 90v
    plt.scatter(age_female, data_bio_array_female[:, i],
                color = cmap(norm(90)), label = 'Female',
                alpha = 0.4,  s = 20)
    # Males as 10
    plt.scatter(age_male, data_bio_array_male[:, i],
                color = cmap(norm(10)), label = 'Male',
                alpha = 0.4, s = 20)
    plt.ylabel(name)
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig(path_figures + "biobank_behavioral_vs_age_scatterplots_female_male.svg",
            dpi = 300)
plt.show()

# Plot
plt.figure(figsize = (20, 32))
for i, name in enumerate(bio_names):
    ax = plt.subplot(9, 4, i + 1)
    # Females as 90v
    ax.scatter(age_female, data_bio_array_female[:, i],
                color = cmap(norm(90)), label = 'Female',
                alpha = 0.4, s = 20)
    # Males as 10
    ax.scatter(age_male, data_bio_array_male[:, i],
                color = cmap(norm(10)), label = 'Male',
                alpha = 0.4, s = 20)
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Remove numerical labels but keep tick marks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(axis = 'both', length = 4, width = 1)

plt.savefig(path_figures + "biobank_behavioral_vs_age_scatterplots_female_male_notitle.png",
            dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END