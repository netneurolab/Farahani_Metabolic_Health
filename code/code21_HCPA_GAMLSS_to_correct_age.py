"""

HCP-A

Clean both sides from age effect (biomarkers and brain features) - using GAMLSS models

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
from globals import path_figures, path_results

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

sex_label = 'Male' # This can be Female or Male!

data = np.load(path_results + f'brain_data_PLS_{sex_label}.npy')
behaviour_data_array = np.load(path_results + f'bio_data_PLS_{sex_label}.npy')

names = np.load(path_results + f'names_bio_data_PLS_{sex_label}.npy', allow_pickle = True)

Y = behaviour_data_array[:,1:]
X = data

age_sex = behaviour_data_array[:, 0]/12

# Activate R sapce
pandas2ri.activate()
gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')

#------------------------------------------------------------------------------
# Fit GAMLSS for each sex group
#------------------------------------------------------------------------------

cmap = get_cmap("coolwarm")

if sex_label == 'Female':
    sex_color = cmap(0.9)  # red end
elif sex_label == 'Male':
    sex_color = cmap(0.1)  # blue end

# Biomarker data
fig, axs = plt.subplots(7, 4, figsize = (20, 28))
axs = axs.flatten()

residuals_Y = np.zeros_like(Y.T)
for i in range(27):
    x = age_sex
    y = Y[:, i]
    df_r = pd.DataFrame({'x': x, 'y': y})
    ro.globalenv['df'] = pandas2ri.py2rpy(df_r)
    ro.r('library(gamlss)')
    ro.r('library(gamlss.add)')
    ro.r('model <- gamlss(y ~ fp(x), sigma.fo = ~ fp(x), data = df, family = NO())')
    mu_fitted = np.array(ro.r('fitted(model, what = "mu")'))
    ax = axs[i]
    ax.scatter(x, y, s = 10, color = 'silver', alpha = 0.8)
    ax.scatter(x, mu_fitted, color = sex_color, s = 10, label = 'GAMLSS fit', alpha = 0.7)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel(names[i])
    ax.set_title(f'{names[i]} vs. Age')
    ax.legend(fontsize = 8)
    residuals_Y[i, :] = y - mu_fitted
    print(i)

# Hide any unused subplots
for j in range(27, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig(path_figures + f'gamlss_fit_biomarkers_{sex_label}.png', dpi = 300)
plt.show()

np.save(path_results + f'bio_data_PLS_{sex_label}_clean.npy', residuals_Y)
names = names[1:]
np.save(path_results + f'names_bio_data_PLS_{sex_label}_clean.npy', names)

# Brain data
residuals_X = np.zeros_like(X.T)
for i in range(3289):
    x = age_sex
    y = X[:, i]
    df_r = pd.DataFrame({'x': x, 'y': y})
    ro.globalenv['df'] = pandas2ri.py2rpy(df_r)
    ro.r('library(gamlss)')
    try:
        ro.r('model <- gamlss(y ~ fp(x), sigma.fo = ~ fp(x), data = df, family = NO())')
        mu_fitted = np.array(ro.r('fitted(model, what = "mu")'))
        residuals_X[i, :] = y - mu_fitted

    except Exception as e:
        print(f"⚠️ Skipping feature {i} due to GAMLSS error: {e}")
        residuals_X[i, :] = np.nan
        print('------------------------------')
    print(i)

np.save(path_results + f'brain_data_PLS_{sex_label}_clean.npy', residuals_X)

#------------------------------------------------------------------------------
# END
