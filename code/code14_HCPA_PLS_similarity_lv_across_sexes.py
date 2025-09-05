"""

HCP-A

How similar are brain loadings for aging/metabolic axes across females and males?

lv = 0 - CORTEX

    ----------
    Perfusion
    0.7891445746207308
    ----------
    Arrival
    0.8199395044116358
    ----------
    Thickness
    0.838429540721035
    ----------
    Myelin
    0.7370417886807593
    ----------
    FA
    0.8477533780010352
    ----------
    MD
    0.8456332641241484
    ----------
    FC
    0.7152707617905143
    ----------
    SC
    0.8045420096320182
    ----------
FDR - corrected p-values:
    0.000999, 0.000999, 0.000999, 0.000999, 0.000999, 0.000999, 0.000999, 0.000999

lv = 1 - CORTEX

    ----------
    Perfusion
    0.6175663707803797
    ----------
    Arrival
    0.8036595899194436
    ----------
    Thickness
    0.4102975397808486
    ----------
    Myelin
    0.2842744119530277
    ----------
    FA
    0.16435931641431234
    ----------
    MD
    0.5031683835607639
    ----------
    FC
    0.23956974064803158
    ----------
    SC
    0.028757783929292367
    ----------

FDR - corrected p-values:
    0.001998  , 0.001998  , 0.00799201, 0.001998  , 0.01027544, 0.001998  , 0.00639361, 0.43456543

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from neuromaps.images import load_data
from neuromaps.images import dlabel_to_gifti
from functions import vasa_null_Schaefer, pval_cal
from netneurotools.datasets import fetch_schaefer2018
from statsmodels.stats.multitest import multipletests
from globals import path_results, path_figures, path_yeo
from functions import save_parcellated_data_in_SchaeferTian_forVis

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

lv = 0 # This can be 0 or 1

nnodes = 400
cortical_features = ["Perfusion", "Arrival", "Thickness", "Myelin", "FA", "MD", "FC", "SC"]
variances = np.zeros(len(cortical_features))

#------------------------------------------------------------------------------
# Compare each feature map across subjects
#------------------------------------------------------------------------------

# Download the yeo networks
schaefer = fetch_schaefer2018('fslr32k')[str(nnodes) + 'Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))

yeo7 = np.load(path_yeo + 'Schaefer2018_400Parcels_7Networks.npy')
yeo_labels = yeo7 + 1 # assuming shape (400,) with labels from 1 to 7

num_yeo_networks = 7

# Yeo 7 Networks Colors
yeo_colors = {
    1: [120/255, 18/255 , 134/255], # Visual Network 1
    2: [70/255 , 130/255, 180/255], # Visual Network 2
    3: [0/255  , 118/255, 14/255],  # Somatomotor Network
    4: [196/255, 58/255 , 250/255], # Dorsal Attention Network
    5: [220/255, 248/255, 164/255], # Ventral Attention Network
    6: [230/255, 148/255, 34/255],  # Limbic Network
    7: [205/255, 62/255 , 78/255] } # Default Mode Network

#------------------------------------------------------------------------------
corr_values = np.zeros(len(cortical_features))

for i in range(len(cortical_features)):
    data_male = np.load(path_results + 'loadings_CORTEX_' + cortical_features[i] + '_Male_lv_' + str(lv) + '.npy')
    data_female = np.load(path_results + 'loadings_CORTEX_' + cortical_features[i] + '_Female_lv_' + str(lv) + '.npy')
    corr_values[i] = pearsonr(data_male, data_female)[0]
    plt.figure(figsize = (5, 5))
    plt.scatter(data_male, data_female, color = 'silver')
    plt.show()
    print(cortical_features[i])
    print(corr_values[i])
    print('----------')

nspins = 1000 # number of spins
spins = vasa_null_Schaefer(nspins)

def corr_spin(x, y, spins, nspins):
    """
    Spin test - account for spatial autocorrelation
    """
    rho, _ = pearsonr(x, y)
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
        null[i], _ = pearsonr(x, y[spins[:, i]])
    return rho, null

p_values = np.zeros(len(cortical_features))
for i in range(len(cortical_features)):
    # Calculate Correlation + Spin-Test
    data_male = np.load(path_results + 'loadings_CORTEX_' + cortical_features[i] + '_Male_lv_' + str(lv) + '.npy')
    data_female = np.load(path_results + 'loadings_CORTEX_' + cortical_features[i] + '_Female_lv_' + str(lv) + '.npy')
    
    r, generated_null = corr_spin(data_male.flatten(),
                                data_female.flatten(),
                                spins,
                                nspins)
    p_values[i] = pval_cal(r, generated_null, nspins)

# Multi-test correction
rejected, pvals_corrected, _, _ = multipletests(p_values,
                                                alpha = 0.05,
                                                method = 'fdr_bh')

#------------------------------------------------------------------------------
# Where is the change more dominant in the cortex
#------------------------------------------------------------------------------

# This part of code just saves the yeo networks on the cortical surface
yeo = np.load(path_yeo + 'Schaefer2018_400Parcels_7Networks.npy')
save_parcellated_data_in_SchaeferTian_forVis(yeo,
                                             'cortex',
                                             'X',
                                             path_results,
                                             'yeo')
yeo_classes = ['visual',
               'motor',
               'dorsal att',
               'ventral att',
               'limbic', 
               'default',
               'fronto']

fig, axs = plt.subplots(2, 4, figsize = (20, 10))
axs = axs.flatten()

for i, feature in enumerate(cortical_features):
    x = np.load(path_results + 'loadings_CORTEX_' + cortical_features[i] + '_Male_lv_' + str(lv) + '.npy')
    y = np.load(path_results + 'loadings_CORTEX_' + cortical_features[i] + '_Female_lv_' + str(lv) + '.npy')
    ax = axs[i]
    for j in range(nnodes):
        label = yeo_labels[j]
        color = yeo_colors.get(label, [0.5, 0.5, 0.5])
        ax.scatter(x[j], y[j], color = color, s = 20, edgecolor = 'k', linewidth = 0.2)

    minval, maxval = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([minval, maxval], [minval, maxval], linestyle = '--', color = 'gray')
    
    ax.set_title(f'{feature} (r = {corr_values[i]:.2f}, p = {pvals_corrected[i]:.3f})')
    ax.set_xlabel('Male Loadings')
    ax.set_ylabel('Female Loadings')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for j in range(len(cortical_features), len(axs)):
    fig.delaxes(axs[j])
    
plt.tight_layout()
plt.savefig(f"{path_figures}scatter_Male_vs_Female_CORTEX_lv_{lv}.svg", dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# The same for WMHs
#------------------------------------------------------------------------------

# Load data
x = np.load(path_results + f'loadings_WMH_Male_lv_{lv}.npy')
y = np.load(path_results + f'loadings_WMH_Female_lv_{lv}.npy')
corr_values_WMH = pearsonr(x, y)[0]

# Areal labels for each WMH region
areal_names = [
    'whole brain',
    'Frontal - L',
    'Frontal - R',
    'Parietal - L',
    'Parietal - R',
    'Temporal - L',
    'Temporal - R',
    'Occipital - L',
    'Occipital - R'
]

# Create figure and axis
fig, ax = plt.subplots(figsize = (6, 6))

# Scatter plot and add labels
for j in range(len(x)):
    ax.scatter(x[j], y[j], color = 'silver', s = 40, edgecolor = 'k', linewidth = 0.2)
    ax.text(x[j] + 0.005, y[j], areal_names[j], fontsize = 8, ha = 'left', va = 'center')

# Identity line
minval, maxval = min(x.min(), y.min()), max(x.max(), y.max())
ax.plot([minval, maxval], [minval, maxval], linestyle = '--', color = 'gray')

# Labels and styling
ax.set_title(f'WMH (r = {corr_values_WMH:.2f})')
ax.set_xlabel('Male Loadings')
ax.set_ylabel('Female Loadings')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{path_figures}scatter_Male_vs_Female_WMH_lv_{lv}_labeled.svg", dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# The same for WM tracts
#------------------------------------------------------------------------------

name_columns = ['Anterior thalamic radiation L',
                'Anterior thalamic radiation R',
                'Corticospinal tract L',
                'Corticospinal tract R',
                'Cingulum (cingulate gyrus) L',
                'Cingulum (cingulate gyrus) R',
                'Cingulum (hippocampus) L',
                'Cingulum (hippocampus) R',
                'Forceps major',
                'Forceps minor',
                'Inferior fronto-occipital fasciculus L',
                'Inferior fronto-occipital fasciculus R',
                'Inferior longitudinal fasciculus L',
                'Inferior longitudinal fasciculus R',
                'Superior longitudinal fasciculus L',
                'Superior longitudinal fasciculus R',
                'Uncinate fasciculus L',
                'Uncinate fasciculus R',
                'Superior longitudinal fasciculus (temporal part) L',
                'Superior longitudinal fasciculus (temporal part) R']

# JHU features
names_jhu = ["Perfusion", "Arrival", "FA", "MD"]

fig, axs = plt.subplots(1, 4, figsize = (20, 5))
axs = axs.flatten()

for i, feature in enumerate(names_jhu):
    x = np.load(f"{path_results}loadings_WM_{feature}_Male_lv_{lv}.npy")
    y = np.load(f"{path_results}loadings_WM_{feature}_Female_lv_{lv}.npy")
    ax = axs[i]
    
    for j in range(len(x)):
        ax.scatter(x[j], y[j], color = 'silver', s = 30, edgecolor = 'k', linewidth = 0.2)
        ax.text(x[j] + 0.005, y[j], name_columns[j], fontsize = 7, ha = 'left', va = 'center')
    
    minval, maxval = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([minval, maxval], [minval, maxval], linestyle = '--', color = 'gray')

    ax.set_title(f'{feature} (r = {pearsonr(x, y)[0]:.2f})')
    ax.set_xlabel('Male Loadings')
    ax.set_ylabel('Female Loadings')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Remove unused subplots
for j in range(len(names_jhu), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig(f"{path_figures}scatter_Male_vs_Female_WM_lv_{lv}_labeled.svg", dpi=300)
plt.show()

#------------------------------------------------------------------------------
# END
