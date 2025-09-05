"""

HCP-A - age corrected using GAMLSS models

PLS cross validation (5 fold)

The first latent variables are cross-validated!

lv = 0: age axis
lv = 1: metabolic axis

################################# Females #####################################

    pval is (for lv = 0): 0.009900990099009901
    pval is (for lv = 1): 0.0891089108910891

################################## Males ######################################

    pval is (for lv = 0): 0.009900990099009901
    pval is (for lv = 1): 0.22772277227722773

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
from pyls import behavioral_pls
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
from sklearn.model_selection import KFold
from globals import path_results, path_figures

#------------------------------------------------------------------------------
# Load clean data
#------------------------------------------------------------------------------

sex_label = 'Male' # you can change this to Female or Male
lv = 1             # you can change this to teh lv of interest

Y1 = np.load(path_results + f'bio_data_PLS_{sex_label}_clean.npy').T
Y2 = np.load(path_results + f'bio_data_PLS_{sex_label}.npy').T[:1,:]/12
Y = np.concatenate((Y2.T, Y1), axis = 1)
X = np.load(path_results + f'brain_data_PLS_{sex_label}_clean.npy').T

#names = np.load(path_results + f'names_bio_data_PLS_{sex_label}.npy', allow_pickle=True)
name_write = [
    "Age","LH", "FSH", "Estradiol", "Testosterone", "ALT", "Cholesterol", "Triglycerides",
    "Urea", "Protein", "CO2", "Calcium", "Bilirubin", "Albumin", "HDL", "Liver_ALK", "HbA1c",
    "Insulin", "LDL", "Vitamin_D", "AST", "Glucose", "Chloride", "Creatinine", "Potassium",
    "Sodium", "MAP", "BMI"
]

#------------------------------------------------------------------------------
# PLS cross validation
#------------------------------------------------------------------------------

# Fill NaNs
nan_mask = np.isnan(X)
col_means = np.nanmean(X, axis = 0)
X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

# Standardize
X = zscore(X, axis = 0)
Y = zscore(Y, axis = 0)

# Replace NaNs in X with random values ~ N(0, 1)
nan_mask_X = np.isnan(X)
X[nan_mask_X] = np.random.normal(loc = 0.0, scale = 1.0, size = np.sum(nan_mask_X))

# Replace NaNs in Y with random values ~ N(0, 1)
nan_mask_Y = np.isnan(Y)
Y[nan_mask_Y] = np.random.normal(loc = 0.0, scale = 1.0, size = np.sum(nan_mask_Y))

n_splits = 5
nperm = 100

def cv_cal(X, Y):
    corr_test = np.zeros((n_splits, nperm))
    corr_train = np.zeros((n_splits, nperm))

    for iter_ind in range(nperm):
        kf = KFold(n_splits = n_splits, shuffle = True)
        c = 0
        for train_index, test_index in kf.split(X):

            Xtrain, Xtest = X[train_index], X[test_index]
            Ytrain, Ytest = Y[train_index], Y[test_index]

            train_result = behavioral_pls(Xtrain,
                                          Ytrain,
                                          n_boot = 0,
                                          n_perm = 0,
                                          test_split = 0,
                                          seed = 10)
            corr_train[c, iter_ind], _ = pearsonr(train_result['x_scores'][:, lv],
                                            train_result['y_scores'][:, lv])

            # project weights, correlate predicted scores in the test set
            corr_test[c, iter_ind], _ = pearsonr(Xtest @ train_result['x_weights'][:, lv],
                                   Ytest @ train_result['y_weights'][:, lv])
            c = c + 1
    return(corr_train, corr_test)

corr_train, corr_test = cv_cal(X, Y)

#------------------------------------------------------------------------------
# Permutation step
#------------------------------------------------------------------------------

def single_cv_cal(X, Y):
    corr_test = np.zeros((n_splits, 1))
    corr_train = np.zeros((n_splits, 1))
    kf = KFold(n_splits = n_splits, shuffle = True)
    c = 0
    for train_index, test_index in kf.split(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

        train_result = behavioral_pls(Xtrain,
                                      Ytrain,
                                      n_boot = 0,
                                      n_perm = 0,
                                      test_split = 0,
                                      seed = 10)
        corr_train[c, 0], _ = pearsonr(train_result['x_scores'][:, lv],
                                        train_result['y_scores'][:, lv])

        # project weights, correlate predicted scores in the test set
        corr_test[c, 0], _ = pearsonr(Xtest @ train_result['x_weights'][:, lv],
                               Ytest @ train_result['y_weights'][:, lv])
        c = c + 1
    return(corr_train.flatten(), corr_test.flatten())

per_train_corr = np.zeros((n_splits, nperm))
per_test_corr = np.zeros((n_splits, nperm))

num_subjects = len(X)
perms_y = np.zeros((num_subjects, nperm))

for perm_ind in range(nperm):
    perms_y[:, perm_ind] = np.random.permutation(range(0, num_subjects))

for perm_ind in range(nperm):
    tempy = perms_y[:, perm_ind].astype(int)
    Y_permuted = Y[tempy]

    per_train_corr[:, perm_ind], per_test_corr[:, perm_ind] = single_cv_cal(X, Y_permuted)
    print(perm_ind)

# VISUALIZATION ---------------------------------------------------------------

flat_train = corr_train[0, :].flatten()
flat_test  = corr_test[0, :].flatten()

flat_train_per = per_train_corr[0, :].flatten()
flat_test_per  = per_test_corr[0, :].flatten()

p_val = (1 + np.count_nonzero(((flat_test_per - np.mean(flat_test_per)))
                                > ((np.mean(flat_test) - np.mean(flat_test_per))))) / (nperm + 1)
print('pval is (for lv = ' + str(lv) + '): ' + str(p_val))

p_val = (1 + np.count_nonzero((flat_test_per > (np.mean(flat_test))))) / (nperm + 1)

print('pval is (for lv = ' + str(lv) + '): ' + str(p_val))
combined_train_test = [flat_train, flat_train_per, flat_test, flat_test_per]

plt.boxplot(combined_train_test)
plt.xticks([1, 2, 3, 4], ['train', 'train permute', 'test', 'test permute'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
ax = plt.gca()
y_ticks = np.linspace(-0.5, 1, num = 5)
ax.set_yticks(y_ticks)
plt.savefig(path_figures + 'GAM_PLS_cross_validation_lv_' + str(lv) + f'_{sex_label}.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

#------------------------------------------------------------------------------
# END
