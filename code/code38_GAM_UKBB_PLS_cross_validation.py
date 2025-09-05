"""

UK Biobank - GAMLSS models are used to correct age-effects.

PLS cross validation (5 fold)
lv = 0: metabolic aixs

################################## Males ######################################
    pval is (for lv = 0): 0.009900990099009901

################################# Females #####################################
    pval is (for lv = 0): 0.009900990099009901

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
# Load data
#------------------------------------------------------------------------------

sex_label = 'Female'     # Do you want to study PLS cross validation for female or male groups?
lv = 0                   # You can change this to the lv of interest!

data_bio_array =  np.load(path_results + f'biobank_biomarkers_{sex_label}_clean_centercorrected.npy') # (1582, 32)
data_brain_array = np.load(path_results + f'biobank_brain_data_{sex_label}_clean_centercorrected.npy') # (1582, 485)

brain_names = np.load(path_results + f'biobank_names_brain_data_{sex_label}_clean_centercorrected.npy', allow_pickle=True) # 485
bio_names = np.load(path_results + f'biobank_names_biomarkers_{sex_label}_clean_centercorrected.npy', allow_pickle=True)

age_0 = np.load(path_results + f'biobank_age_{sex_label}_0_0.npy').reshape(-1,1)
age_2 = np.load(path_results + f'biobank_age_{sex_label}_2_0.npy').reshape(-1,1)

data_bio_array = np.concatenate((age_0, age_2, data_bio_array), axis = 1)
bio_names = np.concatenate((np.array(['age_0']), np.array(['age_2']), bio_names), axis=0)

#------------------------------------------------------------------------------
# PLS cross validation
#------------------------------------------------------------------------------

# Z-score
X = zscore(data_brain_array, axis = 0)
Y = zscore(data_bio_array, axis = 0)

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
plt.savefig(path_figures + 'GAM_UKBB_PLS_cross_validation_lv_' + str(lv) + f'_{sex_label}.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

#------------------------------------------------------------------------------
# END
