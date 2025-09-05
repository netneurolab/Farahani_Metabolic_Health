"""

HCP-A

Parcellate and save white matter tract blood perfusion and ATT per individual subject

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from globals import path_info_sub, path_wm_asl, path_native_jhu_files, path_results

#------------------------------------------------------------------------------
# Parcellate individuals' WM ASL data
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
subject_ids = df.src_subject_id

nnodes = 20 # number of WM tracts
nsub = len(subject_ids)

perfusion = np.zeros((nnodes, nsub))
arrival = np.zeros((nnodes, nsub))

for s, subid in enumerate(subject_ids):
    print(s) # show progress

    # Load JHU subject parcellation
    jhu_img = nib.load(path_native_jhu_files + 'jhu_' + subid + '.nii.gz')
    jhu_data = jhu_img.get_fdata()
        
    perf_img = nib.load(path_wm_asl + subid + '_V1_MR/MNINonLinear/ASL/pvcorr_perfusion_wm_calib_masked.nii.gz')
    arr_img = nib.load(path_wm_asl + subid + '_V1_MR/MNINonLinear/ASL/pvcorr_arrival_wm_masked.nii.gz')

    # Resample to JHU space
    perf_resampled = resample_from_to(perf_img, jhu_img).get_fdata()
    arr_resampled = resample_from_to(arr_img, jhu_img).get_fdata()

    # Set zeros to NaN
    perf_resampled[perf_resampled == 0] = np.nan
    arr_resampled[arr_resampled == 0] = np.nan

    # Parcellate
    for j in range(nnodes):
        perfusion[j, s] = np.nanmean(perf_resampled[jhu_data == j + 1])
        arrival[j, s] = np.nanmean(arr_resampled[jhu_data == j + 1])

    # Save per-subject
    np.save(path_wm_asl + subid + '_perfusion_jhu.npy', perfusion[:, s])
    np.save(path_wm_asl + subid + '_arrival_jhu.npy', arrival[:, s])

np.save(path_results + 'perfusion_jhu.npy', perfusion)
np.save(path_results + 'arrival_jhu.npy', arrival)

#------------------------------------------------------------------------------
# END