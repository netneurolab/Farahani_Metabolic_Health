"""

HCP-A

PLot brain loadings pf fsLR tempalte space - cortical features only.

"""
#------------------------------------------------------------------------------
# load libraries
#------------------------------------------------------------------------------

import globals
import numpy as np
import nibabel as nib
from wbplot import dscalar
from globals import path_results, path_figures

#------------------------------------------------------------------------------

lv = 0 # this can be 0 or 1!

names = [
        'loadings_Perfusion_lv_' + str(lv) + '_Female',
        'loadings_Arrival_lv_' + str(lv) + '_Female',
        'loadings_Thickness_lv_' + str(lv) + '_Female',
        'loadings_Myelin_lv_' + str(lv) + '_Female',
        'loadings_FA_lv_' + str(lv) + '_Female',
        'loadings_MD_lv_' + str(lv) + '_Female',
        'loadings_FC_lv_' + str(lv) + '_Female',
        'loadings_SC_lv_' + str(lv) + '_Female',
        ]

for i in range(8):
    img = nib.cifti2.load(path_results + names[i] + '_cortex_parcellated.dscalar.nii')
    dscalars_gradient = img.get_fdata()[0,:globals.num_cort_vertices_noMW].T

    params = dict()
    params['disp-zero'] = True
    params['neg-user'] = (0, np.mean(dscalars_gradient) - np.std(dscalars_gradient))
    params['pos-user'] = (0, np.mean(dscalars_gradient) + np.std(dscalars_gradient))
    dscalar(file_out = path_figures + names[i] + '.png',
            dscalars = dscalars_gradient,
            orientation = 'landscape',
            hemisphere = None,
            palette = 'cool-warm',
            palette_params = params,
            transparent = False)

names = [
        'loadings_Perfusion_lv_' + str(lv) + '_Male',
        'loadings_Arrival_lv_' + str(lv) + '_Male',
        'loadings_Thickness_lv_' + str(lv) + '_Male',
        'loadings_Myelin_lv_' + str(lv) + '_Male',
        'loadings_FA_lv_' + str(lv) + '_Male',
        'loadings_MD_lv_' + str(lv) + '_Male',
        'loadings_FC_lv_' + str(lv) + '_Male',
        'loadings_SC_lv_' + str(lv) + '_Male',
        ]

for i in range(8):
    img = nib.cifti2.load(path_results + names[i] + '_cortex_parcellated.dscalar.nii')
    dscalars_gradient = img.get_fdata()[0,:globals.num_cort_vertices_noMW].T

    params = dict()
    params['disp-zero'] = True
    params['neg-user'] = (0, np.mean(dscalars_gradient) - np.std(dscalars_gradient))
    params['pos-user'] = (0, np.mean(dscalars_gradient) + np.std(dscalars_gradient))
    dscalar(file_out = path_figures + names[i] + '.png',
            dscalars = dscalars_gradient,
            orientation = 'landscape',
            hemisphere = None,
            palette = 'cool-warm',
            palette_params = params,
            transparent = False)

#------------------------------------------------------------------------------
# END