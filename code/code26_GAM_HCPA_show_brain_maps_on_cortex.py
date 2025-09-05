"""

HCP-A - age corrected using GAMLSS models

Plot brain loadings - cortical loadings only!

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

lv = 0
names = [
        'loadings_perfusion_lv' + str(lv) + '_Female',
        'loadings_arrival_lv' + str(lv) + '_Female',
        'loadings_thickness_lv' + str(lv) + '_Female',
        'loadings_myelin_lv' + str(lv) + '_Female',
        'loadings_fa_lv' + str(lv) + '_Female',
        'loadings_md_lv' + str(lv) + '_Female',
        'loadings_fc_lv' + str(lv) + '_Female',
        'loadings_sc_lv' + str(lv) + '_Female',
        ]

for i in range(8):
    img = nib.cifti2.load(path_results + 'GAM_jhu_' + names[i] + '_cortex_parcellated.dscalar.nii')
    dscalars_gradient = img.get_fdata()[0,:globals.num_cort_vertices_noMW].T
    params = dict()
    params['disp-zero'] = True
    params['neg-user'] = (0, np.mean(dscalars_gradient) - np.std(dscalars_gradient))
    params['pos-user'] = (0, np.mean(dscalars_gradient) + np.std(dscalars_gradient))
    dscalar(file_out = path_figures + 'GAM_jhu_' + names[i] + '.png',
            dscalars = dscalars_gradient,
            orientation = 'landscape',
            hemisphere = None,
            palette = 'cool-warm',
            palette_params = params,
            transparent = False)

names = [
        'loadings_perfusion_lv' + str(lv) + '_Male',
        'loadings_arrival_lv' + str(lv) + '_Male',
        'loadings_thickness_lv' + str(lv) + '_Male',
        'loadings_myelin_lv' + str(lv) + '_Male',
        'loadings_fa_lv' + str(lv) + '_Male',
        'loadings_md_lv' + str(lv) + '_Male',
        'loadings_fc_lv' + str(lv) + '_Male',
        'loadings_sc_lv' + str(lv) + '_Male',
        ]

for i in range(8):
    img = nib.cifti2.load(path_results + 'GAM_jhu_' + names[i] + '_cortex_parcellated.dscalar.nii')
    dscalars_gradient = img.get_fdata()[0,:globals.num_cort_vertices_noMW].T
    params = dict()
    params['disp-zero'] = True
    params['neg-user'] = (0, np.mean(dscalars_gradient) - np.std(dscalars_gradient))
    params['pos-user'] = (0, np.mean(dscalars_gradient) + np.std(dscalars_gradient))
    dscalar(file_out = path_figures + 'GAM_jhu_' + names[i] + '.png',
            dscalars = dscalars_gradient,
            orientation = 'landscape',
            hemisphere = None,
            palette = 'cool-warm',
            palette_params = params,
            transparent = False)

#------------------------------------------------------------------------------
# END