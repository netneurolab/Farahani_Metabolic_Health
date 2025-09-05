"""
HCP-A

lv = 0
 	name                                                                    	corr_male	pval_male	pval_fdr_male	corr_female	pval_female	pval_fdr_female
0	nih_crycogcomp_unadjusted (Crystal Cognition Composite Score unadjusted)	-0.2072737526159824	0.0029970029970029	0.00449550449550435	-0.1390730490023907	0.0169830169830169	0.0169830169830169
1	nih_fluidcogcomp_unadjusted (Fluid Cognition Composite Score unadjusted)	0.4405262944188222	0.0009990009990009	0.0029970029970027	0.5067521912575811	0.0009990009990009	0.00149850149850135
2	nih_totalcogcomp_unadjusted (Total Cognition Composite Score unadjusted)	0.1758594745890116	0.0079920079920079	0.0079920079920079	0.2961849260879056	0.0009990009990009	0.00149850149850135

lv = 1
	name                                                                    	corr_male	pval_male	pval_fdr_male	corr_female	pval_female	pval_fdr_female
0	nih_crycogcomp_unadjusted (Crystal Cognition Composite Score unadjusted)	0.0876499001713999	0.2077922077922078	0.2077922077922078	0.0706359105323849	0.2337662337662337	0.3426573426573426
1	nih_fluidcogcomp_unadjusted (Fluid Cognition Composite Score unadjusted)	0.1587828490074104	0.0239760239760239	0.03596403596403585	0.0587620804414154	0.3426573426573426	0.3426573426573426
2	nih_totalcogcomp_unadjusted (Total Cognition Composite Score unadjusted)	0.1652861456034695	0.0169830169830169	0.03596403596403585	0.0762459411726835	0.2127872127872128	0.3426573426573426

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from globals import path_results, path_figures
from statsmodels.stats.multitest import multipletests

lv = 0

# Load male and female results
df_male = pd.read_csv(path_results + 'Male_spearman_permutation_results_x_score_lv_' + str(lv) + '.csv')
df_female = pd.read_csv(path_results + 'Female_spearman_permutation_results_x_score_lv_' + str(lv) + '.csv')

# FDR correction
df_male['p_fdr'] = multipletests(df_male['p_value_notcorrected'], method = 'fdr_bh')[1]
df_female['p_fdr'] = multipletests(df_female['p_value_notcorrected'], method = 'fdr_bh')[1]

# Find common features
common_names = set(df_male['name']).intersection(df_female['name'])

# Keep only common features
df_male = df_male[df_male['name'].isin(common_names)].reset_index(drop = True)
df_female = df_female[df_female['name'].isin(common_names)].reset_index(drop = True)

# Sort both for alignment
df_male = df_male.sort_values('name').reset_index(drop = True)
df_female = df_female.sort_values('name').reset_index(drop = True)

df_male['p_value'] = (df_male['p_value_notcorrected'])
df_female['p_value'] = (df_female['p_value_notcorrected'])

# Combine into a single dataframe
df_merged = pd.DataFrame({
    'name': df_male['name'],
    'corr_male': df_male['observed_corr'],
    'pval_male': df_male['p_value'],
    'pval_fdr_male': df_male['p_fdr'],
    'corr_female': df_female['observed_corr'],
    'pval_female': df_female['p_value'],
    'pval_fdr_female': df_female['p_fdr']
    
})

# Save merged results
df_merged.to_csv(path_results + f'combined_behavioral_corr_and_fdr_lv_{lv}.csv', index=False)

# Show significant results for either sex (optional)
print(df_merged[(df_merged['pval_fdr_male'] < 0.05) | (df_merged['pval_fdr_female'] < 0.05)])

#------------------------------------------------------------------------------
# Show results as barplots
#------------------------------------------------------------------------------

# Filter out rows with p < 0.05
df_female_filtered = df_merged[df_merged['pval_fdr_female'] < 0.05]
df_male_filtered = df_merged[df_merged['pval_fdr_male'] < 0.05]

print(df_female_filtered)
print(df_male_filtered)

#------------------------------------------------------------------------------
# Plot for paper
#------------------------------------------------------------------------------

plot_df = df_merged

# If you want to sort by effect size (optional):
plot_df['mean_abs_corr'] = (np.abs(plot_df['corr_male']) + np.abs(plot_df['corr_female'])) / 2
plot_df = plot_df.sort_values('mean_abs_corr', ascending = False).drop(columns = 'mean_abs_corr')

# Handle empty case
if plot_df.empty:
    print("No features pass the FDR threshold in either sex.")
else:
    names = plot_df['name'].values
    male_vals = plot_df['corr_male'].values
    female_vals = plot_df['corr_female'].values

    x = np.arange(len(names))
    width = 0.38

    cmap = cm.get_cmap('coolwarm')
    male_color = cmap(0.10)   # male color
    female_color = cmap(0.90) # female color

    fig, ax = plt.subplots(figsize = (12, 12))
    ax.bar(x - width/2, male_vals, width, label = 'Male', color = male_color, edgecolor = 'none')
    ax.bar(x + width/2, female_vals, width, label = 'Female', color = female_color, edgecolor = 'none')

    # Axis cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation = 45, ha = 'right')
    ax.set_ylabel('Spearman r')
    ax.set_title(f'Brain–behavior correlations (LV {lv}) — features significant in either sex')
    ax.axhline(0, linewidth = 1, color = 'k', alpha = 0.5)
    ax.legend(frameon = False, ncol = 2)

    # Tight layout + save
    plt.tight_layout()
    out_path = path_figures + f'barplot_behavior_male_female_lv_{lv}.svg'
    plt.savefig(out_path, dpi = 300, bbox_inches = 'tight')
    plt.show()
    print(f"Saved figure to: {out_path}")
#------------------------------------------------------------------------------
# END