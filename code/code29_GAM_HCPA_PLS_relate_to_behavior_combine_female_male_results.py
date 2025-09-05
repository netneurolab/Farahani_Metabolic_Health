"""

HCP-A - age corrected using GAMLSS models

lv = 0 # metabolic-axis - age-corrected
	name	corr_male	pval_male	pval_fdr_male	corr_female	pval_female	pval_fdr_female
0	nih_crycogcomp_unadjusted (Crystal Cognition Composite Score unadjusted)	0.0987397105956757	0.1378621378621378	0.1378621378621378	0.0597348555099129	0.3046953046953047	0.3046953046953047
1	nih_fluidcogcomp_unadjusted (Fluid Cognition Composite Score unadjusted)	0.1039810500075665	0.1198801198801198	0.1378621378621378	0.1783706453050003	0.0029970029970029	0.0089910089910087
2	nih_totalcogcomp_unadjusted (Total Cognition Composite Score unadjusted)	0.1134248219156276	0.0879120879120879	0.1378621378621378	0.13871474784325	0.0169830169830169	0.02547452547452535

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

#------------------------------------------------------------------------------
# Load and merge results
#------------------------------------------------------------------------------

df_male = pd.read_csv(path_results + f'GAM_Male_spearman_permutation_results_x_score_lv_{lv}.csv')
df_female = pd.read_csv(path_results + f'GAM_Female_spearman_permutation_results_x_score_lv_{lv}.csv')

# FDR correction
df_male['p_fdr'] = multipletests(df_male['p_value_notcorrected'], method='fdr_bh')[1]
df_female['p_fdr'] = multipletests(df_female['p_value_notcorrected'], method='fdr_bh')[1]

# Find common features
common_names = set(df_male['name']).intersection(df_female['name'])

# Keep only common features
df_male = df_male[df_male['name'].isin(common_names)].reset_index(drop=True)
df_female = df_female[df_female['name'].isin(common_names)].reset_index(drop=True)

# Sort both for alignment
df_male = df_male.sort_values('name').reset_index(drop=True)
df_female = df_female.sort_values('name').reset_index(drop=True)


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
df_merged.to_csv(path_results + f'GAM_combined_corr_and_fdr_lv_{lv}.csv', index=False)

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
plot_df = plot_df.sort_values('mean_abs_corr', ascending=False).drop(columns='mean_abs_corr')

# Handle empty case
if plot_df.empty:
    print("No features pass the FDR threshold in either sex.")
else:
    names = plot_df['name'].values
    male_vals = plot_df['corr_male'].values
    female_vals = plot_df['corr_female'].values

    x = np.arange(len(names))
    width = 0.38

    # Colors
    cmap = cm.get_cmap('coolwarm')
    male_color = cmap(0.10)   # male color
    female_color = cmap(0.90) # female color

    fig, ax = plt.subplots(figsize=(12,12))
    ax.bar(x - width/2, male_vals, width, label='Male', color=male_color, edgecolor='none')
    ax.bar(x + width/2, female_vals, width, label='Female', color=female_color, edgecolor='none')

    # Axis cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Spearman r')
    ax.set_title(f'Brain–behavior correlations (LV {lv}) — features significant in either sex')
    ax.axhline(0, linewidth=1, color='k', alpha=0.5)
    ax.legend(frameon=False, ncol=2)

    # Tight layout + save
    plt.tight_layout()
    out_path = path_figures + f'GAM_barplot_behavior_male_female_lv_{lv}.svg'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved figure to: {out_path}")
#------------------------------------------------------------------------------
# END
