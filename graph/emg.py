import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(8, 1, figsize=(19.2,10.03*1.5))

df_model_A1000000 = pd.read_csv('measured_data_10_A1000000.csv')

measure_start = 2387
measure_fin = 2711
xlines2 = np.linspace(0.0,1.0,measure_fin-measure_start+1)
A_HFL_A1000000 = df_model_A1000000.loc[measure_start:measure_fin, 'A_HFL'].values
A_GLU_A1000000 = df_model_A1000000.loc[measure_start:measure_fin, 'A_GLU'].values
A_VAS_A1000000 = df_model_A1000000.loc[measure_start:measure_fin, 'A_VAS'].values
A_SOL_A1000000 = df_model_A1000000.loc[measure_start:measure_fin, 'A_SOL'].values
A_GAS_A1000000 = df_model_A1000000.loc[measure_start:measure_fin, 'A_GAS'].values
A_TA_A1000000  = df_model_A1000000.loc[measure_start:measure_fin, 'A_TA'].values
A_HAM_A1000000 = df_model_A1000000.loc[measure_start:measure_fin, 'A_HAM'].values
A_RF_A1000000  = df_model_A1000000.loc[measure_start:measure_fin, 'A_RF'].values


row = 0
axes[row].fill_between(xlines2, 0, A_HFL_A1000000, color='gray', edgecolor=None)
axes[row].tick_params(labelsize = 20)
axes[row].set_ylabel('HFL', fontsize=40)
axes[row].set_xlim(0.0,1.0)
axes[row].set_ylim(0.0,1.0)
axes[row].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[row].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
stance_to_swing = (2555-measure_start+1)/(measure_fin-measure_start)
axes[row].axvline(x=stance_to_swing, color='maroon',linestyle='dashed',linewidth='3')

row = 1
axes[row].fill_between(xlines2, 0, A_GLU_A1000000, color='gray', edgecolor=None)
axes[row].tick_params(labelsize = 20)
axes[row].set_ylabel('GLU', fontsize=40)
axes[row].set_xlim(0.0,1.0)
axes[row].set_ylim(0.0,1.0)
axes[row].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[row].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
axes[row].axvline(x=stance_to_swing, color='maroon',linestyle='dashed',linewidth='3')

row = 2
axes[row].fill_between(xlines2, 0, A_VAS_A1000000, color='gray', edgecolor=None)
axes[row].tick_params(labelsize = 20)
axes[row].set_ylabel('VAS', fontsize=40)
axes[row].set_xlim(0.0,1.0)
axes[row].set_ylim(0.0,1.0)
axes[row].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[row].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
axes[row].axvline(x=stance_to_swing, color='maroon',linestyle='dashed',linewidth='3')

row = 3
axes[row].fill_between(xlines2, 0, A_SOL_A1000000, color='gray', edgecolor=None)
axes[row].tick_params(labelsize = 20)
axes[row].set_ylabel('SOL', fontsize=40)
axes[row].set_xlim(0.0,1.0)
axes[row].set_ylim(0.0,1.0)
axes[row].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[row].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
axes[row].axvline(x=stance_to_swing, color='maroon',linestyle='dashed',linewidth='3')

row = 4
axes[row].fill_between(xlines2, 0, A_GAS_A1000000, color='gray', edgecolor=None)
axes[row].tick_params(labelsize = 20)
axes[row].set_ylabel('GAS', fontsize=40)
axes[row].set_xlim(0.0,1.0)
axes[row].set_ylim(0.0,1.0)
axes[row].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[row].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
axes[row].axvline(x=stance_to_swing, color='maroon',linestyle='dashed',linewidth='3')

row = 5
axes[row].fill_between(xlines2, 0, A_TA_A1000000, color='gray', edgecolor=None)
axes[row].tick_params(labelsize = 20)
axes[row].set_ylabel('TA', fontsize=40)
axes[row].set_xlim(0.0,1.0)
axes[row].set_ylim(0.0,1.0)
axes[row].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[row].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
axes[row].axvline(x=stance_to_swing, color='maroon',linestyle='dashed',linewidth='3')

row = 6
axes[row].fill_between(xlines2, 0, A_HAM_A1000000, color='gray', edgecolor=None)
axes[row].tick_params(labelsize = 20)
axes[row].set_ylabel('HAM', fontsize=40)
axes[row].set_xlim(0.0,1.0)
axes[row].set_ylim(0.0,1.0)
axes[row].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[row].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
axes[row].axvline(x=stance_to_swing, color='maroon',linestyle='dashed',linewidth='3')

row = 7
axes[row].fill_between(xlines2, 0, A_RF_A1000000, color='gray', edgecolor=None)
axes[row].tick_params(labelsize = 20)
axes[row].set_ylabel('RF', fontsize=40)
axes[row].set_xlim(0.0,1.0)
axes[row].set_ylim(0.0,1.0)
axes[row].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[row].set_xticklabels(['0%','20%','40%','60%','80%','100%'])
axes[row].axvline(x=stance_to_swing, color='maroon',linestyle='dashed',linewidth='3')


fig.subplots_adjust(bottom=0.05)
fig.tight_layout(rect=[0.0, 0.05, 1, 1.0])
fig.text(0.5,0.025,'% gait cycle',fontsize=50, verticalalignment='center', horizontalalignment='center')
# plt.show()
plt.savefig('emg.pdf')