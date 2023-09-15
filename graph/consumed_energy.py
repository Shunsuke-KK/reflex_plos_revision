import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

fig = plt.figure(1,figsize=(19.2,10.03*0.5),dpi=100)
ax = fig.add_subplot(1,1,1)

df_base1 = pd.read_csv(os.path.join(os.getcwd(), 'how_gait_changed_base1_10.csv')) # 1952
df_base2 = pd.read_csv(os.path.join(os.getcwd(), 'how_gait_changed_base2_10.csv')) # 3929
df_base3   = pd.read_csv(os.path.join(os.getcwd(), 'how_gait_changed_target_10.csv')) # 2818

HFL_base1  = np.sum(df_base1.loc[:,'HFL'].values)
HFL_base1 += np.sum(df_base1.loc[:,'L_HFL'].values)
GLU_base1  = np.sum(df_base1.loc[:,'GLU'].values)
GLU_base1 += np.sum(df_base1.loc[:,'L_GLU'].values)
VAS_base1  = np.sum(df_base1.loc[:,'VAS'].values)
VAS_base1 += np.sum(df_base1.loc[:,'L_VAS'].values)
SOL_base1  = np.sum(df_base1.loc[:,'SOL'].values)
SOL_base1 += np.sum(df_base1.loc[:,'L_SOL'].values)
GAS_base1  = np.sum(df_base1.loc[:,'GAS'].values)
GAS_base1 += np.sum(df_base1.loc[:,'L_GAS'].values)
TA_base1   = np.sum(df_base1.loc[:,'TA'].values)
TA_base1  += np.sum(df_base1.loc[:,'L_TA'].values)
HAM_base1  = np.sum(df_base1.loc[:,'HAM'].values)
HAM_base1 += np.sum(df_base1.loc[:,'L_HAM'].values)
RF_base1   = np.sum(df_base1.loc[:,'RF'].values)
RF_base1  += np.sum(df_base1.loc[:,'L_RF'].values)
base1 = np.array([HFL_base1, GLU_base1, VAS_base1, SOL_base1, GAS_base1, TA_base1, HAM_base1, RF_base1])*0.005

HFL_base2  = np.sum(df_base2.loc[:,'HFL'].values)
HFL_base2 += np.sum(df_base2.loc[:,'L_HFL'].values)
GLU_base2  = np.sum(df_base2.loc[:,'GLU'].values)
GLU_base2 += np.sum(df_base2.loc[:,'L_GLU'].values)
VAS_base2  = np.sum(df_base2.loc[:,'VAS'].values)
VAS_base2 += np.sum(df_base2.loc[:,'L_VAS'].values)
SOL_base2  = np.sum(df_base2.loc[:,'SOL'].values)
SOL_base2 += np.sum(df_base2.loc[:,'L_SOL'].values)
GAS_base2  = np.sum(df_base2.loc[:,'GAS'].values)
GAS_base2 += np.sum(df_base2.loc[:,'L_GAS'].values)
TA_base2   = np.sum(df_base2.loc[:,'TA'].values)
TA_base2  += np.sum(df_base2.loc[:,'L_TA'].values)
HAM_base2  = np.sum(df_base2.loc[:,'HAM'].values)
HAM_base2 += np.sum(df_base2.loc[:,'L_HAM'].values)
RF_base2   = np.sum(df_base2.loc[:,'RF'].values)
RF_base2  += np.sum(df_base2.loc[:,'L_RF'].values)
base2 = np.array([HFL_base2, GLU_base2, VAS_base2, SOL_base2, GAS_base2, TA_base2, HAM_base2, RF_base2])*0.005

HFL_base3  = np.sum(df_base3.loc[:,'HFL'].values)
HFL_base3 += np.sum(df_base3.loc[:,'L_HFL'].values)
GLU_base3  = np.sum(df_base3.loc[:,'GLU'].values)
GLU_base3 += np.sum(df_base3.loc[:,'L_GLU'].values)
VAS_base3  = np.sum(df_base3.loc[:,'VAS'].values)
VAS_base3 += np.sum(df_base3.loc[:,'L_VAS'].values)
SOL_base3  = np.sum(df_base3.loc[:,'SOL'].values)
SOL_base3 += np.sum(df_base3.loc[:,'L_SOL'].values)
GAS_base3  = np.sum(df_base3.loc[:,'GAS'].values)
GAS_base3 += np.sum(df_base3.loc[:,'L_GAS'].values)
TA_base3   = np.sum(df_base3.loc[:,'TA'].values)
TA_base3  += np.sum(df_base3.loc[:,'L_TA'].values)
HAM_base3  = np.sum(df_base3.loc[:,'HAM'].values)
HAM_base3 += np.sum(df_base3.loc[:,'L_HAM'].values)
RF_base3   = np.sum(df_base3.loc[:,'RF'].values)
RF_base3  += np.sum(df_base3.loc[:,'L_RF'].values)
base3 = np.array([HFL_base3, GLU_base3, VAS_base3, SOL_base3, GAS_base3, TA_base3, HAM_base3, RF_base3])*0.005

width = 0.25
xlines = np.arange(8)
ax.bar([i-width for i in xlines], base1, align='center', width=0.25, color='tab:blue', label='$A=1$')
ax.bar(xlines, base3, align='center', width=0.25, color='tab:pink', label='modulate reflex circuit 1&2')
ax.bar([i+width for i in xlines], base2, align='center', width=0.25, color='tab:purple', label='$A=10^{6}$')
ax.tick_params(labelsize = 30)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax.set_xticklabels(['HFL', 'GLU', 'VAS', 'SOL', 'GAS', 'TA', 'HAM', 'RF'])
ax.set_ylim(0,2800)
ax.legend(loc = 'upper left',fontsize=25,framealpha=1,edgecolor='black',fancybox=False,ncol=1)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(),'consumed_energy.pdf'))