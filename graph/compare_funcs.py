import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

fig = plt.figure(1,figsize=(19.2,10.03*0.5),dpi=100)
ax = fig.add_subplot(1,1,1)

df_base1 = pd.read_csv('how_gait_changed_base1_10.csv') # 1952
df_base3   = pd.read_csv('how_gait_changed_target_10.csv') # 2818

def normalize(input_array):
    norm_value = np.empty(0)
    for i in range(99):
        j = 0
        norm_length = 100*np.arange(len(input_array))/len(input_array)
        while True:
            if i>=norm_length[j] and i<norm_length[j+1]:
                dx = (i-norm_length[j])/(norm_length[j+1]-norm_length[j])
                input_value = (1-dx)*input_array[j] + dx*input_array[j+1]
                norm_value = np.append(norm_value, input_value)
                break
            else:
                j+=1
    norm_value = np.append(norm_value, input_array[-1])
    return norm_value

# swing phase for tar_vel=1.0 m/s with A=1
base1_circuit1_1  = normalize(df_base1.loc[333:488,'circuit1'].values)
base1_circuit1_2  = normalize(df_base1.loc[658:813,'circuit1'].values)
base1_circuit1_3  = normalize(df_base1.loc[982:1138,'circuit1'].values)
base1_circuit1_4  = normalize(df_base1.loc[1306:1462,'circuit1'].values)
base1_circuit1_5  = normalize(df_base1.loc[1631:1787,'circuit1'].values)
base1_circuit2_1  = normalize(df_base1.loc[333:488,'circuit2'].values)
base1_circuit2_2  = normalize(df_base1.loc[658:813,'circuit2'].values)
base1_circuit2_3  = normalize(df_base1.loc[982:1138,'circuit2'].values)
base1_circuit2_4  = normalize(df_base1.loc[1306:1462,'circuit2'].values)
base1_circuit2_5  = normalize(df_base1.loc[1631:1787,'circuit2'].values)
base1_u_1  = normalize(df_base1.loc[333+1:488+1,'A_HFL'].values)
base1_u_2  = normalize(df_base1.loc[658+1:813+1,'A_HFL'].values)
base1_u_3  = normalize(df_base1.loc[982+1:1138+1,'A_HFL'].values)
base1_u_4  = normalize(df_base1.loc[1306+1:1462+1,'A_HFL'].values)
base1_u_5  = normalize(df_base1.loc[1631+1:1787+1,'A_HFL'].values)


base3_circuit1_1  = normalize(df_base3.loc[274:425,'circuit1'].values)
base3_circuit1_2  = normalize(df_base3.loc[591:741,'circuit1'].values)
base3_circuit1_3  = normalize(df_base3.loc[1217:1369,'circuit1'].values)
base3_circuit1_4  = normalize(df_base3.loc[1306:1462,'circuit1'].values)
base3_circuit1_5  = normalize(df_base3.loc[1534:1684,'circuit1'].values)
base3_circuit2_1  = normalize(df_base3.loc[274:425,'circuit2'].values)
base3_circuit2_2  = normalize(df_base3.loc[591:741,'circuit2'].values)
base3_circuit2_3  = normalize(df_base3.loc[1217:1369,'circuit2'].values)
base3_circuit2_4  = normalize(df_base3.loc[1306:1462,'circuit2'].values)
base3_circuit2_5  = normalize(df_base3.loc[1534:1684,'circuit2'].values)
base3_u_1  = normalize(df_base3.loc[274+1:425+1,'A_HFL'].values)
base3_u_2  = normalize(df_base3.loc[591+1:741+1,'A_HFL'].values)
base3_u_3  = normalize(df_base3.loc[1217+1:1369+1,'A_HFL'].values)
base3_u_4  = normalize(df_base3.loc[1306+1:1462+1,'A_HFL'].values)
base3_u_5  = normalize(df_base3.loc[1534+1:1684+1,'A_HFL'].values)

plot_base1_circuit1 = np.empty(0)
plot_base1_circuit2 = np.empty(0)
plot_base1_u = np.empty(0)
plot_base3_circuit1 = np.empty(0)
plot_base3_circuit2 = np.empty(0)
plot_base3_u = np.empty(0)
plot_base1_circuit1_std = np.empty(0)
plot_base1_circuit2_std = np.empty(0)
plot_base1_u_std = np.empty(0)
plot_base3_circuit1_std = np.empty(0)
plot_base3_circuit2_std = np.empty(0)
plot_base3_u_std = np.empty(0)
for i in range(100):
    plot_base1_circuit1 = np.append(plot_base1_circuit1, np.mean([base1_circuit1_1[i],base1_circuit1_2[i],base1_circuit1_3[i],base1_circuit1_4[i],base1_circuit1_5[i]]))
    plot_base1_circuit2 = np.append(plot_base1_circuit2, np.mean([base1_circuit2_1[i],base1_circuit2_2[i],base1_circuit2_3[i],base1_circuit2_4[i],base1_circuit2_5[i]]))
    plot_base1_u = np.append(plot_base1_u, np.mean([base1_u_1[i],base1_u_2[i],base1_u_3[i],base1_u_4[i],base1_u_5[i]]))
    plot_base3_circuit1 = np.append(plot_base3_circuit1, np.mean([base3_circuit1_1[i],base3_circuit1_2[i],base3_circuit1_3[i],base3_circuit1_4[i],base3_circuit1_5[i]]))
    plot_base3_circuit2 = np.append(plot_base3_circuit2, np.mean([base3_circuit2_1[i],base3_circuit2_2[i],base3_circuit2_3[i],base3_circuit2_4[i],base3_circuit2_5[i]]))
    plot_base3_u = np.append(plot_base3_u, np.mean([base3_u_1[i],base3_u_2[i],base3_u_3[i],base3_u_4[i],base3_u_5[i]]))
    plot_base1_circuit1_std = np.append(plot_base1_circuit1_std, np.std([base1_circuit1_1[i],base1_circuit1_2[i],base1_circuit1_3[i],base1_circuit1_4[i],base1_circuit1_5[i]]))
    plot_base1_circuit2_std = np.append(plot_base1_circuit2_std, np.std([base1_circuit2_1[i],base1_circuit2_2[i],base1_circuit2_3[i],base1_circuit2_4[i],base1_circuit2_5[i]]))
    plot_base1_u_std = np.append(plot_base1_u_std, np.std([base1_u_1[i],base1_u_2[i],base1_u_3[i],base1_u_4[i],base1_u_5[i]]))
    plot_base3_circuit1_std = np.append(plot_base3_circuit1_std, np.std([base3_circuit1_1[i],base3_circuit1_2[i],base3_circuit1_3[i],base3_circuit1_4[i],base3_circuit1_5[i]]))
    plot_base3_circuit2_std = np.append(plot_base3_circuit2_std, np.std([base3_circuit2_1[i],base3_circuit2_2[i],base3_circuit2_3[i],base3_circuit2_4[i],base3_circuit2_5[i]]))
    plot_base3_u_std = np.append(plot_base3_u_std, np.std([base3_u_1[i],base3_u_2[i],base3_u_3[i],base3_u_4[i],base3_u_5[i]]))

xlines = np.linspace(0.0,100,100)

ax.plot(xlines, plot_base1_u, linewidth=5.0, color='tab:blue', label='$A=1$')
ax.plot(xlines, plot_base3_u, linewidth=5.0, color='tab:pink', label='modulate reflex circuit 1&2')
ax.set_xlim(0,100)
ax.set_ylim(0,0.5)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
ax.tick_params(labelsize = 30)
ax.legend(loc='upper right',fontsize=30,framealpha=1,edgecolor='black',fancybox=False,ncol=1)
ax.set_xlabel('% swing phase', fontsize=35)
ax.set_ylabel('stimulation to\nHFL, $u_{HFL}$', fontsize=35)

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'compare_func.pdf'))