import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(19.2*0.8,10.03))
ax1 = fig.add_subplot(1,1,1)


###1
parasets = ['base1', 'base2', 'tar_params_1', 'tar_params_2', 'tar_params_12']
datas = ['relative_normal1.pickle', 'relative_normal2.pickle', 'relative_normal3.pickle']
culCoT_base1  = np.empty(0)
culCoT_base2  = np.empty(0)
culCoT_para1  = np.empty(0)
culCoT_para2  = np.empty(0)
culCoT_para12 = np.empty(0)

data_dic = {}
degree = 2
for data in datas:
    if data=='relative_normal1.pickle':
        vel_min = 0.7
        vel_max = 1.6
    elif data=='relative_normal2.pickle':
        vel_min = 0.8
        vel_max = 1.6
    elif data=='relative_normal3.pickle':
        vel_min = 0.7
        vel_max = 1.5
    xlines = np.linspace(vel_min,vel_max,200)
    delta_x = (vel_max-vel_min)/200

    with open(data, mode='rb') as g:
        plot_data = pickle.load(g)

    for para in parasets:
        # we did not include data from the generated gait at the target velocity of 1.6 m/s in "relative_normal1.picke" to calculate the estimated curve 
        # because the generated walking velocity was changed by 0.1 m/s when modulating circuit 2.
        if data=='relative_normal1.pickle':
            vel = np.delete(plot_data[para]['vel'],[-1])
            cot = np.delete(plot_data[para]['cot'],[-1])
        else:
            vel = plot_data[para]['vel']
            cot = plot_data[para]['cot']

        res = np.polyfit(vel,cot,degree)
        ylines = np.poly1d(res)(xlines)
        integral_y = np.sum(ylines*delta_x)

        if para==parasets[0]:
            culCoT_base1 = np.append(culCoT_base1,integral_y)
        elif para==parasets[1]:
            culCoT_base2 = np.append(culCoT_base2,integral_y)
        elif para==parasets[2]:
            culCoT_para1 = np.append(culCoT_para1,integral_y)
        elif para==parasets[3]:
            culCoT_para2 = np.append(culCoT_para2,integral_y)
        elif para==parasets[4]:
            culCoT_para12= np.append(culCoT_para12,integral_y)

culCoT_base2  /= culCoT_base1
culCoT_para1  /= culCoT_base1
culCoT_para2  /= culCoT_base1
culCoT_para12 /= culCoT_base1
culCoT_base1  /= culCoT_base1

relative_cot_mean = np.array([np.mean(culCoT_base1), np.mean(culCoT_para1), np.mean(culCoT_para2), np.mean(culCoT_para12), np.mean(culCoT_base2)])
relative_cot_std  = np.array([np.std(culCoT_base1), np.std(culCoT_para1), np.std(culCoT_para2), np.std(culCoT_para12), np.std(culCoT_base2)])

xlines = [i for i in range(5)]
ax1.bar(xlines, relative_cot_mean, align='center', width=0.5, yerr=relative_cot_std, ecolor='black', capsize=10, error_kw={'elinewidth':5}, color=['tab:blue', 'tab:cyan', 'tab:olive', 'tab:pink', 'tab:purple'])
ax1.set_ylim(0.97,1.01)
ax1.tick_params(labelsize = 30)
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xticklabels(['$A=1$', 'modulate\nreflex\ncircuit 1', 'modulate\nreflex\ncircuit 2','modulate\nreflex\ncircuit 1&2','$A=10^{6}$'])
ax1.set_yticks([1.01, 1.0, 0.99, 0.98])
ax1.set_yticklabels(['101%','100%', '99%', '98%'])
ax1.set_ylabel('relative $\int\mathrm{CoT}$ values to $A=1$\n(n=3)', fontsize=40)
ax1.axhline(y=1.0, color='maroon',linestyle='dashed',linewidth='3')
ax1.axhline(y=np.mean(culCoT_base2), color='maroon',linestyle='dashed',linewidth='3')

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'relative_int_cot.pdf'))