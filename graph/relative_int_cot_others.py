import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(19.2*1.6,10.03))
ax2 = fig.add_subplot(1,3,1)
ax3 = fig.add_subplot(1,3,2)
ax4 = fig.add_subplot(1,3,3)


### different body structure
parasets = ['base1', 'base2', 'tar_params_1', 'tar_params_2', 'tar_params_12']
datas = ['relative_shleg1.pickle', 'relative_shleg2.pickle']
culCoT_base1  = np.empty(0)
culCoT_base2  = np.empty(0)
culCoT_para1  = np.empty(0)
culCoT_para2  = np.empty(0)
culCoT_para12 = np.empty(0)

data_dic = {}

degree = 2
for data in datas:
    if data=='relative_shleg1.pickle':
        vel_min = 0.6
        vel_max = 1.0
    elif data=='relative_shleg2.pickle':
        vel_min = 0.6
        vel_max = 1.0

    xlines = np.linspace(vel_min,vel_max,200)
    delta_x = (vel_max-vel_min)/200

    with open(data, mode='rb') as g:
        plot_data = pickle.load(g)

    for para in parasets:
        # we did not include data from the generated gait at the target velocity of 0.8 m/s in "relative_shleg1.picke" to calculate the estimated curve 
        # because the generated walking velocity was changed by 0.1 m/s when modulating circuit 2.
        if data=='relative_shleg1.pickle':
            vel = np.delete(plot_data[para]['vel'],[2])
            cot = np.delete(plot_data[para]['cot'],[2])
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
ax2.bar(xlines, relative_cot_mean, align='center', width=0.5, yerr=relative_cot_std, ecolor='black', capsize=10, error_kw={'elinewidth':5}, color=['tab:blue', 'tab:cyan', 'tab:olive', 'tab:pink', 'tab:purple'])
ax2.set_title('different body structure',fontsize='45')
ax2.set_ylim(0.965,1.01)
ax2.tick_params(labelsize = 30)
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.set_xticklabels(['$A=1$', 'circuit\n1', 'circuit\n2','circuit\n1&2','$A=10^{6}$'])
ax2.set_yticks([1.01, 1.0, 0.99, 0.98, 0.97])
ax2.set_yticklabels(['101%','100%', '99%', '98%', '97%'])
ax2.axhline(y=1.0, color='maroon',linestyle='dashed',linewidth='3')
ax2.axhline(y=np.mean(culCoT_base2), color='maroon',linestyle='dashed',linewidth='3')



### different neural sysytem
parasets = ['base1', 'base2', 'tar_params_1', 'tar_params_2', 'tar_params_12']
datas = ['relative_timedelay1.pickle', 'relative_timedelay2.pickle']
culCoT_base1  = np.empty(0)
culCoT_base2  = np.empty(0)
culCoT_para1  = np.empty(0)
culCoT_para2  = np.empty(0)
culCoT_para12 = np.empty(0)

data_dic = {}
degree = 2
for data in datas:
    if data=='relative_timedelay1.pickle':
        vel_min = 0.8
        vel_max = 1.7
    elif data=='relative_timedelay2.pickle':
        vel_min = 1.0
        vel_max = 1.6

    xlines = np.linspace(vel_min,vel_max,200)
    delta_x = (vel_max-vel_min)/200

    with open(data, mode='rb') as g:
        plot_data = pickle.load(g)

    for para in parasets:
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
ax3.bar(xlines, relative_cot_mean, align='center', width=0.5, yerr=relative_cot_std, ecolor='black', capsize=10, error_kw={'elinewidth':5}, color=['tab:blue', 'tab:cyan', 'tab:olive', 'tab:pink', 'tab:purple'])
ax3.set_title('different neural sysytem',fontsize='45')
ax3.set_ylim(0.965,1.01)
ax3.tick_params(labelsize = 30)
ax3.set_xticks([0, 1, 2, 3, 4])
ax3.set_xticklabels(['$A=1$', 'circuit\n1', 'circuit\n2','circuit\n1&2','$A=10^{6}$'])
ax3.set_yticks([1.01, 1.0, 0.99, 0.98, 0.97])
ax3.set_yticklabels(['101%','100%', '99%', '98%', '97%'])
ax3.axhline(y=1.0, color='maroon',linestyle='dashed',linewidth='3')
ax3.axhline(y=np.mean(culCoT_base2), color='maroon',linestyle='dashed',linewidth='3')



###4 differet weight coeeficients in the cost function
parasets = ['base1', 'base2', 'tar_params_1', 'tar_params_2', 'tar_params_12']
datas = ['relative_cost1.pickle', 'relative_cost2.pickle']
culCoT_base1  = np.empty(0)
culCoT_base2  = np.empty(0)
culCoT_para1  = np.empty(0)
culCoT_para2  = np.empty(0)
culCoT_para12 = np.empty(0)

data_dic = {}

degree = 2
for data in datas:
    if data=='relative_cost1.pickle':
        vel_min = 0.8
        vel_max = 1.7
    elif data=='relative_cost2.pickle':
        vel_min = 0.7
        vel_max = 1.6
    xlines = np.linspace(vel_min,vel_max,200)
    delta_x = (vel_max-vel_min)/200

    with open(data, mode='rb') as g:
        plot_data = pickle.load(g)

    for para in parasets:
        if data=='relative_cost2.pickle':
            # we did not include data from the generated gait at the target velocity of 1.5 and 1.6 m/s in "relative_cost2.picke" to calculate the estimated curve 
            # because the generated walking velocity was changed by 0.1 m/s when modulating circuit 2.
            vel = plot_data[para]['vel'][:-2]
            cot = plot_data[para]['cot'][:-2]
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
ax4.bar(xlines, relative_cot_mean, align='center', width=0.5, yerr=relative_cot_std, ecolor='black', capsize=10, error_kw={'elinewidth':5}, color=['tab:blue', 'tab:cyan', 'tab:olive', 'tab:pink', 'tab:purple'])
ax4.set_title('different cost function',fontsize='45')
ax4.set_ylim(0.97,1.01)
ax4.tick_params(labelsize = 30)
ax4.set_xticks([0, 1, 2, 3, 4])
ax4.set_xticklabels(['$A=1$', 'circuit\n1', 'circuit\n2','circuit\n1&2','$A=10^{6}$'])
ax4.set_yticks([1.01, 1.0, 0.99, 0.98])
ax4.set_yticklabels(['101%','100%', '99%', '98%'])
ax4.axhline(y=1.0, color='maroon',linestyle='dashed',linewidth='3')
ax4.axhline(y=np.mean(culCoT_base2), color='maroon',linestyle='dashed',linewidth='3')

fig.tight_layout(rect=[0.06,0,1,1])
fig.text(0.03,0.5,'relative $\int\mathrm{CoT}$ values to $A=1$\n(n=2)',fontsize=45,rotation=90, verticalalignment='center', horizontalalignment='center')
plt.savefig(os.path.join(os.getcwd(),'relative_int_cot_others.pdf'))