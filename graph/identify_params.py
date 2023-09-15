import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

data = {}
coefficient = {}

ctrl_params_label=[
    'p_SOL', 'p_TA', 'p_GAS', 'p_VAS', 'p_HAM', 'p_RF', 'p_GLU', 'p_HFL', 'q_SOL', 'q_TA', 'q_GAS', 'q_VAS', 'q_HAM', 'q_RF', 'q_GLU', 'q_HFL', 
    'G_SOL', 'G_TA_stance', 'G_TA_swing', 'G_SOL_TA', 'G_GAS', 'G_VAS', 'G_HAM', 'G_GLU', 'G_HFL', 'G_HAMHFL', 'l_off_TA_stance', 'l_off_TA_swing',
    'l_off_HFL', 'l_off_HAM', 'phi_k_off', 'theta_ref', 'k_phi', 'k_lean', 'kp_HAM', 'kp_GLU', 'kp_HFL', 'kd_HAM','kd_GLU', 'kd_HFL', 
    'delta_S_GLU', 'delta_S_HFL', 'delta_S_RF', 'delta_S_VAS', 'd_DS', 'd_SP', 'kp_SP_VAS', 'kp_SP_GLU', 'kp_SP_HFL', 'kd_SP_VAS', 'kd_SP_GLU', 'kd_SP_HFL', 
    'phi_k_off_SP', 'phi_h_off_SP', 'c_d', 'c_v']

ctrl_params_label_latex=[
    '$p_{SOL}$', '$p_{TA}$', '$p_{GAS}$', '$p_{VAS}$', '$p_{HAM}$', '$p_{RF}$', '$p_{GLU}$', '$p_{HFL}$', '$q_{SOL}$', '$q_{TA}$', '$q_{GAS}$', '$q_{VAS}$', '$q_{HAM}$', '$q_{RF}$', '$q_{GLU}$', '$q_{HFL}$', 
    '$G_{SOL}$', '$G_{TA_st}$', '$G_{TA_sw}$', '$G_{SOL\_TA}$', '$G_{GAS}$', '$G_{VAS}$', '$G_{HAM}$', '$G_{GLU}$', '$G_{HFL}$', '$G_{HAM\_HFL}$', '$l^{tar}_{TA\_{st}}$', '$l^{tar}_{TA\_{sw}}$', 
    '$l^{tar}_{HFL}$', '$l^{tar}_{HAM}$', '$\\theta^{off}_{k}$', '$\\theta^{tar}_{t}$', '$k_{\\theta_{k}}$', '$k_{lean}$', '$K_{HAM}$', '$K_{GLU}$', '$K_{HFL}$', '$D_{HAM}$','$D_{GLU}$', '$D_{HFL}$', 
    '$s_{GLU}$', '$s_{HFL}$', '$s_{RF}$', '$s_{VAS}$', '$\\theta_{DS}$', '$\\theta_{SP}$', '$K_{SP\_VAS}$', '$K_{SP\_GLU}$', '$K_{SP\_HFL}$', '$D_{SP\_VAS}$', '$D_{SP\_GLU}$', '$D_{SP\_HFL}$', 
    '$\\theta^{tar}_{k}$','$\\theta^{tar}_{h}$', '$c_{d}$', '$c_{v}$',]


# 1: normal1 0.7m/s-1.6m/s
degree = 2
xlines = np.linspace(0.7,1.6,200)
delta_x = (1.6-0.7)/200
with open(os.path.join('identify_normal1.pickle'), mode='rb') as g:
    plot_data = pickle.load(g)
    num = 0
    vel0 = plot_data['base']['vel']
    cot0 = plot_data['base']['cot']
    res = np.polyfit(vel0,cot0,degree)
    base_cot = np.poly1d(res)(xlines)
    integral_base = np.sum(base_cot*delta_x)
    for para in ctrl_params_label:
        vel = plot_data[para]['vel']
        cot = plot_data[para]['cot']
        res = np.polyfit(vel,cot,degree)
        estimeted_cot = np.poly1d(res)(xlines)
        integral_para = np.sum(estimeted_cot*delta_x)
        dev = np.array([abs(integral_para-integral_base)/integral_base])
        data[para] = dev

# 2: normal2 0.8m/s-1.6m/s
degree = 2
xlines = np.linspace(0.8,1.6,200)
delta_x = (1.6-0.8)/200
with open(os.path.join('identify_normal2.pickle'), mode='rb') as g:
    plot_data = pickle.load(g)
    num = 0
    vel0 = plot_data['base']['vel']
    cot0 = plot_data['base']['cot']
    res = np.polyfit(vel0,cot0,degree)
    base_cot = np.poly1d(res)(xlines)
    integral_base = np.sum(base_cot*delta_x)
    for para in ctrl_params_label:
        vel = plot_data[para]['vel']
        cot = plot_data[para]['cot']
        res = np.polyfit(vel,cot,degree)
        estimeted_cot = np.poly1d(res)(xlines)
        integral_para = np.sum(estimeted_cot*delta_x)
        dev = abs(integral_para-integral_base)/integral_base
        data[para] = np.append(data[para], dev)

# 3: normal3 0.8m/s-1.5m/s
xlines = np.linspace(0.8,1.5,200)
delta_x = (1.5-0.8)/200
with open(os.path.join('identify_normal3.pickle'), mode='rb') as g:
    plot_data = pickle.load(g)
    num = 0
    vel0 = plot_data['base']['vel']
    cot0 = plot_data['base']['cot']
    res = np.polyfit(vel0,cot0,degree)
    base_cot = np.poly1d(res)(xlines)
    integral_base = np.sum(base_cot*delta_x)
    for para in ctrl_params_label:
        vel = plot_data[para]['vel']
        cot = plot_data[para]['cot']
        res = np.polyfit(vel,cot,degree)
        estimeted_cot = np.poly1d(res)(xlines)
        integral_para = np.sum(estimeted_cot*delta_x)
        dev = abs(integral_para-integral_base)/integral_base
        data[para] = np.append(data[para], dev)


dev_mean = np.empty(0)
dev_std = np.empty(0)
for para in ctrl_params_label:
    dev_mean = np.append(dev_mean,np.mean(data[para]))
    dev_std = np.append(dev_std,np.std(data[para]))


rank = np.argsort(dev_mean)[::-1]
dev_mean_sorted = np.empty(0)
dev_std_sorted = np.empty(0)
label_latex_sorted = []

for i in rank:
    dev_mean_sorted = np.append(dev_mean_sorted,dev_mean[i])
    dev_std_sorted = np.append(dev_std_sorted,dev_std[i])
    label_latex_sorted.append(ctrl_params_label_latex[i])


fig = plt.figure(1,figsize=(19.2*1.5,10.03),dpi=100)
ax = fig.add_subplot(1,1,1)

xlines = [i for i in range(len(ctrl_params_label))]

color = ['lightskyblue' for i in range(len(ctrl_params_label))]
color[0] = 'lightcoral'
color[1] = 'lightcoral'
color[3] = 'lightcoral'
color[13] = 'lightcoral'
ax.bar(xlines, dev_mean_sorted, align='center', width=0.5, yerr=dev_std_sorted, ecolor='black', capsize=10, error_kw={'elinewidth':2,'lw':2}, color=color)
ax.tick_params(axis='x',labelsize = 30)
ax.tick_params(axis='y',labelsize = 30)
x = np.arange(len(ctrl_params_label))
ax.set_xticks(x)
ax.set_xticklabels(ctrl_params_label_latex)
ax.set_xticklabels(label_latex_sorted, rotation=90)
ax.set_yticks([0, 0.005, 0.010, 0.015, 0.020])
ax.set_yticklabels(['0%', '0.5%', '1%', '1.5%', '2.0%'])
ax.set_ylabel('$\Delta|\int\mathrm{CoT}|$  (n=3)', fontsize=40)
ax.set_xlim(min(x)-0.5,max(x)+0.5)
ax.set_ylim(0,0.02)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(),'identify_params.pdf'))