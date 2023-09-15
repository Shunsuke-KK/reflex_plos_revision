import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(19.2*1.7,10.03*1.5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

data = 'variousA_normal1.pickle'
with open(data, mode='rb') as g:
    plot_data = pickle.load(g)
A_candidate = [1, 10, 100, 10000, 1000000]
x1 = plot_data[str(A_candidate[0])]['vel']
y1 = plot_data[str(A_candidate[0])]['cot']
x2 = plot_data[str(A_candidate[1])]['vel']
y2 = plot_data[str(A_candidate[1])]['cot']
x3 = plot_data[str(A_candidate[2])]['vel']
y3 = plot_data[str(A_candidate[2])]['cot']
x4 = plot_data[str(A_candidate[3])]['vel']
y4 = plot_data[str(A_candidate[3])]['cot']
x5 = plot_data[str(A_candidate[4])]['vel']
y5 = plot_data[str(A_candidate[4])]['cot']

ax1.set_xlim(0.65,1.65)
ax1.set_xlabel('$v_{x}$ [m/s]',fontsize=50)
ax1.set_ylabel('CoT',fontsize=50)
ax1.tick_params(labelsize = 40)
ax1.grid(linestyle='--')
ax1.set_xticks([0.7,1.0,1.3,1.6])
ax1.set_yticks([0.5, 0.6, 0.7, 0.8])

degree = 2
xlines = np.linspace(0.7,1.6,200)
delta_x = (1.6-0.7)/200

res1 = np.polyfit(x1,y1,2)
print('res1 = ',res1)
cot_curve_A1 = np.poly1d(res1)(xlines)
int_cot_A1 = np.sum(cot_curve_A1*delta_x)

res2 = np.polyfit(x2,y2,2)
print('res2 = ',res2)
cot_curve_A10 = np.poly1d(res2)(xlines)
int_cot_A10 = np.sum(cot_curve_A10*delta_x)

res3 = np.polyfit(x3,y3,2)
print('res3 = ',res3)
cot_curve_A100 = np.poly1d(res3)(xlines)
int_cot_A100 = np.sum(cot_curve_A100*delta_x)

res4 = np.polyfit(x4,y4,2)
print('res4 = ',res4)
cot_curve_A10000 = np.poly1d(res4)(xlines)
int_cot_A10000 = np.sum(cot_curve_A10000*delta_x)

res5 = np.polyfit(x5,y5,2)
print('res5 = ',res5)
cot_curve_A1000000 = np.poly1d(res5)(xlines)
int_cot_A1000000 = np.sum(cot_curve_A1000000*delta_x)

ax1.plot(xlines,cot_curve_A1,linewidth=5.0,linestyle='dashed',label='A=1',color='tab:blue')
ax1.plot(xlines,cot_curve_A10,linewidth=5.0,linestyle='dashed',label='A=10',color='tab:orange')
ax1.plot(xlines,cot_curve_A100,linewidth=5.0,linestyle='dashed',label='$A=10^{2}$',color='tab:green')
ax1.plot(xlines,cot_curve_A10000,linewidth=5.0,linestyle='dashed',label='$A=10^{4}$',color='tab:red')
ax1.plot(xlines,cot_curve_A1000000,linewidth=5.0,linestyle='dashed',label='$A=10^{6}$',color='tab:purple')
ax1.scatter(x1, y1, s=350,marker='o')
ax1.scatter(x2, y2, s=350,marker='o')
ax1.scatter(x3, y3, s=350,marker='o')
ax1.scatter(x4, y4, s=350,marker='o')
ax1.scatter(x5, y5, s=350,marker='o')
ax1.legend(loc = 'best',fontsize=35,framealpha=1,edgecolor='black',fancybox=False,ncol=3)



# 1: f14b10 0.7m/s-1.6m/s
cot_cul1 = np.array([cot_curve_A1, cot_curve_A10, cot_curve_A100, cot_curve_A10000, cot_curve_A1000000])
cot_cul1 /= cot_cul1[0]

# 2: f30b25 0.8m/s-1.6m/s
data = 'variousA_normal2.pickle'
with open(data, mode='rb') as g:
    plot_data = pickle.load(g)
A_candidate = [1, 10, 100, 10000, 1000000]
x1 = plot_data[str(A_candidate[0])]['vel']
y1 = plot_data[str(A_candidate[0])]['cot']
x2 = plot_data[str(A_candidate[1])]['vel']
y2 = plot_data[str(A_candidate[1])]['cot']
x3 = plot_data[str(A_candidate[2])]['vel']
y3 = plot_data[str(A_candidate[2])]['cot']
x4 = plot_data[str(A_candidate[3])]['vel']
y4 = plot_data[str(A_candidate[3])]['cot']
x5 = plot_data[str(A_candidate[4])]['vel']
y5 = plot_data[str(A_candidate[4])]['cot']

degree = 2
xlines = np.linspace(0.8,1.6,200)
delta_x = (1.6-0.8)/200

res1 = np.polyfit(x1,y1,2)
cot_curve_A1 = np.poly1d(res1)(xlines)
int_cot_A1 = np.sum(cot_curve_A1*delta_x)

res2 = np.polyfit(x2,y2,2)
cot_curve_A10 = np.poly1d(res2)(xlines)
int_cot_A10 = np.sum(cot_curve_A10*delta_x)

res3 = np.polyfit(x3,y3,2)
cot_curve_A100 = np.poly1d(res3)(xlines)
int_cot_A100 = np.sum(cot_curve_A100*delta_x)

res4 = np.polyfit(x4,y4,2)
cot_curve_A10000 = np.poly1d(res4)(xlines)
int_cot_A10000 = np.sum(cot_curve_A10000*delta_x)

res5 = np.polyfit(x5,y5,2)
cot_curve_A1000000 = np.poly1d(res5)(xlines)
int_cot_A1000000 = np.sum(cot_curve_A1000000*delta_x)

cot_cul2 = np.array([cot_curve_A1, cot_curve_A10, cot_curve_A100, cot_curve_A10000, cot_curve_A1000000])
cot_cul2 /= cot_cul2[0]

# 3: f29_b9 0.8m/s-1.5m/s
data = 'variousA_normal3.pickle'
with open(data, mode='rb') as g:
    plot_data = pickle.load(g)
A_candidate = [1, 10, 100, 10000, 1000000]
x1 = plot_data[str(A_candidate[0])]['vel']
y1 = plot_data[str(A_candidate[0])]['cot']
x2 = plot_data[str(A_candidate[1])]['vel']
y2 = plot_data[str(A_candidate[1])]['cot']
x3 = plot_data[str(A_candidate[2])]['vel']
y3 = plot_data[str(A_candidate[2])]['cot']
x4 = plot_data[str(A_candidate[3])]['vel']
y4 = plot_data[str(A_candidate[3])]['cot']
x5 = plot_data[str(A_candidate[4])]['vel']
y5 = plot_data[str(A_candidate[4])]['cot']

degree = 2
xlines = np.linspace(0.8,1.5,200)
delta_x = (1.5-0.8)/200

res1 = np.polyfit(x1,y1,2)
cot_curve_A1 = np.poly1d(res1)(xlines)
int_cot_A1 = np.sum(cot_curve_A1*delta_x)

res2 = np.polyfit(x2,y2,2)
cot_curve_A10 = np.poly1d(res2)(xlines)
int_cot_A10 = np.sum(cot_curve_A10*delta_x)

res3 = np.polyfit(x3,y3,2)
cot_curve_A100 = np.poly1d(res3)(xlines)
int_cot_A100 = np.sum(cot_curve_A100*delta_x)

res4 = np.polyfit(x4,y4,2)
cot_curve_A10000 = np.poly1d(res4)(xlines)
int_cot_A10000 = np.sum(cot_curve_A10000*delta_x)

res5 = np.polyfit(x5,y5,2)
cot_curve_A1000000 = np.poly1d(res5)(xlines)
int_cot_A1000000 = np.sum(cot_curve_A1000000*delta_x)

cot_cul3 = np.array([cot_curve_A1, cot_curve_A10, cot_curve_A100, cot_curve_A10000, cot_curve_A1000000])
cot_cul3 /= cot_cul3[0]



relative_cot_mean = np.empty(0)
relative_cot_std = np.empty(0)
for i in range(5):
    temp = np.array([cot_cul1[i], cot_cul2[i], cot_cul3[i]])
    relative_cot_mean = np.append(relative_cot_mean, np.mean(temp))
    relative_cot_std = np.append(relative_cot_std, np.std(temp))

xlines = [i for i in range(5)]
ax2.bar(xlines, relative_cot_mean, align='center', width=0.5, yerr=relative_cot_std, ecolor='black', capsize=10, error_kw={'elinewidth':5}, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
ax2.set_ylim(0.98,1.005)
ax2.tick_params(labelsize = 40)
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.set_xticklabels(['$A=1$' ,'$A=10$','$A=10^{2}$','$A=10^{4}$','$A=10^{6}$'])
ax2.set_yticks([1.0, 0.99, 0.98])
ax2.set_yticklabels(['100%', '99%', '98%'])
ax2.set_ylabel('relative $\int\mathrm{CoT}$ values to $A=1$\n(n=3)', fontsize=50)
ax2.axhline(y=1.0, color='maroon',linestyle='dashed',linewidth='3')
fig.tight_layout(rect=[0,0,1,1])
fig.subplots_adjust(wspace=0.3)

plt.savefig(os.path.join(os.getcwd(),'contribution_PWLS.pdf'))