import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


fig = plt.figure(2,figsize=(19.2,10.03*1.5),dpi=100)
ax1 = fig.add_subplot(2,1,1)

with open('tar_vel_connect_review1.pickle', mode='rb') as f:
    load_checkpoint = pickle.load(f)
    vel  = load_checkpoint['vel_evolution']

tar_vel1 = 0.75
tar_vel2 = 1.6
tar_vel3 = 0.75
t = np.linspace(0.0,60,12000)
y = np.linspace(0.0,60,12000)
count = 0
steady = 2000
change_rate = 0.05*0.005
y[:steady-1000] = tar_vel1
count += steady-1000
print(f'count1:{count}')
for i in range(int(abs(tar_vel2-tar_vel1)/change_rate)):
    if tar_vel1<tar_vel2:
        y[count+i] = change_rate + y[count+i-1]
    elif tar_vel1>tar_vel2:
        y[count+i] = -change_rate + y[count+i-1]
count += int(abs(tar_vel2-tar_vel1)/change_rate)
print(f'count2:{count}')
y[count:count+steady] = tar_vel2
count += steady
for i in range(int(abs(tar_vel3-tar_vel2)/change_rate)):
    if tar_vel2<tar_vel3:
        y[count+i] = change_rate + y[count+i-1]
    elif tar_vel2>tar_vel3:
        y[count+i] = -change_rate + y[count+i-1]
count += int(abs(tar_vel3-tar_vel2)/change_rate)
y[count:] = tar_vel3

ax1.plot(t,y,color='black',linestyle='dashed',label='$v^{tar}_{x}$',linewidth=5)
ax1.plot(t,vel,color='orangered',label='$v_{x}$',alpha=0.8)

ax1.legend(loc = 'lower left',fontsize=35,framealpha=1,edgecolor='black',fancybox=False,ncol=2)
ax1.set_xlabel('time [s]',fontsize=40)
ax1.set_ylabel('velocity [m/s]',fontsize=40)
ax1.set_xlim(0,60)
ax1.set_ylim(0,2.0)
ax1.set_xticks([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0])
ax1.set_yticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])

ax1.tick_params(labelsize = 30)
ax1.grid(linestyle='--')



with open('tar_vel_connect_review2.pickle', mode='rb') as f:
    load_checkpoint = pickle.load(f)
    vel  = load_checkpoint['vel_evolution']

ax2 = fig.add_subplot(2,1,2)
ax2.set_xlim(0,40)
ax2.set_ylim(0,2.0)
ax2.set_xticks([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0])
ax2.set_yticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])

tar_vel1 = 1.2
tar_vel2 = 0.8
tar_vel3 = 1.0
tar_vel4 = 1.5
tar_vel5 = 1.3
t = np.linspace(0.0,60,12000)
y = np.linspace(0.0,60,12000)
count = 0
steady = 1600
change_rate = 0.05*0.005
y[:steady-600] = tar_vel1
count += steady-600
print(f'count1:{count}')
for i in range(int(abs(tar_vel2-tar_vel1)/change_rate)):
    if tar_vel1<tar_vel2:
        y[count+i] = change_rate + y[count+i-1]
    elif tar_vel1>tar_vel2:
        y[count+i] = -change_rate + y[count+i-1]
count += int(abs(tar_vel2-tar_vel1)/change_rate)
print(f'count2:{count}')
y[count:count+steady] = tar_vel2
count += steady
for i in range(int(abs(tar_vel3-tar_vel2)/change_rate)):
    if tar_vel2<tar_vel3:
        y[count+i] = change_rate + y[count+i-1]
    elif tar_vel2>tar_vel3:
        y[count+i] = -change_rate + y[count+i-1]
count += int(abs(tar_vel3-tar_vel2)/change_rate)
y[count:count+steady] = tar_vel3
count += steady
for i in range(int(abs(tar_vel4-tar_vel3)/change_rate)):
    if tar_vel3<tar_vel4:
        y[count+i] = change_rate + y[count+i-1]
    elif tar_vel3>tar_vel4:
        y[count+i] = -change_rate + y[count+i-1]
count += int(abs(tar_vel4-tar_vel3)/change_rate)
y[count:count+steady] = tar_vel4
count += steady
for i in range(int(abs(tar_vel5-tar_vel4)/change_rate)):
    if tar_vel4<tar_vel5:
        y[count+i] = change_rate + y[count+i-1]
    elif tar_vel4>tar_vel5:
        y[count+i] = -change_rate + y[count+i-1]
count += int(abs(tar_vel5-tar_vel4)/change_rate)
y[count:] = tar_vel5

ax2.plot(t,y,color='black',linestyle='dashed',label='$v^{tar}_{x}$',linewidth=5)
ax2.plot(t,vel,color='orangered',label='$v_{x}$',alpha=0.8)

ax2.legend(loc = 'lower left',fontsize=35,framealpha=1,edgecolor='black',fancybox=False,ncol=2)
ax2.set_xlabel('time [s]',fontsize=40)
ax2.set_ylabel('velocity [m/s]',fontsize=40)

ax2.tick_params(labelsize = 30)
ax2.grid(linestyle='--')

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(),'tarvel_connect_review.pdf'))