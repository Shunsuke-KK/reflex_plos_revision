from matplotlib import colors, patches
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import os


fig = plt.figure(1,figsize=(19.2*0.5,10.03),dpi=100)
ax1 = fig.add_subplot(1,1,1)
vel = [0.7146674663184464, 0.7771088114913853, 0.8816546132434374, 0.9909372819566468, 1.096746312964077, 1.2278056146065979, 1.313709012916247, 1.431525516685448, 1.4899242240783501, 1.5834055325282106]
# tar_vel = [0.1*(i+5) for i in range(len(vel))]
tar_vel = [0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
ax1.plot([0.0,2.0],[0.0,2.0],color='black',linestyle='dashed',label='$v_{x}=v^{tar}_{x}$')
ax1.scatter(tar_vel, vel, s=350,color='orangered',marker='*',label='measured velocity')

ax1.legend(loc = 'upper left',fontsize=30,framealpha=1,edgecolor='black',fancybox=False,ncol=1)
ax1.set_xlabel('$v^{tar}_{x} [m/s]$',fontsize=40)
ax1.set_ylabel('$v_{x}$ [m/s]',fontsize=40)
ax1.set_xlim(0.6,1.8)
ax1.set_ylim(0.6,1.8)
ax1.set_xticks([0.6,0.9,1.2,1.5,1.8])
ax1.set_yticks([0.6,0.9,1.2,1.5,1.8])
ax1.tick_params(labelsize = 30)
# ax2.tick_params(labelsize = 30)
ax1.grid(linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'tarvel_vel.pdf'))