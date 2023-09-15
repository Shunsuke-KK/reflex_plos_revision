import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

folder_name1 = 'review_back_1'
folder_name2 = 'review_forw_1'

save_path = os.path.join(os.getcwd(),'reflex_opt/save_data')
save_folder1 = save_path + '/' + folder_name1
save_folder2 = save_path + '/' + folder_name2


if __name__ == '__main__':
        best_value_change = {}
        value_mean = {}

        num = 0
        vel = 12
        while True:
            filename = f'vel_gen{num}.pickle'
            if os.path.exists(os.path.join(save_folder1,filename)):
                print(vel)
                with open(os.path.join(save_folder1,filename), mode='rb') as f:
                    load_checkpoint = pickle.load(f)
                    best_value_change[str(vel)] = load_checkpoint['best_value_change']
                    value_mean[str(vel)] = load_checkpoint['value_mean']
                num += 1
                vel -= 1
            else:
                break

        num = 0
        vel = 13
        while True:
            filename = f'vel_gen{num}.pickle'
            if os.path.exists(os.path.join(save_folder2,filename)):
                print(vel)
                with open(os.path.join(save_folder2,filename), mode='rb') as f:
                    load_checkpoint = pickle.load(f)
                    best_value_change[str(vel)] = load_checkpoint['best_value_change']
                    value_mean[str(vel)] = load_checkpoint['value_mean']
                num += 1
                vel += 1
            else:
                break



fig = plt.figure(1,figsize=(19.2,10.03),dpi=100)
ax = fig.add_subplot(1,1,1)
for i in range(4,21):
    xlines = np.linspace(0,300,len(best_value_change[str(i)]))
    ax.plot(xlines,best_value_change[str(i)],color=cm.hsv(i/17),linewidth=3.0,label=f'{str(round(0.1*i,1))} m/s')
ax.set_xlim(0,300)
ax.set_xticks([0,100,200,300])
ax.set_ylim(-5000,7000)
ax.set_yticks([-5000, -2500, 0, 2500, 5000])
ax.set_xlabel('generation number ($G$)',fontsize=40)
ax.set_ylabel('best cost value',fontsize=40)
ax.tick_params(labelsize = 30)
ax.grid(linestyle='--')
ax.legend(loc = 'best',fontsize=30,framealpha=1,edgecolor='black',fancybox=False,ncol=4)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(),'optimization_develope.pdf'))