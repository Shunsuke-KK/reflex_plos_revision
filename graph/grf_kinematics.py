import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math

fig, axes = plt.subplots(5, 3, figsize=(19.2*1.5,10.03*2.7))

file_path = 'human_gait_data2.xls'
sheet_name = 'Ground Reaction Forces'

df_grf_ap_s = pd.read_excel(file_path, sheet_name='Ground Reaction Forces', usecols='F:H,', skiprows=2, header=None, nrows=34)
df_grf_ap_n  = pd.read_excel(file_path, sheet_name='Ground Reaction Forces', usecols='I:K,', skiprows=2, header=None, nrows=32)
df_grf_ap_f  = pd.read_excel(file_path, sheet_name='Ground Reaction Forces', usecols='L:N,', skiprows=2, header=None, nrows=32)

df_grf_vr_s  = pd.read_excel(file_path, sheet_name='Ground Reaction Forces', usecols='F:H,', skiprows=53, header=None, nrows=34)
df_grf_vr_n   = pd.read_excel(file_path, sheet_name='Ground Reaction Forces', usecols='I:K,', skiprows=53, header=None, nrows=32)
df_grf_vr_f   = pd.read_excel(file_path, sheet_name='Ground Reaction Forces', usecols='L:N,', skiprows=53, header=None, nrows=32)

df_hip_rotation_s = pd.read_excel(file_path, sheet_name='Joint Rotations', usecols='F:H,', skiprows=308, header=None, nrows=51)
df_hip_rotation_n  = pd.read_excel(file_path, sheet_name='Joint Rotations', usecols='I:K,', skiprows=308, header=None, nrows=51)
df_hip_rotation_f  = pd.read_excel(file_path, sheet_name='Joint Rotations', usecols='L:N,', skiprows=308, header=None, nrows=51)

df_knee_rotation_s = pd.read_excel(file_path, sheet_name='Joint Rotations', usecols='F:H,', skiprows=461, header=None, nrows=51)
df_knee_rotation_n  = pd.read_excel(file_path, sheet_name='Joint Rotations', usecols='I:K,', skiprows=461, header=None, nrows=51)
df_knee_rotation_f  = pd.read_excel(file_path, sheet_name='Joint Rotations', usecols='L:N,', skiprows=461, header=None, nrows=51)

df_ankle_rotation_s = pd.read_excel(file_path, sheet_name='Joint Rotations', usecols='F:H,', skiprows=614, header=None, nrows=51)
df_ankle_rotation_n  = pd.read_excel(file_path, sheet_name='Joint Rotations', usecols='I:K,', skiprows=614, header=None, nrows=51)
df_ankle_rotation_f  = pd.read_excel(file_path, sheet_name='Joint Rotations', usecols='L:N,', skiprows=614, header=None, nrows=51)

df_model_s = pd.read_csv('measured_data_090.csv') # 1952
df_model_n = pd.read_csv('measured_data_125.csv') # 3929
df_model_f = pd.read_csv('measured_data_165.csv') # 2818

### kinematics
def rad_to_degree(rad):
    return rad*(180/math.pi)

textsize_graph = 35

measure_start = 1940
measure_fin = 2273

row, col = 0, 0
footoff = 62
xlines = np.linspace(0.0,footoff,len(df_grf_vr_s))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_s.loc[measure_start:measure_fin,'right_x_N'])), df_model_s.loc[measure_start:measure_fin,'right_x_N']/80/9.8, linewidth=5.0)
axes[row,col].fill_between(xlines,df_grf_ap_s.iloc[:,0], df_grf_ap_s.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
axes[row,col].axhline(y=0.0, color='maroon',linestyle='dashed',linewidth='3')
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_grf_ap_s.iloc[:,1].values))
y_human = df_grf_ap_s.iloc[:,1].values
y_human_modify = np.empty(0)
for i in range(footoff):
    point = 100*i/footoff
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
while len(y_human_modify)<100:
    y_human_modify = np.append(y_human_modify, 0)
x_human = np.linspace(0.0,100,len(y_human_modify))
y_human = y_human_modify
y_model = df_model_s.loc[measure_start:measure_fin,'right_x_N'].values
y_human_modify2 = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify2 = np.append(y_human_modify2, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify2-np.mean(y_human_modify2)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(60,-0.5,'R={:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph )


row, col = 1, 0
axes[row,col].plot(np.linspace(0.0,100,len(df_model_s.loc[measure_start:measure_fin,'right_z_N'])), df_model_s.loc[measure_start:measure_fin,'right_z_N']/80/9.8, linewidth=5.0)
axes[row,col].fill_between(xlines, df_grf_vr_s.iloc[:,0], df_grf_vr_s.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
axes[row,col].axhline(y=0.0, color='maroon',linestyle='dashed',linewidth='3')
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_grf_vr_s.iloc[:,1].values))
y_human = df_grf_vr_s.iloc[:,1].values
y_human_modify = np.empty(0)
for i in range(footoff):
    point = 100*i/footoff
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
while len(y_human_modify)<100:
    y_human_modify = np.append(y_human_modify, 0)
x_human = np.linspace(0.0,100,len(y_human_modify))
y_human = y_human_modify
y_model = df_model_s.loc[measure_start:measure_fin,'right_z_N'].values
y_human_modify2 = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify2 = np.append(y_human_modify2, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify2-np.mean(y_human_modify2)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(60,2,'R={:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph )


row, col = 2, 0
xlines = np.linspace(0.0,100,len(df_hip_rotation_s.iloc[:,0]))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_s.loc[measure_start:measure_fin,'hip'])), rad_to_degree(df_model_s.loc[measure_start:measure_fin,'hip']+df_model_s.loc[measure_start:measure_fin,'theta']), linewidth=5.0)
axes[row,col].fill_between(xlines, df_hip_rotation_s.iloc[:,0], df_hip_rotation_s.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_yticks([-10,0,10,20,30])
axes[row,col].set_xlim(0.0,100)
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_hip_rotation_s.iloc[:,1].values))
y_human = df_hip_rotation_s.iloc[:,1].values
y_model = df_model_s.loc[measure_start:measure_fin,'hip'].values
y_human_modify = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify-np.mean(y_human_modify)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(80,0,'R=\n{:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)


row, col = 3, 0
xlines = np.linspace(0.0,100,len(df_knee_rotation_s.iloc[:,0]))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_s.loc[measure_start:measure_fin,'knee'])), rad_to_degree(abs(df_model_s.loc[measure_start:measure_fin,'knee'])), linewidth=5.0)
axes[row,col].fill_between(xlines, df_knee_rotation_s.iloc[:,0], df_knee_rotation_s.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_knee_rotation_s.iloc[:,1].values))
y_human = df_knee_rotation_s.iloc[:,1].values
y_model = -df_model_s.loc[measure_start:measure_fin,'knee'].values
y_human_modify = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify-np.mean(y_human_modify)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(30,40,'R={:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)


row, col = 4, 0
xlines = np.linspace(0.0,100,len(df_ankle_rotation_s.iloc[:,0]))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_s.loc[measure_start:measure_fin,'ankle'])), rad_to_degree(df_model_s.loc[measure_start:measure_fin,'ankle']), linewidth=5.0)
axes[row,col].fill_between(xlines, df_ankle_rotation_s.iloc[:,0], df_ankle_rotation_s.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_ankle_rotation_s.iloc[:,1].values))
y_human = df_ankle_rotation_s.iloc[:,1].values
y_model = df_model_s.loc[measure_start:measure_fin,'ankle'].values
y_human_modify = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify-np.mean(y_human_modify)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(20,-30,'R=\n{:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)



measure_start = 2483
measure_fin = 2777

row, col = 0, 1
footoff = 59
xlines = np.linspace(0.0,footoff,len(df_grf_vr_n))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_n.loc[measure_start:measure_fin,'right_x_N'])), df_model_n.loc[measure_start:measure_fin,'right_x_N']/80/9.8, linewidth=5.0)
axes[row,col].fill_between(xlines, df_grf_ap_n.iloc[:,0], df_grf_ap_n.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize=20)
axes[row,col].set_xlim(0.0,100)
axes[row,col].axhline(y=0.0, color='maroon',linestyle='dashed',linewidth='3')
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_grf_ap_n.iloc[:,1].values))
y_human = df_grf_ap_n.iloc[:,1].values
y_human_modify = np.empty(0)
for i in range(footoff):
    point = 100*i/footoff
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
while len(y_human_modify)<100:
    y_human_modify = np.append(y_human_modify, 0)
x_human = np.linspace(0.0,100,len(y_human_modify))
y_human = y_human_modify
y_model = df_model_n.loc[measure_start:measure_fin,'right_x_N'].values
y_human_modify2 = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify2 = np.append(y_human_modify2, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify2-np.mean(y_human_modify2)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(60,-0.5,'R={:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)


row, col = 1, 1
axes[row,col].plot(np.linspace(0.0,100,len(df_model_n.loc[measure_start:measure_fin,'right_z_N'])), df_model_n.loc[measure_start:measure_fin,'right_z_N']/80/9.8, linewidth=5.0)
axes[row,col].fill_between(xlines, df_grf_vr_n.iloc[:,0], df_grf_vr_n.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
axes[row,col].axhline(y=0.0, color='maroon',linestyle='dashed',linewidth='3')
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_grf_vr_n.iloc[:,1].values))
y_human = df_grf_vr_n.iloc[:,1].values
y_human_modify = np.empty(0)
for i in range(footoff):
    point = 100*i/footoff
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
while len(y_human_modify)<100:
    y_human_modify = np.append(y_human_modify, 0)
x_human = np.linspace(0.0,100,len(y_human_modify))
y_human = y_human_modify
y_model = df_model_n.loc[measure_start:measure_fin,'right_z_N'].values
y_human_modify2 = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify2 = np.append(y_human_modify2, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify2-np.mean(y_human_modify2)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(60,2,'R={:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph )


row, col = 2, 1
xlines = np.linspace(0.0,100,len(df_hip_rotation_n.iloc[:,0]))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_n.loc[measure_start:measure_fin,'hip'])), rad_to_degree(df_model_n.loc[measure_start:measure_fin,'hip']+df_model_n.loc[measure_start:measure_fin,'theta']), linewidth=5.0)
axes[row,col].fill_between(xlines, df_hip_rotation_n.iloc[:,0], df_hip_rotation_n.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_hip_rotation_n.iloc[:,1].values))
y_human = df_hip_rotation_n.iloc[:,1].values
y_model = df_model_n.loc[measure_start:measure_fin,'hip'].values
y_human_modify = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify-np.mean(y_human_modify)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(80,0,'R=\n{:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)


row, col = 3, 1
xlines = np.linspace(0.0,100,len(df_knee_rotation_n.iloc[:,0]))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_n.loc[measure_start:measure_fin,'knee'])), rad_to_degree(abs(df_model_n.loc[measure_start:measure_fin,'knee'])), linewidth=5.0)
axes[row,col].fill_between(xlines, df_knee_rotation_n.iloc[:,0], df_knee_rotation_n.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_knee_rotation_n.iloc[:,1].values))
y_human = df_knee_rotation_n.iloc[:,1].values
y_model = -df_model_n.loc[measure_start:measure_fin,'knee'].values
y_human_modify = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify-np.mean(y_human_modify)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(30,40,'R={:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)


row, col = 4, 1
xlines = np.linspace(0.0,100,len(df_ankle_rotation_n.iloc[:,0]))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_n.loc[measure_start:measure_fin,'ankle'])), rad_to_degree(df_model_n.loc[measure_start:measure_fin,'ankle']), linewidth=5.0)
axes[row,col].fill_between(xlines, df_ankle_rotation_n.iloc[:,0], df_ankle_rotation_n.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_ankle_rotation_n.iloc[:,1].values))
y_human = df_ankle_rotation_n.iloc[:,1].values
y_model = df_model_n.loc[measure_start:measure_fin,'ankle'].values
y_human_modify = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify-np.mean(y_human_modify)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(20,-35,'R=\n{:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)


measure_start = 1859
measure_fin = 2098

row, col = 0, 2
footoff=58
xlines = np.linspace(0.0,footoff,len(df_grf_vr_f))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_f.loc[measure_start:measure_fin,'right_x_N'])), df_model_f.loc[measure_start:measure_fin,'right_x_N']/80/9.8, linewidth=5.0) # 3537
axes[row,col].fill_between(xlines, df_grf_ap_f.iloc[:,0], df_grf_ap_f.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
axes[row,col].axhline(y=0.0, color='maroon',linestyle='dashed',linewidth='3')
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_grf_ap_f.iloc[:,1].values))
y_human = df_grf_ap_f.iloc[:,1].values
y_human_modify = np.empty(0)
for i in range(footoff):
    point = 100*i/footoff
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
while len(y_human_modify)<100:
    y_human_modify = np.append(y_human_modify, 0)
x_human = np.linspace(0.0,100,len(y_human_modify))
y_human = y_human_modify
y_model = df_model_f.loc[measure_start:measure_fin,'right_x_N'].values
y_human_modify2 = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify2 = np.append(y_human_modify2, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify2-np.mean(y_human_modify2)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(60,-0.5,'R={:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph )



row, col = 1, 2
axes[row,col].plot(np.linspace(0.0,100,len(df_model_f.loc[measure_start:measure_fin,'right_z_N'])), df_model_f.loc[measure_start:measure_fin,'right_z_N']/80/9.8, linewidth=5.0) # 3537
axes[row,col].fill_between(xlines, df_grf_vr_f.iloc[:,0], df_grf_vr_f.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
axes[row,col].axhline(y=0.0, color='maroon',linestyle='dashed',linewidth='3')
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_grf_vr_f.iloc[:,1].values))
y_human = df_grf_vr_f.iloc[:,1].values
y_human_modify = np.empty(0)
for i in range(footoff):
    point = 100*i/footoff
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
while len(y_human_modify)<100:
    y_human_modify = np.append(y_human_modify, 0)
x_human = np.linspace(0.0,100,len(y_human_modify))
y_human = y_human_modify
y_model = df_model_f.loc[measure_start:measure_fin,'right_z_N'].values
y_human_modify2 = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify2 = np.append(y_human_modify2, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify2-np.mean(y_human_modify2)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(60,2,'R={:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)


row, col = 2, 2
xlines = np.linspace(0.0,100,len(df_hip_rotation_f.iloc[:,0]))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_f.loc[measure_start:measure_fin,'hip'])), rad_to_degree(df_model_f.loc[measure_start:measure_fin,'hip']+df_model_f.loc[measure_start:measure_fin,'theta']), linewidth=5.0)
axes[row,col].fill_between(xlines, df_hip_rotation_f.iloc[:,0], df_hip_rotation_f.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_hip_rotation_f.iloc[:,1].values))
y_human = df_hip_rotation_f.iloc[:,1].values
y_model = df_model_f.loc[measure_start:measure_fin,'hip'].values
y_human_modify = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify-np.mean(y_human_modify)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(80,0,'R=\n{:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)


row, col = 3, 2
xlines = np.linspace(0.0,100,len(df_knee_rotation_f.iloc[:,0]))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_f.loc[measure_start:measure_fin,'knee'])), rad_to_degree(abs(df_model_f.loc[measure_start:measure_fin,'knee'])), linewidth=5.0)
axes[row,col].fill_between(xlines, df_knee_rotation_f.iloc[:,0], df_knee_rotation_f.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_knee_rotation_f.iloc[:,1].values))
y_human = df_knee_rotation_f.iloc[:,1].values
y_model = -df_model_f.loc[measure_start:measure_fin,'knee'].values
y_human_modify = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify-np.mean(y_human_modify)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(30,40,'R={:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)


row, col = 4, 2
xlines = np.linspace(0.0,100,len(df_ankle_rotation_f.iloc[:,0]))
axes[row,col].plot(np.linspace(0.0,100,len(df_model_f.loc[measure_start:measure_fin,'ankle'])), rad_to_degree(df_model_f.loc[measure_start:measure_fin,'ankle']), linewidth=5.0)
axes[row,col].fill_between(xlines, df_ankle_rotation_f.iloc[:,0], df_ankle_rotation_f.iloc[:,2], color='gray', alpha=0.5, edgecolor=None)
axes[row,col].tick_params(labelsize = 20)
axes[row,col].set_xlim(0.0,100)
# cross-correlation values
x_human = np.linspace(0.0,100,len(df_ankle_rotation_f.iloc[:,1].values))
y_human = df_ankle_rotation_f.iloc[:,1].values
y_model = df_model_f.loc[measure_start:measure_fin,'ankle'].values
y_human_modify = np.empty(0)
for i in range(len(y_model)):
    point = 100*i/len(y_model)
    j = 0
    while True:
        if point >= x_human[j] and point < x_human[j+1]:
            ratio = 1-(point-x_human[j])/(x_human[j+1]-x_human[j])
            y_human_modify = np.append(y_human_modify, ratio*y_human[j]+(1-ratio)*y_human[j+1])
            break
        j += 1
human_dev = y_human_modify-np.mean(y_human_modify)
model_dev = y_model-np.mean(y_model)
xc = np.dot(human_dev, model_dev)
xc /= np.linalg.norm(human_dev, ord=2)*np.linalg.norm(model_dev,ord=2)
axes[row,col].text(20,-30,'R=\n{:.3f}'.format(xc),ha='center',va='center',fontsize=textsize_graph)



### coordinate graph
fig.subplots_adjust(left=0.1, top=0.95, bottom=0.05, hspace=0.5)
fig.tight_layout(rect=[0.08,0.05,1,0.95])
fig.text(0.1+1/6*0.9,0.975,'slow',fontsize=50,verticalalignment='center', horizontalalignment='center')
fig.text(0.1+3/6*0.9,0.975,'normal',fontsize=50,verticalalignment='center', horizontalalignment='center')
fig.text(0.1+5/6*0.9,0.975,'fast',fontsize=50,verticalalignment='center', horizontalalignment='center')

fig.text(0.05,0.95-1/10*0.9,'$\mathrm{GRF}_{x}$\n(N / bw)',fontsize=40,rotation=90, verticalalignment='center', horizontalalignment='center')
fig.text(0.05,0.95-3/10*0.9,'$\mathrm{GRF}_{z}$\n(N / bw)',fontsize=40,rotation=90, verticalalignment='center', horizontalalignment='center')
fig.text(0.05,0.95-5/10*0.9,'hip flexion\n(degree)',fontsize=40,rotation=90, verticalalignment='center', horizontalalignment='center')
fig.text(0.05,0.95-7/10*0.9,'knee flexion\n(degree)',fontsize=40,rotation=90, verticalalignment='center', horizontalalignment='center')
fig.text(0.05,0.95-9/10*0.9,'ankle dorsiflexion\n(degree)',fontsize=40,rotation=90, verticalalignment='center', horizontalalignment='center')

fig.text(1-0.85/2,0.025,'% gait cycle',fontsize=50, verticalalignment='center', horizontalalignment='center')
# plt.show()
plt.savefig(os.path.join(os.getcwd(),'grf_kinematics.pdf'))