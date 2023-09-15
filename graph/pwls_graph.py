import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

fig = plt.figure(2,figsize=(19.2,10.03*1.5),dpi=100)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

def pwls(x, y, p, degree):
    t = y #tベクトルを作成
    t = t*p
    X = np.empty((len(x),0))
    for i in range(0, degree+1):
        c = x ** i
        c = c*p
        X = np.append(X, c.reshape((len(c),1)), axis=1)

    #w=(Φ^T*Φ)^(−1)*Φ^T*tを計算
    # ws = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X).astype(np.float64)), X.T), t)
    ws = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), t)
    #関数f(x)を作成
    def f(x):
        y = 0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    return f, ws


row_num = 9
column_num = 21
x = [i*0.05+0.5 for i in range(column_num) for j in range(row_num)]
y = [(row_num-j)*0.1 for i in range(column_num) for j in range(row_num)]
print(x)
print(y)

p1 = [(i+1)/row_num for i in range(row_num)]
p2 = [1-i/row_num for i in range(row_num)]
p3 = [1/9, 3/9, 5/9, 7/9, 1, 7/9, 5/9, 3/9, 1/9]
p4 = [3/9, 5/9, 7/9, 1, 7/9, 5/9, 3/9, 1/9, 1/9]
p5 = [5/9, 7/9, 1, 7/9, 5/9, 3/9, 1/9, 1/9, 3/9]

p1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ,1.0, 0.5]
p2 = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ,1.0, 1.0]
p3 = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0 ,1.0, 1.0]
p4 = [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0 ,1.0, 1.0]
p5 = [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0 ,1.0, 1.0]
p6 = [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0 ,1.0, 1.0]

# p = p1 + p1 + p1 + p2 + p2 + p2 + p3 + p3 + p3
p = []
for i in range (3):
    for j in range(int(column_num/3)):
        if i==0:
            p += p1
        elif i==1:
            p += p2

p += p3
p += p3
p += p4
p += p4
p += p5
p += p5
p += p6
# p += p6
# p += p6
print(len(x),len(y),len(p))

### 1
sc = ax1.scatter(x, y, vmin=0.5, vmax=1.2, c=p, cmap=cm.gray, s=100)
cb = fig.colorbar(sc,ax=ax1)
cb.set_label(label='Cost of Transport',size=40)
cb.ax.tick_params(labelsize=30)
ax1.set_xlabel('$v_{x}$',fontsize=40)
ax1.set_ylabel('$y_{i}$',fontsize=40)
ax1.set_xlim(0.4, 1.6)
ax1.set_ylim(0.0, 1.05)
ax1.tick_params(labelsize = 30)
ax1.grid(linestyle='--')
ax1.set_xticks([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


### 2
sc = ax2.scatter(x, y, vmin=0.5, vmax=1.2, c=p, cmap=cm.gray, s=100)
cb = fig.colorbar(sc,ax=ax2)
cb.set_label(label='Cost of Transport',size=40)
cb.ax.tick_params(labelsize=30)
ax2.set_xlabel('$v_{x}$',fontsize=40)
ax2.set_ylabel('$y_{i}$',fontsize=40)
ax2.set_xlim(0.4, 1.6)
ax2.set_ylim(0.0, 1.05)
ax2.tick_params(labelsize = 30)
ax2.grid(linestyle='--')
ax2.set_xticks([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
x = np.array(x)
y = np.array(y)
p = np.array(p)

coefficient = 1
weight = coefficient ** ((np.mean(p))-p/np.mean(p))
f1, _ = pwls(x, y, weight, degree=5)
print(weight[7])
print(weight[8])
print('')

coefficient = 5
weight = coefficient ** ((np.mean(p))-p/np.mean(p))
f2, _ = pwls(x, y, weight, degree=5)
print(weight[7])
print(weight[8])
print('')

coefficient = 10
weight = coefficient ** ((np.mean(p))-p/np.mean(p))
f3, _ = pwls(x, y, weight, degree=5)
print(weight[7])
print(weight[8])
print('')

coefficient = 100
weight = coefficient ** ((np.mean(p))-p/np.mean(p))
f4, _ = pwls(x, y, weight, degree=5)
print(weight[7])
print(weight[8])
print('')

coefficient = 1000
weight = coefficient ** ((np.mean(p))-p/np.mean(p))
f5, _ = pwls(x, y, weight, degree=5)
print(weight[7])
print(weight[8])
print('')

xlines = np.linspace(0.5,1.5,200)
# ax1.plot(xlines, f1(xlines),linewidth=5.0,color='orangered',alpha=0.5,label='$A=1$')
ax2.plot(xlines, f1(xlines),linewidth=5.0,label='$A=1$')
ax2.plot(xlines, f2(xlines),linewidth=5.0,label='$A=5$')
ax2.plot(xlines, f3(xlines),linewidth=5.0,label='$A=10$')
ax2.plot(xlines, f4(xlines),linewidth=5.0,label='$A=100$')
ax2.plot(xlines, f5(xlines),linewidth=5.0,label='$A=1000$')
ax2.legend(loc = 'upper center',fontsize=21,framealpha=1,edgecolor='black',fancybox=False,ncol=5)
# ax1.text(0.5,1.05,f'A = {coefficient}',fontsize=50,transform=ax1.transAxes, verticalalignment='center', horizontalalignment='center')
# ax1.text(0.85,1.05,'(normal least square method)',fontsize=30,transform=ax1.transAxes, verticalalignment='center', horizontalalignment='center')
# ax1.text(0.5,1.05,'normal least square method',fontsize=50,transform=ax1.transAxes, verticalalignment='center', horizontalalignment='center')
# ax2.text(0.5,1.05,'PWLS',fontsize=50,transform=ax1.transAxes, verticalalignment='center', horizontalalignment='center')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'pwls_graph.pdf'))