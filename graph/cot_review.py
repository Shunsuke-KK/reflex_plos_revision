import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np


# with open('cot_review.pickle', mode='wb') as f:
#     checkpoint = {'vel':dataset['vel'], 'cot':dataset['cot']}
#     pickle.dump(checkpoint, f)

with open('cot_review.pickle', mode='rb') as f:
    load_checkpoint = pickle.load(f)
    vel = load_checkpoint['vel']
    cot = load_checkpoint['cot']

def regression_curve(x, y, degree):
    t = y
    phi = DataFrame()
    for i in range(0, degree+1):
        p = x ** i
        p.name = "x ** %d" % i
        phi = pd.concat([phi, p], axis = 1)

    ws = np.dot(np.dot(np.linalg.inv(np.dot(phi.T,phi).astype(np.float64)), phi.T), t)
    def f(x):
        y = 0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    return (f, ws)

fig = plt.figure(1,figsize=(19.2*0.6,10.03),dpi=100)
ax1 = fig.add_subplot(1,1,1)

ax1.scatter(vel, cot, s=30,color='black')
ax1.set_xlim(0.5,1.75)
ax1.set_ylim(0.4,1.5)
ax1.set_xlabel('$v_{x}$ [m/s]',fontsize=40)
ax1.set_ylabel('CoT',fontsize=40)

ax1.tick_params(labelsize = 30)
ax1.grid(linestyle='--')
xlines = np.arange(0.6,1.7,0.01)
f, ws = regression_curve(vel, cot, degree=2)
ax1.plot(xlines,f(xlines),color='maroon',linewidth=5.0)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(),'cot_review.pdf'))
plt.savefig(os.path.join(os.getcwd(),'cot_review.png'))