import numpy as np
import matplotlib.pyplot as plt
import os

lmin  = 0.5
lmax  = 1.6
vmax  = 1.5
fpmax = 1.3
fvmax = 1.2

def bump(L, A, mid, B):
    left = 0.5*(A+mid)
    right = 0.5*(mid+B)

    if (L<=A) or (L>=B):
        y = 0
    elif L<left:
        x = (L-A)/(left-A)
        y = 0.5*x*x
    elif L<mid:
        x = (mid-L)/(mid-left)
        y = 1-0.5*x*x
    elif L<right:
        x = (L-mid)/(right-mid)
        y = 1-0.5*x*x
    else:
        x = (B-L)/(B-right)
        y = 0.5*x*x
    return y

# derived quantities
a = 0.5*(lmin+1)
b = 0.5*(1+lmax)
c = fvmax-1

# length and velocity ranges to plot
# LL = np.linspace(lmin, lmax, 51)
# VV = np.linspace(-vmax, vmax, 51)
LL = np.linspace(lmin-0.2, lmax+0.2, 101)
VV = np.linspace(-vmax-0.2, vmax+0.2, 101)


FP = np.zeros(LL.shape)
for i in range(len(LL)):
    L = LL[i]
    if L<=1:
        FP[i] = 0
    elif L<=b:
        x = (L-1)/(b-1)
        FP[i] = 0.25*fpmax*x*x*x
    else:
        x = (L-b)/(b-1)
        FP[i] = 0.25*fpmax*(1+3*x)
FL = np.zeros(LL.shape)
for i in range(len(LL)):
    L = LL[i]
    FL[i] = bump(L, lmin, 1, lmax) + 0.15*bump(L, lmin, 0.5*(lmin+0.95), 0.95)
FV = np.zeros(VV.shape)
for i in range(len(VV)):
    V = VV[i]/vmax
    if V<=-1:
        FV[i] = 0
    elif V<=0:
        FV[i] = (V+1)*(V+1)
    elif V<=c:
        FV[i] = fvmax - (c-V)*(c-V)/c
    else:
        FV[i] = fvmax
fig = plt.figure(4,figsize=(19.2*1.5,10.03),dpi=100)
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.set_xlim(lmin-0.2,lmax+0.2)
ax1.set_ylim(0-0.001,fvmax+0.1)
ax1.set_xlabel('$\\tilde{l}$ [m]',fontsize=50)
ax1.set_ylabel('$F_{l}$',fontsize=50)
ax1.tick_params(labelsize = 30)
ax1.axvline(x=1.0,color='gray',linewidth=3.0)
ax1.axhline(y=1.0,color='gray',linewidth=3.0)
ax1.set_xticks([0.5,0.7,1.0,1.3,1.6])
ax1.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
ax1.plot(LL, FL, 'k', linewidth=3.0, color='orangered')

ax2.set_xlim(lmin-0.2,lmax+0.2)
ax2.set_ylim(0-0.001,fpmax+0.1)
ax2.set_xlabel('$\\tilde{l}$ [m]',fontsize=50)
ax2.set_ylabel('$F_{p}$',fontsize=50)
ax2.tick_params(labelsize = 30)
ax2.axvline(x=1.0,color='gray',linewidth=3.0)
ax2.axhline(y=1.0,color='gray',linewidth=3.0)
ax2.axvline(x=lmax,color='gray',linestyle='dashdot',linewidth=2.0)
ax2.axhline(y=fpmax,color='gray',linestyle='dashdot',linewidth=2.0)
ax2.set_xticks([0.4,0.7,1.0,1.3,1.6])
ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.3,1.4])
ax2.plot(LL, FP, 'k', linewidth=3.0, color='royalblue')

ax3.set_xlim(-vmax-0.2,vmax+0.2)
ax3.set_ylim(0-0.001,fvmax+0.1)
ax3.set_xlabel('$\\tilde{v}$ [m/s]',fontsize=50)
ax3.set_ylabel('$F_{v}$',fontsize=50)
ax3.tick_params(labelsize = 30)
ax3.axvline(x=0.0,color='gray',linewidth=3.0)
ax3.axhline(y=1.0,color='gray',linewidth=3.0)
ax3.axhline(y=fvmax,color='gray',linestyle='dashdot',linewidth=2.0)
ax3.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
ax3.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2])
ax3.plot(VV, FV, 'k', linewidth=3.0, color='seagreen')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(),'FLV.pdf'))