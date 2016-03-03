import numpy as np
import matplotlib.pyplot as plt

#==================================================================================================
#- input
#==================================================================================================

Q_0=100.0

tau_s=[1.65159913, 13.66501919, 37.07774555]
D=[2.59203931, 2.48647256, 0.07372733]


#- minimum and maximum frequencies for optimisation in Hz
f_min=1.0/100.0
f_max=1.0/20.0

#- number of relaxation mechanisms
N=3
#######################################################################
#==================================================================================================
#- computation
#==================================================================================================

#- make logarithmic frequency axis
f=np.logspace(np.log10(f_min),np.log10(f_max),100)
w=2.0*np.pi*f

#- compute tau from target Q
tau=2.0/(np.pi*Q_0)

#- compute effective numerical Q and phase velocity

A=1.0
B=0.0
for p in np.arange(N):
	A+=tau*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
	B+=tau*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)

Q=A/B
v=np.sqrt(2*(A**2+B**2)/(A+np.sqrt(A**2+B**2)))

#==================================================================================================
#- plotting
#==================================================================================================

plt.subplot(121)
plt.semilogx([f_min,f_min],[0.9*Q_0,1.1*Q_0],'r')
plt.semilogx([f_max,f_max],[0.9*Q_0,1.1*Q_0],'r')
plt.semilogx(f,Q,'k',linewidth=2)
plt.xlabel('frequency [Hz]')
plt.ylabel('Q')
plt.title('quality factor Q')

plt.subplot(122)
plt.semilogx([f_min,f_min],[0.9,1.1],'r')
plt.semilogx([f_max,f_max],[0.9,1.1],'r')
plt.semilogx(f,v,'k',linewidth=2)
plt.xlabel('frequency [Hz]')
plt.ylabel('v')
plt.title('phase velocity')

plt.show()