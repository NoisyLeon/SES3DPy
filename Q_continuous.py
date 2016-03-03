from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import random as rd

""" computation and visualisation of a continuous absorption-band model
        C_r=1 and rho=1 are assumed
        tau is computed from the target Q via 1/Q=0.5*tau*pi
"""

#--------------------------------------------------------------------------------------------------------------------------------------
#- input ----------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

#- general setting

Q=1000.0                 #- target Q

f_min=0.01              #- minimum frequency [Hz] for plotting and discrete-case optimisation
f_max=0.05           #- maximum frequency in [Hz] for plotting and discrete-case optimisation
#######################################################################
#- continuous absorption-band model

tau_min=1.0e-3              #- minimum relaxation time [s]
tau_max=1.0e2             #- maximum relaxation time [s]

#--------------------------------------------------------------------------------------------------------------------------------------
#- initialisations ----------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

#- make logarithmic frequency axis
f=np.logspace(np.log10(f_min),np.log10(f_max),100)
w=2*np.pi*f

#- compute tau from target Q
tau=2/(np.pi*Q)

#--------------------------------------------------------------------------------------------------------------------------------------
#- computations for continuous absorption-band model --------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

A=1+0.5*tau*np.log((1+w**2*tau_max**2)/(1+w**2*tau_min**2))
B=tau*(np.arctan(w*tau_max)-np.arctan(w*tau_min))

Q_continuous=A/B

v_continuous=np.sqrt(2*(A**2+B**2)/(A+np.sqrt(A**2+B**2)))

#--------------------------------------------------------------------------------------------------------------------------------------
#- plot Q and phase velocity as functions of frequency ---------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

plt.subplot(121)
plt.semilogx(f,1/Q_continuous,'k')
plt.xlabel('frequency [Hz]')
plt.ylabel('1/Q')
plt.title('absorption (1/Q)')

plt.subplot(122)
plt.semilogx(f,v_continuous,'k')
plt.xlabel('frequency [Hz]')
plt.ylabel('v')
plt.title('phase velocity')

plt.show()

