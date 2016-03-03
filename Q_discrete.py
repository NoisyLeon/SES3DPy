
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd

""" Computation and visualisation of a discrete absorption-band model.

    For a given array of target Q values, the code determines the optimal relaxation
    times and weights. This is done within in specified frequency range.

    Before running the code, the input parameters may need to be edited.
"""

#--------------------------------------------------------------------------------------------------
#- input ------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#- Array of target Q's at the reference frequency (f_ref, specified below). The code tries to find
#- optimal relaxation parameters for all given Q_0 values simultaneously.
Q_0=np.array([50.0, 100.0, 500.0])

#- Minimum and maximum frequencies for optimisation in Hz. These values determine the width of the
#- absorption band.
f_min=1.0/100.0
f_max=1.0/20.0

#- Number of relaxation mechanisms. The broader the absorption band, the more mechanisms are needed.
N=3
#######################################################################
#- Optimisation parameters (number of iterations, temperature, temperature decrease). The code runs
#- a simplistic Simulated Annealing optimisation to find optimal relaxation parameters. max_it is 
#- the maximum number of samples, T_0 the initial random step length, and d is the temperature decrease
#- in the sense that temperature decreases from one sample to the next by a factor of d.
max_it=30000
T_0=0.2
d=0.9998

#- Reference frequency in Hz (f_ref) and exponent (alpha) for frequency-dependent Q. For frequency-
#- independent Q you must set alpha=0.0.
f_ref=1.0/20.0
alpha=0.0

#--------------------------------------------------------------------------------------------------
#- initialisations --------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#- make logarithmic frequency axis
f=np.logspace(np.log10(f_min),np.log10(f_max),100)
w=2.0*np.pi*f

#- minimum and maximum frequencies for plotting in Hz
f_min_plot=0.5*f_min
f_max_plot=2.0*f_max

f_plot=np.logspace(np.log10(f_min_plot),np.log10(f_max_plot),100)
w_plot=2.0*np.pi*f_plot

#- compute tau from target Q at reference frequency
tau=1.0/Q_0

#- compute target Q as a function of frequency
Q_target=np.zeros([len(Q_0),len(f)])

for n in range(len(Q_0)):
    Q_target[n,:]=Q_0[n]*(f/f_ref)**alpha

#- compute initial relaxation times: logarithmically distributed
tau_min=1.0/f_max
tau_max=1.0/f_min
tau_s=np.logspace(np.log10(tau_min),np.log10(tau_max),N)/(2*np.pi)

#- make initial weights
D=np.ones(N)

#**************************************************************************************************
#**************************************************************************************************
# STAGE I
# Compute relaxation times for constant Q values and weights all equal
#**************************************************************************************************
#**************************************************************************************************

#- compute initial Q ------------------------------------------------------------------------------
chi=0.0

for n in np.arange(len(Q_0)):
    A=1.0
    B=0.0
    for p in np.arange(N):
        A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
        B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)

    Q=A/B
    chi+=sum((Q-Q_0[n])**2/Q_0[n]**2)

#--------------------------------------------------------------------------------------------------
#- search for optimal parameters ------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#- random search for optimal parameters -----------------------------------------------------------
D_test=np.array(np.arange(N),dtype=float)
tau_s_test=np.array(np.arange(N),dtype=float)

T=T_0

for it in np.arange(max_it):

    #- compute perturbed parameters ---------------------------------------------------------------
    tau_s_test=tau_s*(1.0+(0.5-rd.rand(N))*T)
    D_test=D*(1.0+(0.5-rd.rand(1))*T)

    #- compute test Q -----------------------------------------------------------------------------
    chi_test=0.0

    for n in np.arange(len(Q_0)):
        A=1.0
        B=0.0
        for p in np.arange(N):
            A+=tau[n]*(D_test[p]*w**2*tau_s_test[p]**2)/(1.0+w**2*tau_s_test[p]**2)
            B+=tau[n]*(D_test[p]*w*tau_s_test[p])/(1.0+w**2*tau_s_test[p]**2)

        Q_test=A/B
        chi_test+=sum((Q_test-Q_0[n])**2/Q_0[n]**2)

    #- compute new temperature --------------------------------------------------------------------
    T=T*d

    #- check if the tested parameters are better --------------------------------------------------
    if chi_test<chi:
        D[:]=D_test[:]
        tau_s[:]=tau_s_test[:]
        chi=chi_test

#**************************************************************************************************
#**************************************************************************************************
# STAGE II
# Compute weights for frequency-dependent Q with relaxation times fixed
#**************************************************************************************************
#**************************************************************************************************

#- compute initial Q ------------------------------------------------------------------------------
chi=0.0

for n in np.arange(len(Q_0)):
    A=1.0
    B=0.0
    for p in np.arange(N):
        A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
        B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)

    Q=A/B
    chi+=sum((Q-Q_target[n,:])**2/Q_0[n]**2)

#- random search for optimal parameters -----------------------------------------------------------

T=T_0

for it in np.arange(max_it):

    #- compute perturbed parameters ---------------------------------------------------------------
    D_test=D*(1.0+(0.5-rd.rand(N))*T)

    #- compute test Q -----------------------------------------------------------------------------
    chi_test=0.0

    for n in np.arange(len(Q_0)):
        A=1.0
        B=0.0
        for p in np.arange(N):
            A+=tau[n]*(D_test[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
            B+=tau[n]*(D_test[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)

        Q_test=A/B
        chi_test+=sum((Q_test-Q_target[n,:])**2/Q_0[n]**2)

    #- compute new temperature --------------------------------------------------------------------
    T=T*d

    #- check if the tested parameters are better --------------------------------------------------
    if chi_test<chi:
        D[:]=D_test[:]
        chi=chi_test

#**************************************************************************************************
#**************************************************************************************************
# STAGE III
# Compute partial derivatives dD[:]/dalpha
#**************************************************************************************************
#**************************************************************************************************

#- compute perturbed target Q as a function of frequency
Q_target_pert=np.zeros([len(Q_0),len(f)])

for n in range(len(Q_0)):
    Q_target_pert[n,:]=Q_0[n]*(f/f_ref)**(alpha+0.1)

#- make initial weights
D_pert=np.ones(N)
D_pert[:]=D[:]

#- compute initial Q ------------------------------------------------------------------------------
chi=0.0

for n in np.arange(len(Q_0)):
    A=1.0
    B=0.0
    for p in np.arange(N):
        A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
        B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)

    Q=A/B
    chi+=sum((Q-Q_target_pert[n,:])**2/Q_0[n]**2)

#- random search for optimal parameters -----------------------------------------------------------

T=T_0

for it in np.arange(max_it):

    #- compute perturbed parameters ---------------------------------------------------------------
    D_test_pert=D_pert*(1.0+(0.5-rd.rand(N))*T)

    #- compute test Q -----------------------------------------------------------------------------
    chi_test=0.0

    for n in np.arange(len(Q_0)):
        A=1.0
        B=0.0
        for p in np.arange(N):
            A+=tau[n]*(D_test_pert[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
            B+=tau[n]*(D_test_pert[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)

        Q_test=A/B
        chi_test+=sum((Q_test-Q_target_pert[n,:])**2/Q_0[n]**2)

    #- compute new temperature --------------------------------------------------------------------
    T=T*d

    #- check if the tested parameters are better --------------------------------------------------
    if chi_test<chi:
        D_pert[:]=D_test_pert[:]
        chi=chi_test

#**************************************************************************************************
#**************************************************************************************************
# Output and plotting
#**************************************************************************************************
#**************************************************************************************************

#--------------------------------------------------------------------------------------------------
#- sort weights and relaxation times --------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

decorated=[(tau_s[i], D[i]) for i in range(N)]
decorated.sort()

tau_s=[decorated[i][0] for i in range(N)]
D=[decorated[i][1] for i in range(N)]

#--------------------------------------------------------------------------------------------------
#- print weights and relaxation times -------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

print 'weights: \t\t', D
print 'relaxation times: \t', tau_s
print 'partial derivatives: \t', (D_pert - D)/0.1
print 'cumulative rms error:  ', np.sqrt(chi/(len(Q)*len(Q_0)))

#--------------------------------------------------------------------------------------------------
#- plot Q and phase velocity as function of frequency ---------------------------------------------
#--------------------------------------------------------------------------------------------------

for n in np.arange(len(Q_0)):

    #- compute optimal Q model for misfit calculations
    A=1.0
    B=0.0

    for p in np.arange(N):
        A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
        B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)

    Q=A/B
    chi=np.sqrt(sum((Q-Q_target[n])**2/Q_target[n]**2)/len(Q))

    print 'individual rms error for Q_0='+str(Q_0[n])+':  '+str(chi)

    #- compute optimal Q model for plotting
    A=1.0
    B=0.0

    for p in np.arange(N):
        A+=tau[n]*(D[p]*w_plot**2*tau_s[p]**2)/(1.0+w_plot**2*tau_s[p]**2)
        B+=tau[n]*(D[p]*w_plot*tau_s[p])/(1.0+w_plot**2*tau_s[p]**2)

    Q_plot=A/B
    v_plot=np.sqrt(2*(A**2+B**2)/(A+np.sqrt(A**2+B**2)))


    plt.subplot(121)
    plt.semilogx([f_min,f_min],[0.9*Q_0[n],1.1*Q_0[n]],'r')
    plt.semilogx([f_max,f_max],[0.9*Q_0[n],1.1*Q_0[n]],'r')
    plt.semilogx(f,Q_target[n,:],'r',linewidth=3)
    plt.semilogx(f_plot,Q_plot,'k',linewidth=3)
    plt.xlim([f_min_plot,f_max_plot])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Q')
    plt.title('quality factor Q')

    plt.subplot(122)
    plt.semilogx([f_min,f_min],[0.9,1.1],'r')
    plt.semilogx([f_max,f_max],[0.9,1.1],'r')
    plt.semilogx(f_plot,v_plot,'k',linewidth=2)
    plt.xlim([f_min_plot,f_max_plot])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('v')
    plt.title('phase velocity')

    plt.show()


#--------------------------------------------------------------------------------------------------
#- stress relaxation functions --------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

dt=min(tau_s)/10.0
t=np.arange(0.0,max(tau_s),dt)

for i in range(len(Q_0)):

    c=np.ones(len(t))

    for n in range(N):
        c+=tau[i]*D[n]*np.exp(-t/tau_s[n])

    plt.plot(t,c)
    plt.text(5.0*dt,np.max(c),r'$Q_0=$'+str(Q_0[i]))

plt.xlabel('time [s]')
plt.ylabel('C(t)')
plt.title('stress relaxation functions')

plt.show()


