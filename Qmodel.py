import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import os

class Qmodel(object):
    
    def __init__(self, QArr=np.array([50.0, 100.0, 500.0]), fmin=0.01, fmax=0.1, NumbRM=3):
        self.QArr=QArr
        self.fmin=fmin
        self.fmax=fmax
        self.NumbRM=NumbRM
        return
    
    def Qcontinuous(self, tau_min=1.0e-3, tau_max=1.0e2, plotflag=True ):
        #------------------
        #- initializations 
        #------------------
        #- make logarithmic frequency axis
        f=np.logspace(np.log10(self.fmin),np.log10(self.fmax),100)
        w=2*np.pi*f
        #- compute tau from target Q
        Q=self.QArr[0]
        tau=2/(np.pi*Q)
        #----------------------------------------------------
        #- computations for continuous absorption-band model 
        #----------------------------------------------------
        A=1+0.5*tau*np.log((1+w**2*tau_max**2)/(1+w**2*tau_min**2))
        B=tau*(np.arctan(w*tau_max)-np.arctan(w*tau_min))
        self.Q_continuous=A/B
        self.v_continuous=np.sqrt(2*(A**2+B**2)/(A+np.sqrt(A**2+B**2)))
        if plotflag==True:
            plt.subplot(121)
            plt.semilogx(f,1./self.Q_continuous,'k')
            plt.xlabel('frequency [Hz]')
            plt.ylabel('1/Q')
            plt.title('absorption (1/Q)')
            
            plt.subplot(122)
            plt.semilogx(f,self.v_continuous,'k')
            plt.xlabel('frequency [Hz]')
            plt.ylabel('v')
            plt.title('phase velocity')
            
            plt.show()
        return
    
    def Qdiscrete(self, max_it=30000, T_0=0.2, d=0.9998, f_ref=1.0/20.0, alpha=0.0):
        """
        Computation and visualisation of a discrete absorption-band model.
        For a given array of target Q values, the code determines the optimal relaxation
        times and weights using simulated annealing algorithmn. This is done within in specified
        frequency range.
        Input:
        max_it - number of iterations
        T_0    - the initial random step length
        d      - the temperature decrease (from one sample to the next by a factor of d)
        f_ref  - Reference frequency in Hz
        alpha  - exponent (alpha) for frequency-dependent Q, set to 0 for frequency-independent Q
        """
        #------------------
        #- initialisations 
        #------------------
        #- make logarithmic frequency axis
        f=np.logspace(np.log10(self.fmin),np.log10(self.fmax),100)
        w=2.0*np.pi*f
        #- compute tau from target Q at reference frequency
        tau=1.0/self.QArr
        #- compute target Q as a function of frequency
        Q_target=np.zeros([len(self.QArr), len(f)])
        for n in np.arange(len(self.QArr)):
            Q_target[n,:]=self.QArr[n]*(f/f_ref)**alpha
        #- compute initial relaxation times: logarithmically distributed
        tau_min=1.0/self.fmax
        tau_max=1.0/self.fmin
        tau_s=np.logspace(np.log10(tau_min),np.log10(tau_max),self.NumbRM)/(2*np.pi)
        #- make initial weights
        D=np.ones(self.NumbRM)
        #*********************************************************************
        # STAGE I
        # Compute relaxation times for constant Q values and weights all equal
        #*********************************************************************
        #- compute initial Q -------------------------------------------------
        chi=0.0
        for n in np.arange(len(self.QArr)):
            A=1.0
            B=0.0
            for p in np.arange(self.NumbRM):
                A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
                B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)
            Q=A/B
            chi+=sum((Q-self.QArr[n])**2/self.QArr[n]**2)
        #--------------------------------
        #- search for optimal parameters 
        #--------------------------------
        #- random search for optimal parameters ------------------------------
        D_test=np.array(np.arange(self.NumbRM),dtype=float)
        tau_s_test=np.array(np.arange(self.NumbRM),dtype=float)
        T=T_0
        for it in np.arange(max_it):
            #- compute perturbed parameters ----------------------------------
            tau_s_test=tau_s*(1.0+(0.5-rd.rand(self.NumbRM))*T)
            D_test=D*(1.0+(0.5-rd.rand(1))*T) 
            #- compute test Q ------------------------------------------------
            chi_test=0.0
            for n in np.arange(len(self.QArr)):
                A=1.0
                B=0.0
                for p in np.arange(self.NumbRM):
                    A+=tau[n]*(D_test[p]*w**2*tau_s_test[p]**2)/(1.0+w**2*tau_s_test[p]**2)
                    B+=tau[n]*(D_test[p]*w*tau_s_test[p])/(1.0+w**2*tau_s_test[p]**2)
                Q_test=A/B
                chi_test+=sum((Q_test-self.QArr[n])**2/self.QArr[n]**2)
            #- compute new temperature ----------------------------------------
            T=T*d
            #- check if the tested parameters are better ----------------------
            if chi_test<chi:
                D[:]=D_test[:]          # equivalent to D=D_test.copy()
                tau_s[:]=tau_s_test[:]
                chi=chi_test
        #**********************************************************************
        # STAGE II
        # Compute weights for frequency-dependent Q with relaxation times fixed
        #**********************************************************************
        #- compute initial Q --------------------------------------------------
        chi=0.0
        for n in np.arange(len(self.QArr)):
            A=1.0
            B=0.0
            for p in np.arange(self.NumbRM):
                A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
                B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)
            Q=A/B
            chi+=sum((Q-Q_target[n,:])**2/self.QArr[n]**2)
        #- random search for optimal parameters -------------------------------
        T=T_0
        for it in np.arange(max_it):
            #- compute perturbed parameters -----------------------------------
            D_test=D*(1.0+(0.5-rd.rand(self.NumbRM))*T)
            #- compute test Q -------------------------------------------------
            chi_test=0.0
            for n in np.arange(len(self.QArr)):
                A=1.0
                B=0.0
                for p in np.arange(self.NumbRM):
                    A+=tau[n]*(D_test[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
                    B+=tau[n]*(D_test[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)
                Q_test=A/B
                chi_test+=sum((Q_test-Q_target[n,:])**2/self.QArr[n]**2)
            #- compute new temperature ------------------
            T=T*d
            #- check if the tested parameters are better 
            if chi_test<chi:
                D[:]=D_test[:]
                chi=chi_test
        # print 'Cumulative rms error:  ', np.sqrt(chi/(len(Q)*len(self.QArr)))
        #************************************************
        # STAGE III
        # Compute partial derivatives dD[:]/dalpha
        #************************************************
        #- compute perturbed target Q as a function of frequency
        Q_target_pert=np.zeros([len(self.QArr),len(f)])
        for n in range(len(self.QArr)):
            Q_target_pert[n,:]=self.QArr[n]*(f/f_ref)**(alpha+0.1)
        #- make initial weights
        D_pert=np.ones(self.NumbRM)
        D_pert[:]=D[:]
        #- compute initial Q ------------------------------------------------------------------------------
        chi=0.0
        for n in np.arange(len(self.QArr)):
            A=1.0
            B=0.0
            for p in np.arange(self.NumbRM):
                A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
                B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)
            Q=A/B
            chi+=sum((Q-Q_target_pert[n,:])**2/self.QArr[n]**2)
        #- random search for optimal parameters -----------------------------------------------------------
        T=T_0
        for it in np.arange(max_it):
            #- compute perturbed parameters ---------------------------------------------------------------
            D_test_pert=D_pert*(1.0+(0.5-rd.rand(self.NumbRM))*T)
            #- compute test Q -----------------------------------------------------------------------------
            chi_test=0.0
            for n in np.arange(len(self.QArr)):
                A=1.0
                B=0.0
                for p in np.arange(self.NumbRM):
                    A+=tau[n]*(D_test_pert[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
                    B+=tau[n]*(D_test_pert[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)
                Q_test=A/B
                chi_test+=sum((Q_test-Q_target_pert[n,:])**2/self.QArr[n]**2)
            #- compute new temperature --------------------------------------------------------------------
            T=T*d
            #- check if the tested parameters are better --------------------------------------------------
            if chi_test<chi:
                D_pert[:]=D_test_pert[:]
                chi=chi_test
        #********************
        # Output 
        #********************
        #------------------------------------
        #- sort weights and relaxation times 
        #------------------------------------
        decorated=[(tau_s[i], D[i]) for i in range(self.NumbRM)]
        decorated.sort()
        tau_s=[decorated[i][0] for i in range(self.NumbRM)]
        D=[decorated[i][1] for i in range(self.NumbRM)]
        #-------------------------------------
        #- print weights and relaxation times 
        #-------------------------------------
        print 'Weights: \t\t', D
        print 'Relaxation times: \t', tau_s
        print 'Partial derivatives: \t', (D_pert - D)/0.1
        # print 'Cumulative rms error:  ', np.sqrt(chi/(len(Q)*len(self.QArr)))
        self.D=D
        self.tau_s=tau_s
        self.D_pert=D_pert
        self.chi=chi
        self.Q_target=Q_target
        return
    
    def PlotQdiscrete( self, D=None, tau_s=None, f_ref=1.0/20.0, alpha=0.0 ):
        if D==None or tau_s==None or D.size!=self.NumbRM:
            D=self.D
            tau_s=self.tau_s
        chiTotal=0.0
        #- make logarithmic frequency axis
        f=np.logspace(np.log10(self.fmin),np.log10(self.fmax),100)
        w=2.0*np.pi*f
        #- compute tau from target Q at reference frequency
        tau=1.0/self.QArr
        #- compute target Q as a function of frequency
        Q_target=np.zeros([len(self.QArr), len(f)])
        for n in np.arange(len(self.QArr)):
            Q_target[n,:]=self.QArr[n]*(f/f_ref)**alpha
        #- minimum and maximum frequencies for plotting in Hz
        f_min_plot=0.5*self.fmin
        f_max_plot=2.0*self.fmax
        f_plot=np.logspace(np.log10(f_min_plot),np.log10(f_max_plot),100)
        w_plot=2.0*np.pi*f_plot
        #-----------------------------------------------------
        #- plot Q and phase velocity as function of frequency 
        #-----------------------------------------------------
        for n in np.arange(len(self.QArr)):
            #- compute optimal Q model for misfit calculations
            A=1.0
            B=0.0
            for p in np.arange(self.NumbRM):
                A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2)
                B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2)
            Q=A/B
            chi=np.sqrt(sum((Q-Q_target[n])**2/Q_target[n]**2)/len(Q))
            chiTotal+=(chi**2)
            print 'Individual rms error for Q_0='+str(self.QArr[n])+':  '+str(chi)
            #- compute optimal Q model for plotting
            A=1.0
            B=0.0
            for p in np.arange(self.NumbRM):
                A+=tau[n]*(D[p]*w_plot**2*tau_s[p]**2)/(1.0+w_plot**2*tau_s[p]**2)
                B+=tau[n]*(D[p]*w_plot*tau_s[p])/(1.0+w_plot**2*tau_s[p]**2)
            Q_plot=A/B
            v_plot=np.sqrt(2*(A**2+B**2)/(A+np.sqrt(A**2+B**2)))
    
            # plt.subplot(121)
            plt.subplot(111)
            plt.semilogx([self.fmin,self.fmin],[0.9*self.QArr[n],1.1*self.QArr[n]],'r')
            plt.semilogx([self.fmax,self.fmax],[0.9*self.QArr[n],1.1*self.QArr[n]],'r')
            plt.semilogx(f,Q_target[n,:],'r',linewidth=3)
            plt.semilogx(f_plot,Q_plot,'k',linewidth=3)
            plt.xlim([f_min_plot,f_max_plot])
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Q')
            plt.title('quality factor Q')
        
            # plt.subplot(122)
            # plt.semilogx([self.fmin,self.fmin],[0.9,1.1],'r')
            # plt.semilogx([self.fmax,self.fmax],[0.9,1.1],'r')
            # plt.semilogx(f_plot,v_plot,'k',linewidth=2)
            # plt.xlim([f_min_plot,f_max_plot])
            # plt.xlabel('frequency [Hz]')
            # plt.ylabel('v')
            # plt.title('phase velocity')
            plt.show()
        #------------------------------
        #- stress relaxation functions 
        #------------------------------
        dt=min(tau_s)/10.0
        t=np.arange(0.0,max(tau_s),dt)
        for i in range(len(self.QArr)):
            c=np.ones(len(t))
            for n in range(self.NumbRM):
                c+=tau[i]*D[n]*np.exp(-t/tau_s[n])
            plt.plot(t,c)
            plt.text(5.0*dt,np.max(c),r'$Q_0=$'+str(self.QArr[i]))
        plt.xlabel('time [s]')
        plt.ylabel('C(t)')
        plt.title('stress relaxation functions')  
        plt.show()
        print 'Cumulative rms error:  ', np.sqrt(chiTotal/(len(self.QArr)))
        return
    
    def write(self, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        relax_file = (
        "RELAXATION TIMES [s] =====================\n"
        "{relax_times}\n"
        "WEIGHTS OF RELAXATION MECHANISMS =========\n"
        "{relax_weights}").format(
        relax_times="\n".join(["%.6f" % _i for _i in self.tau_s]),
        relax_weights="\n".join([ "%.6f" % _i for _i in self.D]))
        with open(outdir+'/relax', 'wb') as f:
            f.writelines(relax_file)
        return
        