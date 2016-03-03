import numpy as np
import matplotlib.pyplot as plt
import math
stf_fname='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT/stf';
# stf_fname='mystf';
dt=0.05;
data=np.loadtxt(stf_fname);
nt=data.size;
hf = np.fft.fft(data);
hf=hf/math.sqrt(len(hf))
f = np.fft.fftfreq(len(hf), dt);
t=np.arange(nt)*dt;

ax=plt.subplot(211)
plt.plot(t,data,'b', lw=3)
plt.xlim(0.0,float(nt)*dt/4)
ax.tick_params(axis='x', labelsize=20)
plt.xlabel('time(s)', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.title('Source time function (time domain)', fontsize=20)
# plt.show()
# - Frequency domain.
ax=plt.subplot(212)
plt.semilogx(f,np.abs(hf),'r', lw=3)
# T=1./f
plt.plot(f, np.abs(hf),'r', lw=3)
# plt.plot([fmin,fmin],[0.0, np.max(np.abs(hf))],'r--')
# plt.text(1.1*fmin, 0.5*np.max(np.abs(hf)), 'fmin')
# plt.plot([fmax,fmax],[0.0, np.max(np.abs(hf))],'r--')
# plt.text(1.1*fmax, 0.5*np.max(np.abs(hf)), 'fmax')
plt.xlim(0.005,1)
ax.tick_params(axis='x', labelsize=20)
plt.xlabel('frequency(Hz)', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.title('Source time function (frequency domain)',fontsize=20)
plt.show()
