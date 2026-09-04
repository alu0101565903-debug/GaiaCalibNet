import numpy as np

def degrade_to_gaia_xp(x, y, xx):

  #Table describing the resolution R of the externally calibrated spectra (ECS) XP DR3 data in Montegriffo et al. 2023
  xt = np.array([350.,370.,390.,410,430.,450.,470.,490.,510.,530.,550.,570.,590.,610.,630.,640.,660.,680.,700.,720.,740.,760.,780.,800.,820.,840.,860.,880.,900.,920.,940.,960.,980.])
  yt = np.array([71.5,67.2,61.4,55.3,49.9,45.0,41.4,38.0,35.0,32.5,30.0,28.3,25.5,25.2,22.0,74.6,77.8,77.8,76.8,74.5,73.2,71.1,68.8,66.9,65.1,63.4,62.1,60.7,58.2,57.2,54.9,53.8,52.0])
  yt = xt/yt # FWHM = lambda/R

  xt = xt*10.
  yt = yt*10.

  gg = np.zeros((len(x),len(x)))

  for i in range(len(x)):
     sigma=np.interp(x[i],xt,yt)/2.0/np.sqrt(-2.0*np.log(0.5))
     kernel = np.exp(-(x-x[i])**2/2./sigma**2)
     gg[i,:] = kernel/np.sum(kernel)

  rr = np.matmul(gg,np.transpose(y))
  d2 = np.transpose(rr)
  ys = np.interp(xx,x,d2)

  return(ys)

