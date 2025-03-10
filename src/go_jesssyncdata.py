import numpy as np
import os 
import marker2markerless as m2m
import importlib # import reload lib
import libreach as lr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('qtagg')

# folder files for each person
datafolder_string_zara  = r"/Users/zarwareenkhan/Library/CloudStorage/OneDrive-Personal/Undergrad 2024 Shared Folder"
datafolder_string_jer   = r"/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Undergrad 2024 Shared Folder/"
datafolder_string_jess  = r"C:\Users\JC\University of Calgary\Jeremy Wong - Undergrad 2024 Shared Folder"

str_bodyprtcal    = 'right_index_finger_tip'
coord_xdir        = 1

# active person doing data analysis. 
datafolder_string = datafolder_string_jer #one of the four above

#################### Rotation stuff ###################
#  specific to this analysis relating the two datasets!

# for this particular analysis, the ps data are here:
dir_ps = "jess_phasespacedata"
dir_caliscope = "2025-03-05"
# specific data information for this analysis
fname_coord_ps = "coordmatrix4.csv" #slight differences here in how we write down, because the csvs are all in one directory, unlike caliscope.
fname_coord_caliscope = "coordmatrix4"
model_name = "SIMPLE_HOLISTIC"

# load the caliscope data and ROTATE CALISCOPE
path_coord_caliscope = os.path.join(datafolder_string, dir_caliscope,"recordings",fname_coord_caliscope,model_name,"xyz_SIMPLE_HOLISTIC_labelled.csv")
df_caliscope = pd.read_csv(path_coord_caliscope)
fingertip_caliscope  = np.array([df_caliscope[str_bodyprtcal+"_x"],df_caliscope[str_bodyprtcal+"_y"],df_caliscope[str_bodyprtcal+"_z"]]).T
R_caliscope,x0_caliscope,g_caliscope = lr.get_rotmat_x0(fingertip_caliscope, xdir=coord_xdir)

# load the phase space data and ROTATE PHASE SPACE
path_coord_ps = os.path.join(datafolder_string, dir_ps, fname_coord_ps)
dfcoord_ps = m2m.parse_ps_csv(path_coord_ps)
m0 = dfcoord_ps["marker0"]
m1 = dfcoord_ps["marker1"]
m2 = dfcoord_ps["marker2"] #this is the fingertip.
m3 = dfcoord_ps["marker3"]
m4 = dfcoord_ps["marker4"]
fingertip_ps = m2 /1000

fig,ax = plt.subplots()
ax.plot(m0[:,2],label="m0")
plt.plot(m1[:,2],label="m1")
plt.plot(m2[:,2],label="m2") #this is index finger tip
plt.plot(m3[:,2],label="m3") 
plt.plot(m4[:,2],label="m4")
plt.legend()
plt.show(block=True)

fingertip_ps = m2 / 1000  # phasespace is in mm.

#ahh! we have to swap columns. defined a strange coordinate system with the stick.
tocols = [1,2,0]
fingertip_ps = m2m.swap_cols(fingertip_ps, [0,1,2], tocols) 

R_ps,x0_ps,g_ps = lr.get_rotmat_x0(fingertip_ps, xdir=coord_xdir)

############################ DONE coordmatrix4 Rotation stuff ###################
#%%
def get_ps_caliscope_aligned_ranges(fname_trial_caliscope,fname_trial_ps,R_cal,X0_cal,R_ps,X0_ps):
  sr_cal = 30
  sr_ps = 240

  str_bodyprtcal = 'right_index_finger_tip'
  path_caliscope = os.path.join(datafolder_string, dir_caliscope, "recordings", fname_trial_caliscope, model_name, "xyz_SIMPLE_HOLISTIC_labelled.csv")
  df_caliscope = pd.read_csv(path_caliscope)
  f_cal = np.array([df_caliscope[str_bodyprtcal+"_x"],df_caliscope[str_bodyprtcal+"_y"],df_caliscope[str_bodyprtcal+"_z"]]).T

  # load the phase space data
  path_ps = os.path.join(datafolder_string, dir_ps, fname_trial_ps)
  df_ps = m2m.parse_ps_csv(path_ps)
  f_ps = df_ps["marker2"] / 1000
  tocols = [1,2,0]
  f_ps = m2m.swap_cols(f_ps, [0,1,2], tocols)
  # rotate and zero f_ps
  f_ps = R_ps @ f_ps.T
  f_ps = f_ps.T - X0_ps


  #upsample the cal_bdyprt data to match the phase space data
  t_ps = np.arange(0,len(m0)/sr_ps,1/sr_ps)
  t_caliscope = np.arange(0,len(f_cal)/sr_cal,1/sr_cal)
  t_caliscope_up = np.arange(0,t_caliscope[-1],1/sr_ps)

  # upsample using linear interpolation from scipy
  from scipy.interpolate import interp1d
  f = interp1d(t_caliscope,f_cal,axis=0)

  # resample, rotate and zero f_cal_up
  f_cal_up = f(t_caliscope_up)
  f_cal_up = R_cal @ f_cal_up.T
  f_cal_up = f_cal_up.T - X0_cal

  # Fill in NaNs in the f_ps with linearly interpolated values
  nans = np.isnan(f_ps)
  not_nans = ~nans
  indices = np.arange(len(f_ps))
  for i in range(f_ps.shape[1]):
    f_ps[nans[:, i], i] = np.interp(indices[nans[:, i]], indices[not_nans[:, i]], f_ps[not_nans[:, i], i])

  # plot the upsampled caliscope data
  fig,ax = plt.subplots()
  ax.plot(f_cal_up[:,2],label="caliscope")
  # ax.plot(f_cal[422:1068,2],label="snip")
  plt.plot(f_ps[:,2],label="m2")
  plt.legend()


  # get two clicks from the mouse which we use to idenfity the template kernel and then cross correlate. 
  print("Please click twice to select the kernel region for caliscope data.")
  clicks = plt.ginput(2)
  clicks = np.array(clicks).astype(int)
  plt.show(block=True)
  kernel_start, kernel_end = clicks[0][0], clicks[1][0]

  # extract the kernel from caliscope data
  kernel = f_cal_up[kernel_start:kernel_end, 2]
  #create lowpass filter parameters for butterworth
  import scipy.signal as signal
  b, a = signal.butter(4, 2/120, 'low', analog=False)
  kernel = signal.filtfilt(b, a, kernel)
  # take derivative of kernel, since we care about speed correlation, not position.
  kernel = np.diff(kernel)

  # cross correlate the kernel with f_ps, ignore last half of trial(a fair assumption)
  correlation2 = lr.normxcorr2(kernel, np.diff(f_ps[:int(round(f_ps.shape[0]/2)), 2]))
  # compute optimal shift, ignoring length of kernel (- shifts don't mean anything for us.)
  correlation2 = correlation2[kernel.shape[0]:]
  optimal_shift = np.argmax(correlation2)


  # plot the shifted data
  fig, ax = plt.subplots()
  ax.plot(f_cal_up[kernel_start:, 2], label="caliscope")
  ax.plot(f_ps[optimal_shift:optimal_shift + len(f_cal_up[kernel_start:, 2]), 2], label="shifted_ps")
  plt.legend()
  plt.show(block=True)
  # return these pieces of f_cal_up and f_ps
  f_cal_up = f_cal_up[kernel_start:,:]
  f_ps = f_ps[optimal_shift:optimal_shift + len(f_cal_up), :]
  # downsample from the rounded sr_ps / sr_cal ratio
  f_ps = f_ps[::sr_ps // sr_cal, :]
  f_cal_up = f_cal_up[::sr_ps // sr_cal, :]
  return f_ps, f_cal_up

#%%
# test this on a trial
psout, calout = get_ps_caliscope_aligned_ranges("recording_2","recording_2.csv",R_caliscope,x0_caliscope,R_ps,x0_ps)
#%%
# filter both
import scipy.signal as signal
b, a = signal.butter(3, 10/15, 'low', analog=False)
psfilt = signal.filtfilt(b, a, psout, axis=0)
calfilt = signal.filtfilt(b, a, calout, axis=0)

#plot each
fig, axs = plt.subplots(6, 1, figsize=(10, 8), sharex=True)

titles = ['X Coordinate', 'Y Coordinate', 'Z Coordinate']
g_known = [.5,.4,1]
g_rescale = g_known/np.array((g_caliscope[0],g_caliscope[1],1)) # ? use this?
for i in range(3):
  g_ps = [g_ps[0],g_ps[1],1]
  g_caliscope = [g_caliscope[0],g_caliscope[1],1]
  sr_caliscope = 30
  # compute the gradient then filter again
  ps = psfilt[:,i]
  cal = calfilt[:,i]
  axs[i].plot(ps, label='Phase Space Data', color='b')
  axs[i].plot(cal, label='Caliscope Data', color='r', linestyle='--')
  axs[i].set_ylabel(titles[i])
  axs[i].legend()
  axs[i].grid(True)

titles = ['X Vel', 'Y Vel', 'Z Vel']
for i in range(3):
  g_ps = [g_ps[0],g_ps[1],1]
  g_caliscope = [g_caliscope[0],g_caliscope[1],1]
  
  sr_caliscope = 30
  # compute the gradient then filter again
  ps = np.gradient(psfilt[:, i], 1/sr_caliscope, axis=0)
  cal = np.gradient(calfilt[:, i],1/sr_caliscope, axis = 0)
  ps = signal.filtfilt(b, a, ps, axis=0)
  cal = signal.filtfilt(b, a, cal, axis=0)
  axs[i+3].plot(ps, label='Phase Space Data', color='b')
  axs[i+3].plot(cal*g_known[i]/g_caliscope[i], label='Caliscope Data', color='r', linestyle='--')
  axs[i+3].set_ylabel(titles[i])
  axs[i+3].legend()
  axs[i+3].grid(True)

axs[-1].set_xlabel('Sample Index')
plt.suptitle('Comparison of Phase Space and Caliscope Data')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show(block=True)
# %%
