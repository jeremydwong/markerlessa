#%%this is a template data processing file.
# you'll fill out the things in this first block, and then you will run the rest. 
# so the inputs are...
# and the outputs are...

# here are the subject-specific and analyzer-specific things.
# datafolder_string_zara
# datafolder_string_jer
# datafolder_string_jess
# datafolder_string_surabhi

datafolder_string = r"/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Undergrad 2024 Shared Folder/"
date_string       = "2024-11-22"
subj_string       = "pilot"
sr_datacollection = 28 #this defines interpolation. 
fname_coords      = 'coordmatrix4'
fnames_trials     = ['recording_6']
model_type        = 'SIMPLE_HOLISTIC'
str_bodyprtcal    = 'right_wrist'

import os
cached_folder_string = os.path.join(datafolder_string,"cached")
#%%
# python code: 
import sys, os
sys.path.append(os.path.join(os.getcwd()))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import scipy.io
from scipy.signal import find_peaks
lr = importlib.import_module('libreach')
importlib.reload(lr)
#%%
#### steps:
###1. get the rotation matrix from the heel-on-floor trial (either pre-loaded or from clicks). 
  # save matrix R in processed clicks as processed_clicks/fname_rot.mat. we could use this to rescale.
###2. get the reach data from the trials 2-7, scoring starts/ends of the triple movement. 
  # save as processed_clicks/f'{fname[:-4]}_processed_clicks.csv'. 
###3. plot results.
###4. get main sequence data.
  # save middle movement scoring as processed_clicks/f'{fname[:-4]}_mainsequence.mat'. 
###5. plot main sequence data.
### after this file has been used to run through every subject, 
# each of the trials will have a 'f{filename}_mainsequence.mat' file that we can use
# to merge the subjects together. 

# we use two functions to get the full path of the A) data and B) frame time.
def fullfile_expt(trialname):
  return os.path.join(datafolder_string,date_string,"recordings",trialname,model_type, "xyz_"+model_type+"_labelled.csv")

def fullfile_frametime(trialname):
  return os.path.join(datafolder_string,date_string,"recordings",trialname,model_type, "frame_time_history.csv")

#%% step 1. Rotation matrix calculation. 

pddata      = pd.read_csv(fullfile_expt(fname_coords)) 
cal_bdyprt  = np.array([pddata[str_bodyprtcal+"_x"],pddata[str_bodyprtcal+"_y"],pddata[str_bodyprtcal+"_z"]]).T

# check to see if we have computed a rotation matrix.
fname_full = os.path.join(datafolder_string,'cached',subj_string + 'rot.mat')

R = np.eye(3)
x0 = np.zeros(3)
if os.path.exists(fname_full):
  dat = scipy.io.loadmat(fname_full)
  R   = dat["R"]
  x0  = dat["x0"]
else:
  R,x0    = lr.get_rotmat_x0(cal_bdyprt)
  # save R
  scipy.io.savemat(fname_full,{'R':R, 'x0':x0})

### print the rotation matrix
print(R)

# Here is how you can plot an Nx3 dataset given the rotation matrix and x0
bodyprt_rot = R @ cal_bdyprt.T # rotate the wrist data. this is now 3xN
bodyprt_rot = bodyprt_rot.T - x0 # zero the data.

# let's see the wrist data in the rotated frame.
import matplotlib
matplotlib.use('qtagg')#tqagg
f,ax = plt.subplots()
plt.plot(bodyprt_rot[:,0],bodyprt_rot[:,1],label='wrist')
# plot the first time instance in green
plt.plot(bodyprt_rot[0,0],bodyprt_rot[1,0],'go',label='start')
# plot the last time instance in red
plt.plot(bodyprt_rot[-1,0],bodyprt_rot[-1,1],'ro',label='end')
# show the legend in the bottom right
plt.legend(loc='lower right')
plt.show(block=True) #blocking pauses the running of the code until you close the plot

# %% Step 2 and 3: Click the start and end of each triple movement, and plot.

for i in range(len(fnames_trials)):
  pathname_cur  = fullfile_expt(fnames_trials[i])
  pddata_cur    = pd.read_csv(pathname_cur)
  
  
  time_cur = pddata_cur['sync_index'].to_numpy()*1/sr_datacollection
  
  # creates an object that organizes the data nicely; so that for 
  # example reach_trial.wri is nX3.
  reach_trial = lr.ReachDataClass(pddata_cur,
                             time_cur,
                             pathname_cur,
                             sub_name = subj_string, 
                             sr_fixed=sr_datacollection)

  # assign Rotation matrix R and x0 to the current data.
  reach_trial.R   = R
  reach_trial.x0  = x0

  # extract starts and ends with a series of clicks.
  reach_trial.click_add_wrist_starts_ends(cached_folder=cached_folder_string)

  # plot the results: 
  # x-axis: time
  # y axis: peak speed;
  # green and red markers: start and end of each movement sequence Out-Move-Back.
  # reach_trial now has 'mov_starts' and 'mov_ends' defined. let's plot to see them.
  f,ax = plt.subplots()
  ax.plot(reach_trial.time,reach_trial.tanvel_wri)
  ax.plot(reach_trial.time[reach_trial.mov_starts], reach_trial.tanvel_wri[reach_trial.mov_starts], 'go')
  ax.plot(reach_trial.time[reach_trial.mov_ends], reach_trial.tanvel_wri[reach_trial.mov_ends], 'rs')
  ax.set_xlabel('Time')
  ax.set_ylabel('speed wri (mm/s)')
  # ax.set_ylim([0,1500])
  ax.legend(['tanvel_wri', 'Movement Starts', 'Movement Ends'])
  ax.set_title(reach_trial.fraw_name)
  plt.show(block=True)
  # pause for input

#%% step 4: mainsequence analysis.
# note: mainsequence comes from 'mainsequence' Bahill, ..., Stark 1975.
# score the middle movements using peaks_and_valleys, or manual if it's the wrong number.
for i in range(len(fnames_trials)):
  pathname_cur = fullfile_expt(fnames_trials[i])
  print(f'Main Sequence (speed, duration): iteration {i}; filename {pathname_cur}.')
  pddata_cur = pd.read_csv(pathname_cur)
  reach_trial = lr.ReachDataClass(pddata_cur,
                             time_cur,
                             pathname_cur,
                             sub_name = subj_string, 
                             sr_fixed=sr_datacollection)

  # get just the file name from the path
  fname_cur = os.path.basename(pathname_cur)

  # assign Roation matrix R to the current data.
  reach_trial.R = R
  # read in the saved starts/ends.
  reach_trial.click_add_wrist_starts_ends(cached_folder=cached_folder_string)

  distancelist, durationlist, peakspeedlist, indlist_middle_mvmnt_start_end = reach_trial.mainsequence(cached_folder = cached_folder_string)

  # each element in this list is a tuple! time, then wrist.
  tpllist_time_wrist = reach_trial.cut_middle_movements(indlist_middle_mvmnt_start_end)
  #%%
  fig,ax = plt.subplots()
  ax.plot(reach_trial.wri_f - reach_trial.sho_f)
  ax.set_xlabel('sample')
  ax.set_ylabel('position (m)')
  plt.show(block = True)
  
  #%% Step 5: plot mainsequence results.
  # plot in m/s the distance, duration, and speed. 
  #% plot the dist peakspeed
  fig,ax = plt.subplots(2,1)
  #set plot size
  fig.set_size_inches(2,2)
  ax[0].plot(distancelist,durationlist,'o')
  ax[0].set_xlabel('Distance (m)')
  ax[0].set_ylabel('Duration (s)')
  #set xlimit
  # ax[0].set_xlim([0,.5])
  # ax[0].set_ylim([0,1.0])
  ax[1].plot(distancelist,peakspeedlist,'o')
  ax[1].set_xlabel('Distance (m)')
  ax[1].set_ylabel('Peak Speed (m/s)')
  # ax[1].set_xlim([0,0.5])
  # ax[1].set_ylim([0,1.5])
  #%
  #%% Plot 
  # 1. tangential velocity 
  # 2. hand position in 3D
  # 3. rotated hand position in 3D
  fig = plt.figure()
  fig.set_size_inches(2,2)

  tgts = list()
  ax_3d    = fig.add_subplot(221,projection='3d')
  ax_3dr  = fig.add_subplot(222,projection='3d')
  ax_tv   = fig.add_subplot(223) 

  for i in range(len(tpllist_time_wrist)):
    ind01      = indlist_middle_mvmnt_start_end[i] # the indices of the middle movement. can use to get sho/finger any other column.
    inds      = range(ind01[0],ind01[1])
    t         = tpllist_time_wrist[i][0]
    movwri    = tpllist_time_wrist[i][1]
    
    ax_3d.plot(movwri[:,0], movwri[:,1], movwri[:,2])    
    tgt_start = movwri[0,:]
    tgt_end = movwri[-1,:]
    ax_3d.plot(tgt_start[0], tgt_start[1], tgt_start[2], 'ro')
    ax_3d.plot(tgt_end[0], tgt_end[1], tgt_end[2], 'go')
    ax_3dr.set_aspect('equal')

    # zero the movements to the first shoulder position of the first movement.
    sho0 = reach_trial.sho_f[reach_trial.mov_starts[0:1],:]
    wri_f_fromsho = movwri - sho0
    # now rotate and subtract x0
    wri_r = wri_f_fromsho @ R.T - x0
    ax_3dr.plot(wri_r[:,0], wri_r[:,1], wri_r[:,2])
    tgt_start = wri_r[0,:]
    tgt_end = wri_r[-1,:]
    ax_3dr.plot(tgt_start[0], tgt_start[1], tgt_start[2], 'ro')
    ax_3dr.plot(tgt_end[0], tgt_end[1], tgt_end[2], 'go')
    ax_3dr.set_xlabel('x (r+)')
    ax_3dr.set_ylabel('y (f+)')
    ax_3dr.set_zlabel('z (u+)')
    # set axis aspect to be equal in each dimension
    ax_3dr.set_aspect('equal')

    # ax_3dr.set_xlim([-200,800])
    # ax_3dr.set_ylim([-200,800])
    # ax_3dr.set_zlim([-500,500])

    t = reach_trial.time[inds]  
    t = t-t[0]
    
    ax_tv.plot(t,reach_trial.tanvel_wri[inds])
    # end loop through movements  

  plt.xlabel('Time')
  plt.ylabel('wri_f')
  plt.show(block=True)

# %%
# try out cross correlating a signal with itself.
# use the velocity traces snipped out here. 
f,ax = plt.subplots(2,1)

# the snipped segment, the template.
indmov = 4
one_snip = tpllist_time_wrist[indmov]
# one_snip = np.reshape(one_snip[1][:,0], (-1, 1))
one_snip = one_snip[1][:,0:2]
# compute derivative of one_snip using np.gradient
vel_snip = lr.vel(np.arange(0,one_snip.shape[0])*1/sr_datacollection,one_snip)
tv_snip  = np.sqrt(np.sum(vel_snip**2,axis=1))

# the whole trace, the signal.
ind01 = np.arange(reach_trial.mov_starts[indmov],reach_trial.mov_ends[indmov])
#reach tanvel_wri
tv_signal = reach_trial.tanvel_wri[ind01]

cs = lr.normxcorr2(tv_snip,tv_signal)
#
cs2 = cs[tv_snip.shape[0]:-1]
ax[0].plot(tv_snip)
ax[0].plot(tv_signal)
ax[1].plot(cs2)
# get max of cs2
indmax = np.argmax(cs2)
ax[1].plot(indmax,cs2[indmax],'ro')

plt.show(block=True)
# %%
