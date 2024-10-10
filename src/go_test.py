#%%
# import pandas as pd 
import os
import libreach as lr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('qtagg')
# name of the data dir
datadir = "/Users/jeremy/OneDrive - University of Calgary/Zara Thesis Project/Python Refresher Assignments/"

# name of the R file
fname = "recording_11_28_21_gmt-7_by_trajectory.csv" #jer
fname_r_1009 = "/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Zara Thesis Project/Data/2024-10-09/recordings/coords/HOLISTIC_OPENSIM/xyz_HOLISTIC_OPENSIM_labelled.csv"
df    = pd.read_csv(fname_r_1009)
data = df[['left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z']].to_numpy()
R = lr.get_rotmat(data)

# %%
# get the name of the folder containing this file
# dirname = os.path.dirname(__file__)
# fname_data = os.path.join(dirname, 'example_jacks.csv')

#%% in this cell i just want to create a plotting function we can make rapid changes to. 
import matplotlib.pyplot as plt
pddata = pd.read_csv(fname_data)

def draw_body_parts(pddata,R,indrange):
  f,ax = plt.subplots()
  # set axis as 3d
  ax = f.add_subplot(111, projection='3d')
  ''''
  def draw_body_parts
  wip: meant to handle arbirary body parts to be drawn as segment.
  '''
  sh_l = R @ np.array([pddata['left_shoulder_x'][indrange[0:1]],pddata['left_shoulder_y'][indrange[0:1]],pddata['left_shoulder_z'][indrange[0:1]]])
  sh_r = R @ np.array([pddata['right_shoulder_x'][indrange[0:1]],pddata['right_shoulder_y'][indrange[0:1]],pddata['right_shoulder_z'][indrange[0:1]]])
  hi_r = R @ np.array([pddata['right_hip_x'][indrange[0:1]],pddata['right_hip_y'][indrange[0:1]],pddata['right_hip_z'][indrange[0:1]]])
  hi_l = R @ np.array([pddata['left_hip_x'][indrange[0:1]],pddata['left_hip_y'][indrange[0:1]],pddata['left_hip_z'][indrange[0:1]]])

  sh_l = sh_l
  sh_r = sh_r
  hi_r = hi_r
  hi_l = hi_l

  sh_lz = sh_l# - sho_fix
  sh_rz = sh_r# - sho_fix
  hi_rz = hi_r# - sho_fix
  hi_lz = hi_l# - sho_fix

  ax.plot(np.concatenate((sh_lz[0,0:1],sh_rz[0,0:1],hi_rz[0,0:1],hi_lz[0,0:1],sh_lz[0,0:1])),
          np.concatenate((sh_lz[1,0:1],sh_rz[1,0:1],hi_rz[1,0:1],hi_lz[1,0:1],sh_lz[1,0:1])),
          np.concatenate((sh_lz[2,0:1],sh_rz[2,0:1],hi_rz[2,0:1],hi_lz[2,0:1],sh_lz[2,0:1])),c='k',linewidth=3)


  # as above, but draw the lines from right shoulder to right_elbow, right_wrist, right_hand
  el_r = R @ np.array([pddata['right_elbow_x'][indrange[0:1]],pddata['right_elbow_y'][indrange[0:1]],pddata['right_elbow_z'][indrange[0:1]]])
  wr_r = R @ np.array([pddata['right_wrist_x'][indrange[0:1]],pddata['right_wrist_y'][indrange[0:1]],pddata['right_wrist_z'][indrange[0:1]]])
  # ha_r = R @ np.array([pddata['right_hand_x'][indrange[0:1]],pddata['right_hand_y'][indrange[0:1]],pddata['right_hand_z'][indrange[0:1]]])

  el_r = el_r
  wr_r = wr_r
  # ha_r = ha_r

  el_rz = el_r# - sho_fix 
  wr_rz = wr_r# - sho_fix
  # ha_rz = ha_r# - sho_fix

  ax.plot(np.concatenate((sh_rz[0,0:1],el_rz[0,0:1],wr_rz[0,0:1])),
          np.concatenate((sh_rz[1,0:1],el_rz[1,0:1],wr_rz[1,0:1])),
          np.concatenate((sh_rz[2,0:1],el_rz[2,0:1],wr_rz[2,0:1])),c='r',linewidth=3)  

  # as above, plotting the left arm
  el_l = R @ np.array([pddata['left_elbow_x'][indrange[0:1]],pddata['left_elbow_y'][indrange[0:1]],pddata['left_elbow_z'][indrange[0:1]]])
  wr_l = R @ np.array([pddata['left_wrist_x'][indrange[0:1]],pddata['left_wrist_y'][indrange[0:1]],pddata['left_wrist_z'][indrange[0:1]]])
  # ha_l = R @ np.array([pddata['left_hand_x'][indrange[0:1]],pddata['left_hand_y'][indrange[0:1]],pddata['left_hand_z'][indrange[0:1]]])

  el_l = el_l
  wr_l = wr_l
  # ha_l = ha_l

  el_lz = el_l# - sho_fix
  wr_lz = wr_l# - sho_fix
  # ha_lz = ha_l# - sho_fix

  ax.plot(np.concatenate((sh_lz[0,0:1],el_lz[0,0:1],wr_lz[0,0:1])),
          np.concatenate((sh_lz[1,0:1],el_lz[1,0:1],wr_lz[1,0:1])),
          np.concatenate((sh_lz[2,0:1],el_lz[2,0:1],wr_lz[2,0:1])),c='b',linewidth=3)
  
  pm = .5
  # set 3d axis limits to be sh_lz +/- pm
  ax.set_xlim3d([sh_lz[0,0]-pm,sh_rz[0,0]+pm])
  ax.set_ylim3d([sh_lz[1,0]-pm,sh_rz[1,0]+pm])
  ax.set_zlim3d([sh_lz[2,0]-pm,sh_rz[2,0]+pm])
  
  # label x axis
  ax.set_xlabel('X (m)')
  # label y axis
  ax.set_ylabel('Y (m)')
  # label z axis
  ax.set_zlabel('Z (m)')

  # set the default view to be aligned y axis
  ax.view_init(elev=80,azim=-90)
  plt.show(block=True)

  # return the right arm
  return [sh_rz,el_rz,wr_rz]

rightarm = draw_body_parts(pddata,R,[100])

# %%
# el_r = R @ np.array([pddata['right_elbow_x'],pddata['right_elbow_y'],pddata['right_elbow_z']])
# plt.plot(el_r[0,:],el_r[1,:])
# plt.show()