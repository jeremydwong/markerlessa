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
fname = "recording_11_28_21_gmt-7_by_trajectory.csv"
df    = pd.read_csv(os.path.join(datadir, fname))
data = df[['left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z']].to_numpy()
R = lr.get_rotmat(data)

# %%
fname = "recording_12_25_05_gmt-7_by_trajectory.csv"

#%%
import matplotlib.pyplot as plt
pddata = pd.read_csv(os.path.join(datadir, fname))

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

  sh_l = sh_l/1000
  sh_r = sh_r/1000
  hi_r = hi_r/1000
  hi_l = hi_l/1000

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
  ha_r = R @ np.array([pddata['right_hand_x'][indrange[0:1]],pddata['right_hand_y'][indrange[0:1]],pddata['right_hand_z'][indrange[0:1]]])

  el_r = el_r/1000
  wr_r = wr_r/1000
  ha_r = ha_r/1000

  el_rz = el_r# - sho_fix 
  wr_rz = wr_r# - sho_fix
  ha_rz = ha_r# - sho_fix

  ax.plot(np.concatenate((sh_rz[0,0:1],el_rz[0,0:1],wr_rz[0,0:1],ha_rz[0,0:1])),
          np.concatenate((sh_rz[1,0:1],el_rz[1,0:1],wr_rz[1,0:1],ha_rz[1,0:1])),
          np.concatenate((sh_rz[2,0:1],el_rz[2,0:1],wr_rz[2,0:1],ha_rz[2,0:1])),c='r',linewidth=3)  

  # as above, plotting the left arm
  el_l = R @ np.array([pddata['left_elbow_x'][indrange[0:1]],pddata['left_elbow_y'][indrange[0:1]],pddata['left_elbow_z'][indrange[0:1]]])
  wr_l = R @ np.array([pddata['left_wrist_x'][indrange[0:1]],pddata['left_wrist_y'][indrange[0:1]],pddata['left_wrist_z'][indrange[0:1]]])
  ha_l = R @ np.array([pddata['left_hand_x'][indrange[0:1]],pddata['left_hand_y'][indrange[0:1]],pddata['left_hand_z'][indrange[0:1]]])

  el_l = el_l/1000
  wr_l = wr_l/1000
  ha_l = ha_l/1000

  el_lz = el_l# - sho_fix
  wr_lz = wr_l# - sho_fix
  ha_lz = ha_l# - sho_fix

  ax.plot(np.concatenate((sh_lz[0,0:1],el_lz[0,0:1],wr_lz[0,0:1],ha_lz[0,0:1])),
          np.concatenate((sh_lz[1,0:1],el_lz[1,0:1],wr_lz[1,0:1],ha_lz[1,0:1])),
          np.concatenate((sh_lz[2,0:1],el_lz[2,0:1],wr_lz[2,0:1],ha_lz[2,0:1])),c='b',linewidth=3)
  
  pm = 1.0
  # set 3d axis limits to be sh_lz +/- pm
  ax.set_xlim3d([sh_lz[0,0]-pm,sh_lz[0,0]+pm])
  ax.set_ylim3d([sh_lz[1,0]-pm,sh_lz[1,0]+pm])
  ax.set_zlim3d([sh_lz[2,0]-pm,sh_lz[2,0]+pm])
  
  # label x axis
  ax.set_xlabel('X (m)')
  # label y axis
  ax.set_ylabel('Y (m)')
  # label z axis
  ax.set_zlabel('Z (m)')

  plt.show(block=True)

# define Rshoulder

draw_body_parts(pddata,R,[100])


# %%

# %%

# %%

# %%

# %%
