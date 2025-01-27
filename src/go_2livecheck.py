#%%
# import pandas as pd 
import os
import libreach as lr
import pandas as pd
import numpy as np
import matplotlib
from colour import Color
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('qtagg')
# this file is in src/, we want the repo name which is parent
path_repo = os.path.dirname(os.path.dirname(__file__))
import tkinter as tk
from tkinter import filedialog

file_selected = ""

def select_file(txt):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_selected = filedialog.askopenfile(title=txt)  # Open the dialog to select a folder
    return file_selected

# initialize path_coords to empty
path_coords = ""
# loop while path_coords is empty
while not path_coords:
  path_coords = select_file("Choose which file to use for rotation matrix")
  if path_coords:
    print(f"You selected: {path_coords}")
  else:
    print("No file selected")

# initialize file check on the data.
path_check = ""
# loop while path_coords is empty
while not path_check:
  path_check = select_file("Choose data file to check")
  if path_check:
    print(f"You selected: {path_check}")
  else:
    print("No file selected")

#%%
df        = pd.read_csv(path_coords)
bodypart  = "right_index_finger_tip" 
data = df[[bodypart+"_x", bodypart+"_y", bodypart+"_z"]].to_numpy()
R,X0 = lr.get_rotmat_x0(data,pm=.5)

#%% in this cell i just want to create a plotting function we can make rapid changes to. 
import matplotlib.pyplot as plt
pddata = pd.read_csv(path_check)

def draw_body_parts(pddata,R,x0,indrange = None,pm = None,PLOT_FACE = True, PLOT_LEGS = False):

  if indrange is None:
    indrange = range(0,len(pddata),30)

  ## setup figure
  f,ax = plt.subplots()
  # set axis as 3d
  ax = f.add_subplot(111, projection='3d')

  for ir in range(0,len(indrange)):
    
    # Colors!
    blues = lr.generate_color_range(220/360, 0.65, .25, 0.75, len(indrange))
    reds = lr.generate_color_range(360/360, 0.65, .25, 0.75, len(indrange))
    purples = lr.generate_color_range(260/360, 0.65, .25, 0.75, len(indrange))
    # set the plotting color cr to be a darker red, depending on ir
    cr = reds[ir]
    cb = blues[ir]
    cp  = purples[ir]

    # Torso!
    sh_l = R @ np.array([pddata['left_shoulder_x'][indrange[ir]],pddata['left_shoulder_y'][indrange[ir]],pddata['left_shoulder_z'][indrange[ir]]])
    sh_r = R @ np.array([pddata['right_shoulder_x'][indrange[ir]],pddata['right_shoulder_y'][indrange[ir]],pddata['right_shoulder_z'][indrange[ir]]])
    hi_r = R @ np.array([pddata['right_hip_x'][indrange[ir]],pddata['right_hip_y'][indrange[ir]],pddata['right_hip_z'][indrange[ir]]])
    hi_l = R @ np.array([pddata['left_hip_x'][indrange[ir]],pddata['left_hip_y'][indrange[ir]],pddata['left_hip_z'][indrange[ir]]])

    sh_lz = sh_l - x0# - sho_fix
    sh_rz = sh_r - x0# - sho_fix
    hi_rz = hi_r - x0# - sho_fix
    hi_lz = hi_l - x0# - sho_fix

    ax.plot([sh_lz[0],sh_rz[0],hi_rz[0],hi_lz[0],sh_lz[0]],
            [sh_lz[1],sh_rz[1],hi_rz[1],hi_lz[1],sh_lz[1]],
            [sh_lz[2],sh_rz[2],hi_rz[2],hi_lz[2],sh_lz[2]],c=cp,linewidth=3)

    #table!
    corners = np.array([
        [-1, -0.2, 0],    # 
        [1, -.2, 0],  # 
        [1, 1, 0],  # 
        [-1, 1, 0]   # 
    ])    
    # Create 3D polygon collection
    poly = Poly3DCollection([corners], alpha=0.01)
    poly.set_facecolor('gray')
    ax.add_collection3d(poly)
    
    # as above, right arm
    el_r = R @ np.array([pddata['right_elbow_x'][indrange[ir]],pddata['right_elbow_y'][indrange[ir]],pddata['right_elbow_z'][indrange[ir]]])
    wr_r = R @ np.array([pddata['right_wrist_x'][indrange[ir]],pddata['right_wrist_y'][indrange[ir]],pddata['right_wrist_z'][indrange[ir]]])
    # ha_r = R @ np.array([pddata['right_hand_x'][indrange[ir]],pddata['right_hand_y'][indrange[ir]],pddata['right_hand_z'][indrange[ir]]])

    el_rz = el_r - x0
    wr_rz = wr_r - x0
    
    ax.plot([sh_rz[0],el_rz[0],wr_rz[0]],
            [sh_rz[1],el_rz[1],wr_rz[1]],
            [sh_rz[2],el_rz[2],wr_rz[2]],c=cr,linewidth=3)  

    # as above, left arm
    el_l = R @ np.array([pddata['left_elbow_x'][indrange[ir]],pddata['left_elbow_y'][indrange[ir]],pddata['left_elbow_z'][indrange[ir]]])
    wr_l = R @ np.array([pddata['left_wrist_x'][indrange[ir]],pddata['left_wrist_y'][indrange[ir]],pddata['left_wrist_z'][indrange[ir]]])
    # ha_l = R @ np.array([pddata['left_hand_x'][indrange[ir]],pddata['left_hand_y'][indrange[ir]],pddata['left_hand_z'][indrange[ir]]])

    el_lz = el_l - x0
    wr_lz = wr_l - x0 
    # ha_l = ha_l

    ax.plot([sh_lz[0],el_lz[0],wr_lz[0]],
            [sh_lz[1],el_lz[1],wr_lz[1]],
            [sh_lz[2],el_lz[2],wr_lz[2]],c=cb,linewidth=3)

    # Switch to plot legs
    if PLOT_LEGS:
      # as above, right leg
      kn_r = R @ np.array([pddata['right_knee_x'][indrange[ir]],pddata['right_knee_y'][indrange[ir]],pddata['right_knee_z'][indrange[ir]]])
      an_r = R @ np.array([pddata['right_ankle_x'][indrange[ir]],pddata['right_ankle_y'][indrange[ir]],pddata['right_ankle_z'][indrange[ir]]])
      ft_r = R @ np.array([pddata['right_foot_index_x'][indrange[ir]],pddata['right_foot_index_y'][indrange[ir]],pddata['right_foot_index_z'][indrange[ir]]])
                          
      kn_rz = kn_r - x0
      an_rz = an_r - x0
      ft_rz = ft_r - x0

      ax.plot([hi_rz[0],kn_rz[0],an_rz[0],ft_rz[0]],
              [hi_rz[1],kn_rz[1],an_rz[1],ft_rz[1]],
              [hi_rz[2],kn_rz[2],an_rz[2],ft_rz[2]],c=cr,linewidth=3)
      
      # as above, left leg
      kn_l = R @ np.array([pddata['left_knee_x'][indrange[ir]],pddata['left_knee_y'][indrange[ir]],pddata['left_knee_z'][indrange[ir]]])
      an_l = R @ np.array([pddata['left_ankle_x'][indrange[ir]],pddata['left_ankle_y'][indrange[ir]],pddata['left_ankle_z'][indrange[ir]]])
      ft_l = R @ np.array([pddata['left_foot_index_x'][indrange[ir]],pddata['left_foot_index_y'][indrange[ir]],pddata['left_foot_index_z'][indrange[ir]]])

      kn_lz = kn_l - x0
      an_lz = an_l - x0
      ft_lz = ft_l - x0

      ax.plot([hi_lz[0],kn_lz[0],an_lz[0],ft_lz[0]],
              [hi_lz[1],kn_lz[1],an_lz[1],ft_lz[1]],
              [hi_lz[2],kn_lz[2],an_lz[2],ft_lz[2]],c=cb,linewidth=3)

  if PLOT_FACE:
    colname = 'left_inner_eye'
    le_r = R @ np.array([pddata[colname+'_x'][indrange[ir]],pddata[colname+'_y'][indrange[ir]],pddata[colname+'_z'][indrange[ir]]])
    colname = 'right_inner_eye'
    re_r = R @ np.array([pddata[colname+'_x'][indrange[ir]],pddata[colname+'_y'][indrange[ir]],pddata[colname+'_z'][indrange[ir]]])
    le_rz = le_r - x0
    re_rz = re_r - x0
    ax.plot([le_rz[0],re_rz[0]],
            [le_rz[1],re_rz[1]],
            [le_rz[2],re_rz[2]],c=Color(hsl=(60/360,1,.5)).rgb,linewidth=3)

  # Plotting! axis limits and labels  
  # set 3d axis limits to be sh_lz +/- pm, if pm is not none. 
  if pm is not None:
    ax.set_xlim3d([sh_lz[0]-pm,sh_rz[0]+pm])
    ax.set_ylim3d([sh_lz[1]-pm,sh_rz[1]+pm])
    ax.set_zlim3d([sh_lz[2]-pm,sh_rz[2]+pm])

  # label x axis
  ax.set_xlabel('X (m)')
  ax.set_ylabel('Y (m)')
  ax.set_zlabel('Z (m)')

  # set the default view to be aligned y axis
  ax.view_init(elev=80,azim=-90)
  
  plt.show(block=True)

  # return the right arm
  return ([sh_rz,el_rz,wr_rz],ax)
#%%
draw_body_parts(pddata,R, X0)
# plot over time
f2,ax2 = plt.subplots()
bodypart_final = "right_index_finger_DIP"
wr = pddata[[bodypart_final+"_x",bodypart_final+"_y",bodypart_final+"_z"]].to_numpy()
wr = (R @ wr.T).T
wr = wr - X0
ax2.plot(pddata["sync_index"],wr[:,0],label="x")
ax2.plot(pddata["sync_index"],wr[:,1],label="y")
ax2.plot(pddata["sync_index"],wr[:,2],label="z")
ax2.set_xlabel("syncindex")
ax2.set_ylabel("Position (m?)")
ax2.legend()
# get the folder containing the folder of pathcheck, i.e. two levels up.
folder_check = os.path.dirname(os.path.dirname(path_check.name))

#convert to string
plt.suptitle(f"{bodypart_final} Data from {str(os.path.basename(folder_check))}")
plt.show(block=True)

# %%
