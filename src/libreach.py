import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Functions for students to use to simplify their data analysis
# get_list_x_clicks_on_plot(data)


def get_rotmat_x0(data_nx3):
  
  # plot the data, with a title
  f,ax = plt.subplots()
  plt.ion()
  plt.plot(data_nx3)
  titletxt = f"Click 3 times to back-left, front-left,front-right:"
  plt.title(titletxt)
  
  # initialize the list of x-clicks we will collect
  ind_xclicks = []
  # Function to capture mouse clicks
  def onclick(event):
    ind_xclicks.append((event.xdata))
    # break if coordinates are 3
    if len(ind_xclicks) == 3:
      plt.gcf().canvas.mpl_disconnect(cid)
      plt.close()

  # Connect the click event to the function
  cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
  plt.draw()
  plt.show(block=True)

  # round the indices of the clicks
  ind_xclicks = [int(np.round(ind_xclicks[i])) for i in range(3)]
  
  # define vectors in 3D
  pt1 = data_nx3[ind_xclicks[0],:]
  pt2 = data_nx3[ind_xclicks[1],:]
  pt3 = data_nx3[ind_xclicks[2],:]
  
  # define which order. assume standard:
  v1 = pt3 - pt2 # positive right
  v2 = pt2 - pt1 # positive forward
  # normalize v1 and v2 to be unit length
  v1 = v1/np.linalg.norm(v1)
  v2 = v2/np.linalg.norm(v2)
  # remove any component of v2 that is in the direction of v1
  v2 = v2 - np.dot(v2,v1)*v1
  v2 = v2/np.linalg.norm(v2)
  # now take the cross product
  v3 = np.cross(v1,v2)

  # compute the rotation matrix to go from the data frame to that defined by v1,v2,v3
  R = np.array([v1,v2,v3])

  # test by first rotating data_nx3, then subtracting out pt1, then plotting
  data_nx3_rot = R @ (data_nx3.T)
  x0           = data_nx3_rot[:,ind_xclicks[0]]
  data_nx3_rot = data_nx3_rot.T - x0
  
  f,ax = plt.subplots()
  # make it a 3D figure
  ax = f.add_subplot(111, projection='3d')
  ax.plot(data_nx3_rot[:,0],data_nx3_rot[:,1],data_nx3_rot[:,2],'k.')
  
  # plot the rotated points with circles
  pt2_R = R @ (pt2 - pt1)
  pt3_R = R @ (pt3 - pt1)
  ax.plot(pt2_R[0],pt2_R[1],pt2_R[2],'ro')
  ax.plot(pt3_R[0],pt3_R[1],pt3_R[2],'go')
  
  ax.set_xlabel('X (m)')
  ax.set_ylabel('Y (m)')
  ax.set_zlabel('Z (m)')

  #set limits t be -1 to 1
  ax.set_xlim3d([-.1,1.1])
  ax.set_ylim3d([-.1,1.1])
  ax.set_zlim3d([-.1,1.1])

  #set el az to be 70, 90
  ax.view_init(elev=70,azim=-90)

  plt.show(block = True)
  return R,x0

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
          np.concatenate((sh_rz[2,0:1],el_rz[2,0:1],wr_rz[2,0:1],ha_rz[2,0:1])),c='k',linewidth=3)  

  pm = 1.0
  # set 3d axis limits to be sh_lz +/- pm
  ax.set_xlim3d([sh_lz[0,0]-pm,sh_lz[0,0]+pm])

  plt.show(block=True)