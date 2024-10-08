import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_list_x_clicks_on_plot(data):
  f,ax = plt.subplots()
  plt.plot(data)
  coordinates = []
  # Function to capture mouse clicks
  def onclick(event):
    coordinates.append((event.xdata))

  # Connect the click event to the function
  cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
  plt.show()

  return coordinates

def get_rotmat(data):
  
  ind_clicks = get_list_x_clicks_on_plot(data)
  ind_clicks = [int(np.round(ind_clicks[i])) for i in range(3)]
  print(ind_clicks)
  pt1 = data[ind_clicks[0],:]
  pt2 = data[ind_clicks[1],:]
  pt3 = data[ind_clicks[2],:]
  
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
  return R