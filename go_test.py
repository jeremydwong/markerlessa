#%%
# import pandas as pd 
import os
import libreach as lr
import pandas as pd
import numpy as np
%matplotlib widget
# name of the data dir
datadir = "/Users/jeremy/OneDrive - University of Calgary/Zara Thesis Project/Python Refresher Assignments/"

# name of the R file
fname = "recording_11_28_21_gmt-7_by_trajectory.csv"
df    = pd.read_csv(os.path.join(datadir, fname))
data = df[['left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z']].to_numpy()
R = lr.get_rotmat(data)
