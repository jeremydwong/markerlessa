#%% WIP: Homogenous coordinates and transformations to calibrate camera data. 

#%% Step 1. what are homogenous coordinates and why do we care?
# Homogenous coordinates are a way to represent points in space, 
# so that they can be transformed by matrix multiplication.  
# This is useful because it allows us to represent translations
# as well as rotations, in a sequence of matrix multiplications.
# 
# Let's look at an example translation of a point in space, with a
# homogenous transformation matrix that shifts the point by a 
# vector t. 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
w = 1
x = np.array([2,1,w])
t = np.array([2,3]) #translation by vector t
# homogenous transformation matrix
T = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])

# apply transformation
x_t = T @ x
print(x_t)

# plot the results also
f,ax = plt.subplots()
plt.plot(x[0],x[1],'ro')
plt.plot(x_t[0],x_t[1],'bo')
plt.axis('equal')
# set x limit
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.grid()
plt.show()


# %%
