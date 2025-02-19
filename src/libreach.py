import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
from colour import Color 
import statsmodels.formula.api as smf
import matplotlib
import scipy
matplotlib.use('qtagg')#tqagg
# Functions for students to use to simplify their data analysis
# get_list_x_clicks_on_plot(data)

def plot_welch_spectrum(data, sample_rate=1.0):
    freqs, power = signal.welch(data, sample_rate, nperseg=1024)
    plt.semilogy(freqs, power)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    plt.show(block=True)

def tanvel(vel):
  return np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)

def get_rotmat_x0(data_nx3,pm = None, xdir=1):
  
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
  if xdir == 1:
    v1 = pt3 - pt2 # positive right
  else:
    v1 = pt2 - pt3

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
  data_nx3_rot = R @ data_nx3.T
  data_nx3_rot = data_nx3_rot.T
  x0           = data_nx3_rot[ind_xclicks[0],:]
  data_nx3_rot = data_nx3_rot - x0
  
  f,ax = plt.subplots()
  # make it a 3D figure
  ax = f.add_subplot(111, projection='3d')
  ax.plot(data_nx3_rot[ind_xclicks[0]:-1,0],data_nx3_rot[ind_xclicks[0]:-1,1],data_nx3_rot[ind_xclicks[0]:-1,2],'k.')
  
  # plot the rotated points with circles
  pt2_R = R @ (pt2 - pt1)
  pt3_R = R @ (pt3 - pt1)
  ax.plot(0,0,0,'bo')
  ax.plot(pt2_R[0],pt2_R[1],pt2_R[2],'ro')
  ax.plot(pt3_R[0],pt3_R[1],pt3_R[2],'go')
  
  ax.set_xlabel('X (m)')
  ax.set_ylabel('Y (m)')
  ax.set_zlabel('Z (m)')

  if pm is not None:
    ax.set_xlim3d([-pm,pm])
    ax.set_ylim3d([-pm,pm])
    ax.set_zlim3d([-pm,pm])

  #set el az to be 70, -90, a nice overhead view of the movement
  ax.view_init(elev=70,azim=-90)

  plt.show(block = True)
  return R,x0

def lowpass(data:np.array, order=4, fs=30.0, cutoff_freq=10.0):
#lowpass and return a particular data column


    # Calculate normalized cutoff frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    # Design a Butterworth filter
    b, a =signal.butter(order, normal_cutoff, btype='low')

    # Apply the filter to each column
    data_f = signal.filtfilt(b, a, data)

    return data_f

def lowpass_cols(datacols:np.array, order=4, fs=30.0, cutoff_freq=12.0):
  # Initialize an empty array to store the filtered data
  filtered_data = np.empty_like(datacols)

  # Apply lowpass filter to each row in the input array
  for i, row in enumerate(datacols.T):
    filtered_data[:,i] = lowpass(row, order, fs, cutoff_freq)

  return filtered_data

class ReachBody():
  df = np.zeros((0,3))
  name = ""
  filename = ""
  sr   = 1 # needs to be provided
  R    = np.eye(3)
  X0   = np.zeros(3)

  rawmat  = np.zeros((10,3))  # Raw data
  rsmat   = np.zeros((10,3))  # Resampled data
  time    = np.zeros(10)      # Fixed sample time
  mat     = np.zeros((10,3))  # Rotated and zeroed data
  velmat  = np.zeros((10,3)) # Velocity data

  mov_starts = []
  mov_ends = []

  def __init__(self,df,name,fname,R,X0,sr,filtorder=4,cutoff_freq1=10, cutoff_freq2=5):
    self.data = df
    self.name = name
    self.R    = R
    self.X0   = X0
    self.sr   = sr
    self.filename = fname

    bplist = reachbodyparts()
    rawtime   = df['sync_index'].to_numpy()*1/sr

    self.rawmat  = simple_reachbody(df,bplist)
    # catch both time and resampled data in return
    self.time, self.rsmat = resample_data(rawtime,self.rawmat,sr)
    # make a copy of self.rsmat -> self.mat
    self.mat = self.rsmat.copy()
    self.velmat  = np.zeros_like(self.mat)
    
    ## rotate, and raw slices -> overwritten self.mat
    for (ib,bp) in enumerate(bplist):
      if bp[-1] == 'x':
        tempmat = self.mat[:,ib:ib+3]
        self.mat[:,ib:ib+3] = (R @ tempmat.T).T - X0
        setattr(self,"unf_"+bp[:-2],self.mat[:,ib:ib+3])            #slices prefixed 'unf_'

    ## filtered data -> overwrite 'mat'
    self.mat = lowpass_cols(self.mat,order=filtorder,fs=sr,cutoff_freq=cutoff_freq1)
    for (ib,bp) in enumerate(bplist):
      if bp[-1] == 'x':
        setattr(self,bp[:-2],self.mat[:,ib:ib+3])                   #slices, no prefix.
    
    ## create velocity matrix and velocity handles -> velmat, vel_* slices
    self.velmat = vel(self.time, self.mat)

    ## filter velocity -> overwrite velmat
    self.velmat = lowpass_cols(self.velmat,order=filtorder,fs=sr,cutoff_freq=cutoff_freq2)
    for (ib,bp) in enumerate(bplist):
      if bp[-1] == 'x':
        setattr(self,"vel_"+bp[:-2],self.velmat[:,ib:ib+3])            #slices, prefixed vel_

  def mainsequence(self, cached_folder, bodypart = "right_index_finger_tip",thresh1_m_per_s=.2):
    '''
    computes mainsequence() data [D, V, T: Distance, Velocity, Time (duration)] for the reach data file.

    '''
    distances = list()
    durations = list()
    peakspeeds = list()
    valleys    = list()
    threepeaks = list()
    # note the valleys are the middle movement, the one we sort of want most.

    # check for cached file _mainsequence.mat
    # if it exists, load it and return the data
    # if it doesn't exist, compute the data and save it to a file
    base, ext = os.path.splitext(self.filename)
    fsave = os.path.join(cached_folder,self.name+"_"+base+"_mainsequence.mat")
    
    #generalize for using any bodypart
    tv = tanvel(getattr(self, "vel_" + bodypart))
    pos = getattr(self,bodypart)

    if os.path.exists(fsave):
      dat = scipy.io.loadmat(fsave)
      i_whichmax = np.argmax(dat['peakspeeds'])
      # tv_ff = lowpass(tv,fs=30,cutoff_freq= 2) % we used to heavily filter. 
      ind_peaks = dat['threepeaks']
      ind_valleys = dat['valleys']
      f,ax = plt.subplots()
      for i in range(len(self.mov_starts)):
        i0 = self.mov_starts[i]
        i1 = self.mov_ends[i]
        t0 = self.time[i0]

        tv_greaterthan = np.where(tv[i0:i1]>thresh1_m_per_s)
        if tv_greaterthan[0].shape[0] > 0:
          tshift = self.time[tv_greaterthan[0][0]]
        else:
          tshift = 0

        alph = .3
        if i == i_whichmax:
          # alpha solid
          alph = 1

        plt.plot(self.time[i0:i1]-t0-tshift,    tv[i0:i1],alpha = alph)
        # plt.plot(self.time[i0:i1]-t0,       tv_ff[i0:i1], '--', label='lowpass')
        ip = ind_peaks[i]
        iv = ind_valleys[i]
        plt.plot(self.time[ip]-t0-tshift,   tv[ip], "kx",alpha = alph)
        plt.plot(self.time[iv]-t0-tshift, tv[iv], "ko",alpha = alph)

      plt.show()
      ffig = os.path.join(cached_folder,'figures',f'{self.filename}_peaksvalleys.pdf')
      f.savefig(ffig)
      return dat['distances'], dat['durations'], dat['peakspeeds'], dat['valleys']
    
    else:
      print(f"File {fsave} not found. Computing main sequence data.")
      for i in range(len(self.mov_starts)):
        # suffix _ denotes the data for the current movement
        tv_    = tv[self.mov_starts[i]:self.mov_ends[i]]
        pos_ = pos[self.mov_starts[i]:self.mov_ends[i],:]
        time_ = self.time[self.mov_starts[i]:self.mov_ends[i]]

        ind_peaks, ind_valleys = peaks_and_valleys(tv_)
        print(ind_peaks, ind_valleys)

        middle_reach_ = pos_[ind_valleys[0]:ind_valleys[1],:]
        dist_reach = np.sqrt(np.sum((middle_reach_[-1,:]-middle_reach_[0,:])**2))
        distances.append(dist_reach)
        peakspeeds.append(max(tv_[ind_valleys[0]:ind_valleys[1]]))
        durations.append(time_[ind_valleys[1]] - time_[ind_valleys[0]])
        #append to valleys the ind_valleys: relative not to the start of the reach but the whole file.
        # (then can use ind valleys and ind_peaks to cut when we want.)
        valleys.append(ind_valleys + self.mov_starts[i])
        threepeaks.append(ind_peaks + self.mov_starts[i])
        # save the data to a file
      scipy.io.savemat(fsave,{'distances':distances,'durations':durations,'peakspeeds':peakspeeds,'valleys':valleys,'threepeaks':threepeaks})
      return np.array(distances), np.array(durations), np.array(peakspeeds),valleys

  def cut_middle_movements(self,inds_middle_start_end,bodypart='right_index_finger_tip'):
    # make a list of the reaches defined by ind_moves, which is really a list pair of indices
    cutreaches = list()
    for imov in range(len(inds_middle_start_end)):
      inds = np.arange(inds_middle_start_end[imov][0],inds_middle_start_end[imov][1])
      tzeroed = self.time[inds] - self.time[inds[0]]
      cutreaches.append((tzeroed,np.array(getattr(self,bodypart)[inds,:])))
    return cutreaches

  def click_add_reach_starts_ends(self, cached_folder, bodypart = "right_index_finger_tip", numclicks=-1, do_skip_figs = False,ylim=(0,1.5)):
      """
        Click to add starts and ends of specified bodypart, stores in the object, and saves to a csv file.
        Inputs:
        numclicks: int, default -1
          the number of clicks to make. if -1, then click until you close the figure.
        do_skip_figs: bool, default False
          if True, skip the figure and use the saved clicks.
        Returns: (optionally returns)
        mov_starts: np.array
          the indices of the start of the movements
      
      """

      base, ext = os.path.splitext(self.filename)
      cached_file = os.path.join(cached_folder, f'{self.name+"_"+base+"_savedclicks.csv"}')
      # get the name of the file without the path or suffix
      
      indices = []
          
      if os.path.exists(cached_file):
        # Load the file
        clickpd = pd.read_csv(cached_file)
        indices = clickpd['indices'].tolist()
        if do_skip_figs == True:
          print("Skipping clicks, using saved clicks.")
          self.mov_starts = indices[::2]
          self.mov_ends = indices[1::2]
          return indices[::2], indices[1::2]

      # show the velocity of that bodypart.
      if len(indices)==0:
        # which bodypart is moving?
        vel = getattr(self, f'vel_{bodypart}')
        tanvel = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)
        fig, ax = plt.subplots(4, 1)
        ax[0].plot(self.time, vel[:,0])
        ax[0].set_ylabel('v (m/s)')
        ax[0].set_ylim(ylim)
        ax[1].plot(self.time, vel[:,1])
        ax[1].set_ylabel('v (m/s)')
        ax[1].set_ylim(ylim)
        ax[2].plot(self.time, vel[:,2])
        ax[2].set_ylabel('v (m/s)')
        ax[2].set_ylim(ylim)
        ax[3].plot(self.time, tanvel)
        ax[3].set_xlabel('time (s)')
        ax[3].set_ylabel('v (m/s)')
        ax[3].set_ylim(ylim)

        clicks = []
        def onclick(event):
          clicks.append((event.xdata, event.ydata))

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show(block=True)
        
        # Extract x-values from clicks and round them
        x_values = [click[0] for click in clicks]
        
        # Find the closest time value in reachdat.time for each click;
        # these are the indices in the time array.
        indices = [np.abs(self.time - x).argmin() for x in x_values]

        # Save the indices to a csv file
        if numclicks > 0:
          if len(clicks) == numclicks:
            df = pd.DataFrame(indices, columns=['indices'])
            df.to_csv(cached_file, index=False)
          else:
            print("Not saving indices, because you didn't click enough times.")
        else:
          if len(clicks) > 0:
            df = pd.DataFrame(indices, columns=['indices'])
            # check if fsave already exists. if it does, ask
            if os.path.exists(cached_file):
              print(f"{cached_file} already exists. Do you want to overwrite it?")
              print("Enter 'y' to overwrite, anything else to not overwrite.")
              answer = input()
              if answer == 'y':
                df.to_csv(cached_file, index=False)
            else:
              df.to_csv(cached_file, index=False)
          else:
            print("Not saving indices, because you did not click.")
              
      self.mov_starts = indices[::2]
      self.mov_ends = indices[1::2]
      return indices[::2], indices[1::2]

class MainSeq():
  D = []
  V = []
  T = []

  def __init__(self, D=[], T=[],V=[]):
    self.D = D
    self.V = V
    self.T = T

def reachbodyparts():
  listbody = ['index_finger_tip','thumb_tip','shoulder','elbow','wrist','hip']
  sides = ['right','left']
  ax = ['x','y','z']
  bodyparts = []
  for body in listbody:
    for side in sides:
      for a in ax:
        bodyparts.append(f"{side}_{body}_{a}")
  return bodyparts

def plot_power_spectrum(signal, sample_rate=1.0, xlim = None):
    # Compute the FFT
    fft = np.fft.fft(signal)
    # Get the frequencies
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    # Compute power spectrum (magnitude squared)
    power = np.abs(fft)**2
    if xlim is None:
      xlim = (0, sample_rate/2)

    # Plot only the positive frequencies (first half)
    plt.plot(freqs[:len(freqs)//2], power[:len(freqs)//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.xlim(xlim)
    plt.show()

def simple_reachbody(df,bplist = reachbodyparts()):
  """
  Get the simple reach body parts from the dataframe.
  
  Inputs:
  df: pandas dataframe
    the dataframe with the body parts
  
  Returns:
  bodyparts: list
    the list of body parts
  
  """
  bodyxyz = np.zeros((df.shape[0],len(bplist)))
  for (ib,bp) in enumerate(bplist):
    if bp in df.columns:
      bodyxyz[:,ib] = df[bp].to_numpy()
    else:
      print(f"Warning: {bp} not in dataframe.")
  return bodyxyz

def fit_ct(ms, kv = 0.902,kt = 2.264, normV = .3, normT =1.0, verbose = True):
  """
  Returns a CT that fits peakspeed V and duration T for distance D simultaneously (D, V, and T [i.e. reach_fmc.mainsequence() output). 
  
  Inputs:
  ms -> mainsequence object.
  D: np.array
    the distances of the reaches
  V: np.array
    the peak speeds of the reaches
  T: np.array
    the duration of the reaches

  Parameters:
  kv: float, default 0.902
    coefficient in the speed equation:
    V = kv*(D .^3 * ct)^(1/4)
  kt: float, default 2.264
    coefficient in the time equation:
    T = kt*(D ./  ct)^(1/4)
  normV: float, default 0.3
    normalization factor for V
  normT: float, default 1.0
    normalization factor for T
  
  Returns:
  c_t: float
    the estimate of the cost of time
  pvalue: float
    the p-value of the estimate
  results: statsmodels object
  f_v: the speed function
  f_t: the duration function

  Explained math:
  We think people minimize energy and time.
  Energy for reaching is Force-rate (i.e. of units M*D/T^3)
  so the cost function that people minimize is 
  J(T) = c * M*D/T^3 + c_t*T (for integrated f-rate across time) 
  then
  dJdT = c*M*D/T^4 + c_t = 0 
  rearranging to solve for V = D/T or T, we have two propotionalities:
  
  V = kv* CT^1/4 * D ^(3/4)
  T = kt*(D / CT).^(1/4)

  We would like to solve using ordinary least-squares, i.e A*CT = b.
  To do this linearly, and scaling V and T errors with normV and normT, 
  we rearrange: 
  Aspd = (1/normV)^4 * kv^4*D.^3 
  bspd = (V ./ normV).^4 
  
  Adur = (kt^4 * D) ./normT 
  bdur = 1./(T ./ normT) 

  Then stack [Aspd;Adur] and [bspd;bdur] and solve.
  { check: T^4 = kt^4*D/ct ; T^4*ct = kt^4*D 
  
  Example:
  D = np.array([0.1,0.2, 0.3,0.4, 0.5 ,0.60,0.7])
  T = np.array([0.7,0.75,0.8,0.85,0.90,0.95,1.0])
  V = np.array([0.3,0.50,0.7,0.90,1.1 ,1.30,1.5])
  fit_ct(D,V,T)
  """
  D = ms.D
  V = ms.V
  T = ms.T

  Adur = (T / normT) **4
  bdur = (kt) **4 * D # note jer removed kt/normT here.

  Aspd = (1/normV)**4 * kv**4 * D**3
  bspd = (V / normV)**4

  A = np.concatenate((Adur,Aspd))
  b = np.concatenate((bdur,bspd))

  results = smf.ols('b ~ A -1', data = pd.DataFrame({'A':A,'b':b})).fit()
  c_t     = results.params.A #our estimate of the cost of time. 
  pvalue  = results.pvalues.A

  # V = kv* CT^1/4 * D ^(3/4)
  # T = kt*(D / CT).^(1/4)
  def f_v(D,ct): return kv*c_t**(1/4)*D**(3/4)
  def f_t(D,ct): return kt*(1/c_t)**(1/4) * D**(1/4) 

  if verbose:
    print(results.summary())

  return c_t, pvalue, results, f_v, f_t

def peaks_and_valleys(tv_sub,tv_thresh_m_per_s=0.08, domanual=False):
  """
  Find the peaks and valleys in the speed (i.e. tanvel) data.
  
  Each reach-to-grasp is three reaches, each of which have peak speeds.
  The middle reach is the one we are most interested in because it positions the object.
  The peaks and valleys are used to define the start and end of the middle reach:
  the two valleys define the start and end of the middle reach.
  
  Inputs:
  tv_sub: np.array
    the speed data, in mm/s (as per freemocap, units of mm/s)
  tv_thresh_mms: float, default 80
    the threshold for the speed data in mmps

  Returns:
  ind_peaks: np.array
    the indices of the peaks
  ind_valleys: np.array
    the indices of the valleys

  """
  ind_peaks   = []
  ind_valleys = [] # valleys define the start and end of the middle movement.

  if tv_sub.shape[0] > 15: # then it cannot be filtered at 2 Hz.
    # for i in range(len(self.mov_starts)):
    tv_sub_f = lowpass(tv_sub,fs=30,cutoff_freq= 2)

    # Find peaks above tv_thresh
    ind_peaks, _ = signal.find_peaks(tv_sub_f, height=tv_thresh_m_per_s,distance = 10)
    # Loop between each pair of peaks and find the minima between each
    
    for i in range(len(ind_peaks)-1):
      start_index = ind_peaks[i]
      end_index = ind_peaks[i+1]
      minima_index = np.argmin(tv_sub_f[start_index:end_index]) + start_index
      ind_valleys.append(minima_index)

    ind_valleys = np.array(ind_valleys)
  else:
    tv_sub_f = tv_sub
    print("Warning: not enough data to filter.")
    print("Likely this is a double-click by accident. delete the processedclicks.csv file and try again.")

  # currently this does not run. but we could insert a manual flag that checks for each. 
  if domanual:
    plt.plot(tv_sub)
    # plot with dashed line tv_sub_f
    plt.plot(tv_sub_f, '--', label='lowpass')
    plt.plot(ind_peaks, tv_sub[ind_peaks], "x")
    plt.plot(ind_valleys, tv_sub[ind_valleys], "o")
    plt.show()
    print("Enter 'm' to manually score, anything else to continue.")
    answer = input()
    if answer == 'm':
      domanual = True
  
  # if we do not have the right number of peaks/valleys, switch to manual.
  if (len(ind_peaks) + len(ind_valleys) != 5) or domanual:
    print("Warning: not enough peaks and valleys found.")
    print("switching to manual.")
    coordinates = []
    # while len(coordinates) is not equal! to 5, not smaller or not larger.
    while (len(ind_peaks) + len(ind_valleys) != 5): 
      print("switching to manual. Click 5 peaks/valleys in sequence, then close the figure.")
      f,ax = plt.subplots()
      plt.plot(tv_sub)
      plt.plot(tv_sub_f)
      
      # Function to capture mouse clicks
      def onclick(event):
        coordinates.append((event.xdata))

      # Connect the click event to the function
      cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
      plt.show(block=True)
      ind_peaks = [coordinates[0],coordinates[2],[coordinates[4]]]
      ind_peaks = [int(np.round(ind_peaks[i])) for i in range(3)]
      ind_valleys = [coordinates[1],coordinates[3]]
      ind_valleys = [int(np.round(ind_valleys[i])) for i in range(2)]
      ind_peaks = np.array(ind_peaks)
      ind_valleys = np.array(ind_valleys)
      
      # print that we manually scored correctly
      print("Five (peaks + valleys) manually scored.")
      domanual = False

  plt.plot(tv_sub)
  # plot with dashed line tv_sub_f
  plt.plot(tv_sub_f, '--', label='lowpass')
  plt.plot(ind_peaks, tv_sub[ind_peaks], "x")
  plt.plot(ind_valleys, tv_sub[ind_valleys], "o")
  plt.show(block = True)
  # 
  return ind_peaks, ind_valleys

def get_list_x_clicks_on_plot(data3d):
  f,ax = plt.subplots()
  plt.plot(data3d.T)
  coordinates = []
  # Function to capture mouse clicks
  def onclick(event):
    coordinates.append((event.xdata))

  # Connect the click event to the function
  cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
  plt.show()

  return coordinates

def resample_data(time, data, sr, fill_gaps=True):
  """
  Resample the data to a fixed sampling rate, and fill in any gaps in the data.

  Inputs:
  time: np.array
    the time data
  data: np.array
    the data to be resampled
  sr: float
    the fixed sampling rate
  fill_gaps: bool, default True
    whether to fill in any gaps in the data

  Returns:
  time_resamp: np.array
    the resampled time data
  data_resamp: np.array
    the resampled data

  """
  # new time
  time_resamp = np.arange(time[0], time[-1], 1/sr)

  # Resample the data using linear interpolation
  data_resamp = np.zeros((len(time_resamp),data.shape[1]))
  for i in range(data.shape[1]):
    if len(time) != len(data[:,i]):
      print(f"Mismatched lengths at i={i}: length of time: {len(time)}, length of data[{i}]: {len(data[:,i])}")
    data_resamp[:,i] = np.interp(time_resamp, time, data[:,i])
    if fill_gaps:
      # find where the data is nan, and interpolate
      nans = np.isnan(data_resamp[:,i])
      data_resamp[nans,i] = np.interp(time_resamp[nans], time_resamp[~nans], data_resamp[~nans,i])

  return time_resamp, data_resamp

def vel(time:np.array, data:np.array):
  # def vel(time:np.array, data:np.array):
  # take derivative of N rows of input data nparray i.e. data[0,:],data[1,:],data[2,:] wrt time vector
  # use np.gradient

  # initialize an empty array to store the velocity data
  vel = np.empty_like(data)
  # now fill each of vel rows, enumerating over the data columns, not rows, so we transpose
  for i, col in enumerate(data.T):
    vel[:,i] = np.gradient(col, time)
  return vel

def generate_color_range(hue, lightness, sat_low, sat_high, n_colors):
    """
    Generate a range of colors by varying saturation while keeping hue and lightness constant.
    
    Parameters:
    -----------
    hue : float
        Hue value in range [0, 1]
    lightness : float
        Lightness value in range [0, 1]
    sat_low : float
        Lower bound for saturation in range [0, 1]
    sat_high : float
        Upper bound for saturation in range [0, 1]
    n_colors : int
        Number of colors to generate
    
    Returns:
    --------
    list of tuples
        List of RGB colors as (r, g, b) tuples with values in range [0, 1]
    """
    # Generate linearly spaced saturation values
    saturations = np.linspace(sat_low, sat_high, n_colors)
    
    # Convert HSL to RGB for each saturation value
    rgb_colors = []
    for sat in saturations:
        # Create color using HSL values
        color = Color(hsl=(hue, sat, lightness))
        # Get RGB values as tuple
        rgb_colors.append(color.rgb)
    
    return rgb_colors

########################################################################################
# Author: Ujash Joshi, University of Toronto, 2017                                     #
# Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>         #
# Octave/Matlab normxcorr2 implementation in python 3.5                                #
# Details:                                                                             #
# Normalized cross-correlation. Similiar results upto 3 significant digits.            #
# https://github.com/Sabrewarrior/normxcorr2-python/master/norxcorr2.py                #
# http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html    #
# Copyright (c) 2017-2023 Ujash Joshi

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

########################################################################################

import numpy as np
from scipy.signal import fftconvolve

def normxcorr2(template, image, mode="full", reshape_if_needed=True):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # reshape if 1D input
    if np.ndim(template) == 1:
        template = np.reshape(template, (len(template),1))
    if np.ndim(image) == 1:
        image = np.reshape(image, (len(image),1))
        
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    with np.errstate(divide='ignore',invalid='ignore'): 
        out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out


