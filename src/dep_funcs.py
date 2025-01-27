# moving functions etc that are no longer important here. 
def process_caliscope_time(data:np.array):
  """
  Process the time data from the caliscope data.
  Inputs:
  data: pandas dataframe
    the time data from the caliscope file 'frame_time_history.csv' which has
      sync_index
      port
      frame_index
      frame_time

  Returns:
  time: np.array
    the time data in seconds
  """

  # there are multiple rows with the same sync_index. 
  # for now, find the unique values of sync_index:
  sync_index_u = data['sync_index'].unique()
  # loop through the unique values to grab the rows having that sync_index
  time_s        = np.zeros(len(sync_index_u))
  time_s_range  = np.zeros((len(sync_index_u),2))
  time_s_sd     = np.zeros(len(sync_index_u))
  
  for i, sync_index in enumerate(sync_index_u):
    # get all of the rows with that sync_index
    rows = data[data['sync_index'] == sync_index]
    # take the average, sd and range of the frame_time
    time_s[i] = np.mean(rows['frame_time'])
    time_s_sd[i] = np.std(rows['frame_time'])
    time_s_range[i] = [np.min(rows['frame_time'])-time_s[i],np.max(rows['frame_time']-time_s[i])]
  time_s = time_s - time_s[0]
  return time_s, time_s_sd, time_s_range

#%%
# this is a class for the reach data. 
# i find it really difficult to keep typing fmc['right_shoulder_x'] and so on.
# we also typically want to work with the data in numpy arrays, not pandas dataframes.
# and have the xyz data in a single array, not three separate arrays.
class ReachDataClass:
  fraw_name = ''
  sub_name = ''
  path  = ''
  time = []
  sho_r = []
  elb_r = []
  wri_r = []
  sho_f = []
  elb_f = []
  wri_f = []
  tanvel_sho = []
  tanvel_elb = []
  tanvel_wri = []
  vel_sho = []
  vel_elb = []
  vel_wri = []
  vel_sho_r = []
  vel_elb_r = []
  vel_wri_r = []
  R         = []

  mov_starts = []
  mov_ends = []

# constructor for reach data, receiving a pandas dataframe
  def __init__(self, dfr:pd.DataFrame, time_s, path, sub_name = '', sr_fixed = 30.0, cutoff_freq = 5.0):
    self.path = path
    self.fraw_name = os.path.basename(path)
    self.time_s = time_s
    self.sub_name = sub_name

    # get the x and y data for the shoulder, elbow, and wrist
    # make temp variables for sho elb wri
    shotemp = np.array([dfr['right_shoulder_x'], dfr['right_shoulder_y'], dfr['right_shoulder_z']]).T
    elbtemp = np.array([dfr['right_elbow_x'], dfr['right_elbow_y'], dfr['right_elbow_z']]).T
    writemp = np.array([dfr['right_wrist_x'], dfr['right_wrist_y'], dfr['right_wrist_z']]).T
    findxtemp = np.array([dfr['right_index_finger_dip_x'], dfr['right_index_finger_dip_y'], dfr['right_index_finger_dip_z']]).T
    fthmbtemp = np.array([dfr['right_thumb_tip_x'], dfr['right_thumb_tip_y'], dfr['right_thumb_tip_z']]).T

    # resample the data                   
    self.time , self.sho_r = resample_data(time_s, shotemp, sr_fixed)
    _         , self.elb_r = resample_data(time_s, elbtemp, sr_fixed)
    _         , self.wri_r = resample_data(time_s, writemp, sr_fixed)
    _         , self.findx_r = resample_data(time_s, findxtemp, sr_fixed)
    _         , self.fthmb_r = resample_data(time_s, fthmbtemp, sr_fixed)
    
    self.wri_f = lowpass_cols(self.wri_r, fs = sr_fixed, cutoff_freq = cutoff_freq)
    self.sho_f = lowpass_cols(self.sho_r, fs = sr_fixed, cutoff_freq = cutoff_freq)
    self.elb_f = lowpass_cols(self.elb_r, fs = sr_fixed, cutoff_freq = cutoff_freq)
    self.findx_f = lowpass_cols(self.findx_r, fs = sr_fixed, cutoff_freq = cutoff_freq)
    self.fthmb_f = lowpass_cols(self.fthmb_r, fs = sr_fixed, cutoff_freq = cutoff_freq)

    # get the speed data
    vel = vel(self.time, self.wri_f)
    self.vel_elb = vel(self.time, self.elb_f)
    self.vel_sho = vel(self.time, self.sho_f)
    self.vel_findx = vel(self.time, self.findx_f)
    self.vel_fthmb = vel(self.time, self.fthmb_f)

    # raw velocity, which probably we won't use given fmc noise
    vel_r = vel(self.time, self.wri_r)
    self.vel_elb_r = vel(self.time, self.elb_r)
    self.vel_sho_r = vel(self.time, self.sho_r)


    # add tanvel wrist
    self.tanvel_wri = np.sqrt(vel[:,0]**2 + vel[:,0]**2 + vel[:,0]**2)
    self.tanvel_elb = np.sqrt(self.vel_elb[:,1]**2 + self.vel_elb[:,1]**2 + self.vel_elb[:,1]**2)
    self.tanvel_sho = np.sqrt(self.vel_sho[:,2]**2 + self.vel_sho[:,2]**2 + self.vel_sho[:,2]**2)
  
  def mainsequence(self, cached_folder):
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
    fsave = os.path.join(cached_folder,self.sub_name+"_mainsequence.mat")
    if os.path.exists(fsave):
      dat = scipy.io.loadmat(fsave)
      i_whichmax = np.argmax(dat['peakspeeds'])
      tv_ff = lowpass(self.tanvel_wri,fs=30,cutoff_freq= 2)
      ind_peaks = dat['threepeaks']
      ind_valleys = dat['valleys']
      f,ax = plt.subplots()
      for i in range(len(self.mov_starts)):
        i0 = self.mov_starts[i]
        i1 = self.mov_ends[i]
        t0 = self.time[i0]
        tvthreshmms = 200
        tv_greaterthan = np.where(self.tanvel_wri[i0:i1]>200)
        if tv_greaterthan[0].shape[0] > 0:
          tshift = self.time[tv_greaterthan[0][0]]
        else:
          tshift = 0

        alph = .3
        if i == i_whichmax:
          # alpha solid
          alph = 1

        plt.plot(self.time[i0:i1]-t0-tshift,    self.tanvel_wri[i0:i1],alpha = alph)
        # plt.plot(self.time[i0:i1]-t0,       tv_ff[i0:i1], '--', label='lowpass')
        ip = ind_peaks[i]
        iv = ind_valleys[i]
        plt.plot(self.time[ip]-t0-tshift,   self.tanvel_wri[ip], "kx",alpha = alph)
        plt.plot(self.time[iv]-t0-tshift, self.tanvel_wri[iv], "ko",alpha = alph)

      plt.show()
      ffig = os.path.join(cached_folder,'figures',f'{self.fraw_name[:-4]}_peaksvalleys.pdf')
      f.savefig(ffig)
      return dat['distances'], dat['durations'], dat['peakspeeds'], dat['valleys']
    
    else:
      print(f"File {fsave} not found. Computing main sequence data.")
      for i in range(len(self.mov_starts)):
        tv    = self.tanvel_wri[self.mov_starts[i]:self.mov_ends[i]]
        wrist = self.wri_f[self.mov_starts[i]:self.mov_ends[i],:]
        time = self.time[self.mov_starts[i]:self.mov_ends[i]]

        ind_peaks, ind_valleys = peaks_and_valleys(tv)
        print(ind_peaks, ind_valleys)

        mid_reach_wrist = wrist[ind_valleys[0]:ind_valleys[1],:]
        dist_wrist = np.sqrt(np.sum((mid_reach_wrist[:,0]-mid_reach_wrist[:,-1])**2))
        distances.append(dist_wrist)
        peakspeeds.append(max(tv[ind_valleys[0]:ind_valleys[1]]))
        durations.append(time[ind_valleys[1]] - time[ind_valleys[0]])
        #append to valleys the ind_valleys, relative not to the start of the reach but the whole file.
        # (then can use ind valleys and ind_peaks to cut when we want.)
        valleys.append(ind_valleys + self.mov_starts[i])
        threepeaks.append(ind_peaks + self.mov_starts[i])
        # save the data to a file
      scipy.io.savemat(fsave,{'distances':distances,'durations':durations,'peakspeeds':peakspeeds,'valleys':valleys,'threepeaks':threepeaks})
      return np.array(distances), np.array(durations), np.array(peakspeeds),valleys

  def cut_middle_movements(self,inds_middle_start_end):
    # make a list of the reaches defined by ind_moves, which is really a list pair of indices
    cutreaches = list()
    for imov in range(len(inds_middle_start_end)):
      inds = np.arange(inds_middle_start_end[imov][0],inds_middle_start_end[imov][1])
      tzeroed = self.time[inds] - self.time[inds[0]]
      cutreaches.append((tzeroed,np.array(self.wri_f[inds,:])))
    return cutreaches

  def click_add_wrist_starts_ends(self, cached_folder, numclicks=-1, do_skip_figs = False):
    """
      Click to add wrist starts and ends, stores in the object, and saves to a csv file.
      Inputs:
      numclicks: int, default -1
        the number of clicks to make. if -1, then click until you close the figure.
      do_skip_figs: bool, default False
        if True, skip the figure and use the saved clicks.
      Returns: (optionally returns)
      mov_starts: np.array
        the indices of the start of the movements
    
    """

    cached_file = os.path.join(cached_folder, f'{self.sub_name}_savedclicks.csv')
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

    if len(indices)==0:
      fig, ax = plt.subplots(4, 1)
      ax[0].plot(self.time, vel[:,0])
      ax[0].set_ylabel('v (mm/s)')
      ax[1].plot(self.time, vel[:,1])
      ax[1].set_ylabel('v (mm/s)')
      ax[2].plot(self.time, vel[:,2])
      ax[2].set_ylabel('v (mm/s)')
      # ax[3].plot(self.time, tv)
      ax[3].set_xlabel('time (s)')
      ax[3].set_ylabel('v (mm/s)')

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
