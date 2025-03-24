# %% Cell 1, constants: folder for data, and experiment variables!

# A. Hopefully UG students should only have to change this cell.
datafolder_string_zara  = r"/Users/zarwareenkhan/Library/CloudStorage/OneDrive-Personal/Undergrad 2024 Shared Folder"
datafolder_string_jer   = r"/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Undergrad 2024 Shared Folder/"
datafolder_string_jess  = r"C:\Users\JC\University of Calgary\Jeremy Wong - Undergrad 2024 Shared Folder"

# B. variables per-experiment. 
sr_datacollection = 30
datafolder_string = datafolder_string_jer #one of the four above
num_trials = 16
import jh_subjects as sb #ZK: you would swap this to zk_subjects

# C. Targets
# We need to bin the [distance, duration, speed] data to ,make error bar figs. 
# (since that is what most people expect to look at.)
# To do that, we use 'tgts' to define the center of each.
import numpy as np
edges_l = np.array([0,0.02,0.05,.1,.2,.3])
tgts_s = np.array([0,.15,.30,.45,.60]) + 0.0375
tgts_m = np.array([0,.15,.30,.45,.60]) + 0.05625
tgts_b = np.array([0,.15,.30,.45,.60]) + 0.075
# create dictionary of tgts
tgts = {'l':edges_l, 's':tgts_s, 'm':tgts_m, 'b':tgts_b}

# D. edges(tgts) is a helper function to get the edges of the bins.
def edges(tgts,tgtedgemax=0.15):
  edges_d = np.zeros((tgts.shape[0]+1))
  edges_d[0] = 0.0
  for i in range(1,len(tgts)):
    edges_d[i] = (tgts[i-1]+tgts[i])/2
  edges_d[-1] = tgts[-1]+tgtedgemax
  return edges_d

print("cell 1 complete.")
#%% Cell 2/X: import statements, function definitions, MainSeq definition.
import libreach as lr                 # our own code
import pandas as pd                   # tables
import os                             # for paths
import matplotlib.pyplot as plt       # plotting
import statsmodels.formula.api as smf # we could use this to process the statistical models. 
import scipy as sp                    # for loading saved files.
import importlib                      # for reloading the library             

subdata = sb.get_subject_info_list()
#define target distances hardcoded here. 

cts = np.zeros((len(subdata),num_trials))
ps = np.zeros((len(subdata),num_trials))

"""
def fullfile_for_date_and_recordingnum(datestring, trialnum):
just a helper function to get the full file path for a given date and trial number.
"""
def fullfile_for_date_and_recordingnum(datestring, trialnum):
  model_type = "SIMPLE_HOLISTIC"
  return os.path.join(datafolder_string,datestring,"recordings","recording_"+str(trialnum),model_type, "xyz_"+model_type+"_labelled.csv")

"""
A class that groups Distances D, Speeds V and movement Time/Duration T. 
we will sometimes bin these to make averages, and can stuff this class in a table.
note: "Main Sequence" concept is from 1970s, and is a relationship between
speed, distance and time in a movement in eye movements (Bahill Stark 1975).
"""
class MainSeq():
  D = []
  V = []
  T = []
  def __init__(self,D=[],V=[],T=[]):
    self.D = D
    self.V = V
    self.T = T  
  
  def __str__(self):
    return f"mainSeq:\n  D: {np.array2string(np.array(self.D), precision=2, separator=', ')}\n  V: {np.array2string(np.array(self.V), precision=2, separator=', ')}\n  T: {np.array2string(np.array(self.T), precision=2, separator=', ')}"
  
def get_condition_text(short_cond):
  if short_cond == "l":
    return "line (y-axis)"
  elif short_cond == "s":
    return "small circle high accuracy"
  elif short_cond == "m":
    return "medium circle medium accuracy"
  elif short_cond == "b":
    return "big circle low accuracy"
  else:
    return "error!"

"""
save_figure(fig, figname):
save a figure to the project's figures folder.
"""
def save_figure(fig, figname):
  fig.savefig(os.path.join(datafolder_string, sb.get_figures_folder_for_project(), figname))

print("Cell 2 (function definitions) complete.")
#%% cell 3/X: First loop across subjects, many plots.
# Output: 'table', which is a pd dataframe that holds all the data for every trial. 
# here are the columns of table.
table = pd.DataFrame(columns = ['isub','subj','cond','movetype','p','ct','ms'])
isub = 0
subj = subdata[isub].name
#initialize a pd dataframe to hold all subjects' data

for isub, subj in enumerate(subdata):
  subname   = subdata[isub].name
  filenums  = subdata[isub].fnums
  conds     = subdata[isub].conds
  foldername= subdata[isub].foldername
  f,ax      = plt.subplots(4,2)

  vels = list()
  durs = list()
  dists= list()
  
  for ifl,fpath in enumerate(filenums):
    cond = conds[ifl]
    row_plt = 0
    if cond == 'l':
      row_plt= 0
    elif cond == 's':
      row_plt = 1
    elif cond == 'm':
      row_plt = 2
    elif cond == 'b':
      row_plt = 3
    
    movetype = subdata[isub].movetype[ifl]
    
    pathname_cur = fullfile_for_date_and_recordingnum(foldername,filenums[ifl])
    print(f'Main Sequence (speed, duration): iter {ifl}; filename {pathname_cur}.')
    pddata_cur = pd.read_csv(pathname_cur)
    fname_rot = os.path.join(datafolder_string,'cached',subname + 'rot.mat')
    if os.path.exists(fname_rot):
      dat = sp.io.loadmat(fname_rot)
      R   = dat["R"]
      x0  = dat["x0"]

    reach_trial = lr.ReachBody(
      pddata_cur, subname, "recording_"+filenums[ifl], R, x0, sr_datacollection)
    cached_folder_string = os.path.join(datafolder_string,"cached")
    reach_trial.click_add_reach_starts_ends(cached_folder=cached_folder_string,showfigs=False)
    distancelist, durationlist, peakspeedlist, indlist_middle_mvmnt_start_end = \
      reach_trial.mainsequence(cached_folder = cached_folder_string, showfigs = False)

    # filter out outliers. 
    peakspeedmax  = 2.0
    durationmax   = 2.0
    peakspeedlist[0,peakspeedlist[0,:]>peakspeedmax] = np.nan
    durationlist[0,durationlist[0,:]>durationmax] = np.nan

    # i use a 'mainSeq' object to wrap the D, V, T together. 
    ms = MainSeq(D = distancelist[0,:], V = peakspeedlist[0,:],T = durationlist[0,:])
    m_v = np.mean(peakspeedlist[0,:])
    m_t = np.mean(durationlist[0,:])
    ct,p,stats,fun_v,fun_t = lr.fit_ct(ms,normT = 1, normV = 1,kv=2,kt=1) #1.236,kt=1.572
    ps[isub,ifl] = p
    cts[isub,ifl] = ct

    # we will use 'table' to store the data for each subject
    table.loc[len(table)] = [isub,subj,cond,movetype,p,ct,ms]
    ######################################################################
    # Plotting! 
    dline = np.concatenate((np.linspace(0,0.02,100),np.linspace(0.03,.9,100)))   
    c = sb.color_from_condition(cond)
    ax[row_plt,0].plot(ms.D,ms.V,'o',color=c)
    ax[row_plt,0].plot(dline,fun_v(dline,ct),'-',color=c)
    ax[row_plt,1].plot(ms.D,ms.T,'o',color=c)
    ax[row_plt,1].plot(dline,fun_t(dline,ct),'-',color=c)

    # set the limits. 
    # if there are outliers, this will be a potential problem.
    ax[row_plt,0].set_xlim([0,.8])
    ax[row_plt,0].set_ylim([0, 2])
    ax[row_plt,1].set_xlim([0,.8])
    ax[row_plt,1].set_ylim([0, 2])
    
    ax[row_plt,0].set_xlabel('Distance (m)')
    ax[row_plt,0].set_ylabel('Peak speed (m/s)')
    ax[row_plt,1].set_xlabel('Distance (m)')
    ax[row_plt,1].set_ylabel('Duration (s)')
    f.suptitle(subj.name)
    # remote top and right box lines
    ax[row_plt,0].spines['top'].set_visible(False)
    ax[row_plt,0].spines['right'].set_visible(False)
    ax[row_plt,1].spines['top'].set_visible(False)
    ax[row_plt,1].spines['right'].set_visible(False)
        
  print("subject complete")
  plt.show(block=True)
    
# display cts.
np.set_printoptions(suppress=True)
print(np.round(cts.T,2))
print("Cell 3 complete.")

# %% Cell 4/X: Combine mainSeq objects for each subject and condition
# Output: mergereps_table, a table where we have combined the MainSeq 
# rows for each repetition of the same condition, for each subject. 
mergereps_table = pd.DataFrame(columns=['isub', 'cond', 'ms_combined','ct','p'])

for isub in table['isub'].unique():
  for cond in table['cond'].unique():
    # Filter rows for the current subject and condition
    subset = table[(table['isub'] == isub) & (table['cond'] == cond)]
    
    # Combine the mainSeq objects
    combined_D = np.concatenate([row.ms.D for _, row in subset.iterrows()])
    combined_V = np.concatenate([row.ms.V for _, row in subset.iterrows()])
    combined_T = np.concatenate([row.ms.T for _, row in subset.iterrows()])
    
    # Create a new mainSeq object with combined data
    ms_combined = MainSeq(D=combined_D, V=combined_V, T=combined_T)
    
    # Add to the combined table
    mergereps_table.loc[len(mergereps_table)] = [isub, cond, ms_combined, np.nan, np.nan]

# combined_table now contains the merged mainSeq objects for each subject and condition
print("Cell 4 complete.")

# %% Cell 5/X: Plot the combined MainSeq. 
for isub in mergereps_table['isub'].unique():
  f, ax = plt.subplots(4, 2, figsize=(10, 15))
  f.suptitle(f'Main Sequence for Subject {isub}', fontsize=16)

  for i_cond, cond in enumerate(mergereps_table['cond'].unique()):
    # Filter the combined table for the current subject and condition
    row = mergereps_table[(mergereps_table['isub'] == isub) & (mergereps_table['cond'] == cond)]
    if row.empty:
      continue

    ms_combined = row.iloc[0].ms_combined
    D = ms_combined.D
    V = ms_combined.V
    T = ms_combined.T

    # Fit the data using fit_ct
    ct, p, stats, fun_v, fun_t = lr.fit_ct(ms_combined, normT=1, normV=1, kv=2, kt=1.2)
    V_star = fun_v(dline, ct)
    T_star = fun_t(dline, ct)

    # Plot peak speed vs distance
    row_plt = i_cond
    ax[row_plt, 0].scatter(D, V, color=sb.color_from_condition(cond), label=f'Cond: {cond}')
    ax[row_plt, 0].plot(dline, V_star, '-', color=sb.color_from_condition(cond))
    ax[row_plt, 0].set_xlim([0, 0.8])
    ax[row_plt, 0].set_ylim([0, 2])
    ax[row_plt, 0].set_xlabel('Distance (m)')
    ax[row_plt, 0].set_ylabel('Peak speed (m/s)')
    ax[row_plt, 0].spines['top'].set_visible(False)
    ax[row_plt, 0].spines['right'].set_visible(False)

    # Plot duration vs distance
    ax[row_plt, 1].scatter(D, T, color=sb.color_from_condition(cond), label=f'Cond: {cond}')
    ax[row_plt, 1].plot(dline, T_star, '-', color=sb.color_from_condition(cond))
    ax[row_plt, 1].set_xlim([0, 0.8])
    ax[row_plt, 1].set_ylim([0, 1.6])
    ax[row_plt, 1].set_xlabel('Distance (m)')
    ax[row_plt, 1].set_ylabel('Duration (s)')
    ax[row_plt, 1].spines['top'].set_visible(False)
    ax[row_plt, 1].spines['right'].set_visible(False)

    # include title with ct
    ax[row_plt, 0].set_title(f'Condition: {cond}, ct = {ct:.2f}')

    # at this row, set p and ct now that we've run it.
    mergereps_table.at[row.index[0], 'p'] = p
    mergereps_table.at[row.index[0], 'ct'] = ct

  f.tight_layout(rect=[0, 0, 1, 0.96])
  plt.show(block=True)

print("")
print("Cell 5 (combine reps) complete.")

# %% cell 6/X: Digitize data with edges(tgts), then average across.
# output: mergereps_table gets a new column, ms_digitized.

for idx, row in mergereps_table.iterrows():
  ms_combined = row.ms_combined
  # get current cond
  curtgts = tgts[row.cond]
  edges_d = edges(curtgts)
  
  digitized = np.digitize(ms_combined.D, edges_d)
  mean_D = np.array([np.mean(ms_combined.D[digitized == i]) for i in range(1, len(edges_d))])
  mean_V = np.array([np.mean(ms_combined.V[digitized == i]) for i in range(1, len(edges_d))])
  mean_T = np.array([np.mean(ms_combined.T[digitized == i]) for i in range(1, len(edges_d))])
  mergereps_table.at[idx, 'ms_digitized'] = MainSeq(D=mean_D, V=mean_V, T=mean_T)

print("Cell 6 (digitize) complete.")
#%% Cell 7/X: Plot group averages for each condition.
# Loop across subjects to compute averages per condition.

for icond,cond in enumerate(['l','s','m','b']):

  tab_ = mergereps_table[(mergereps_table['cond']==cond)]
  mean_D_all = []
  mean_V_all = []
  mean_T_all = []
  std_V_all = []
  std_T_all = []
  std_D_all = []

  for i_bin in range(1, len(edges_d)):
    V_bin = []
    T_bin = []
    D_bin = []
    for _, row in tab_.iterrows():
      ms_digitized = row.ms_digitized
      if i_bin - 1 < len(ms_digitized.D):
        V_bin.append(ms_digitized.V[i_bin - 1])
        T_bin.append(ms_digitized.T[i_bin - 1])
        D_bin.append(ms_digitized.D[i_bin - 1])
    mean_V_all.append(np.nanmean(V_bin))
    mean_T_all.append(np.nanmean(T_bin))
    mean_D_all.append(np.nanmean(D_bin))
    std_V_all.append(np.nanstd(V_bin))
    std_T_all.append(np.nanstd(T_bin))
    std_D_all.append(np.nanstd(D_bin))

  # Plot averaged data with error bars
  dline = np.linspace(0, 0.8, 100)
  f, ax = plt.subplots(1, 2, figsize=(12, 6))

  # Fit and plot V(D)
  ms_avg = MainSeq(D=np.array(mean_D_all), V=np.array(mean_V_all), T=np.array(mean_T_all))
  ct, _, _, fun_v, fun_t = lr.fit_ct(ms_avg, normT=1, normV=1, kv=2, kt=1.2)
  V_star = fun_v(dline, ct)
  c = sb.color_from_condition(cond)
  ax[0].errorbar(mean_D_all, mean_V_all, yerr=std_V_all, fmt='o', label='Mean ± SD',color='k')
  ax[0].plot(dline, V_star, '-', label=f'Fit (ct={ct:.2f})',color=c)
  ax[0].set_xlim([0, 0.8])
  ax[0].set_ylim([0, 2])
  ax[0].set_xlabel('Distance (m)')
  ax[0].set_ylabel('Peak speed (m/s)')
  ax[0].legend()
  ax[0].spines['top'].set_visible(False)
  ax[0].spines['right'].set_visible(False)

  # Fit and plot T(D)
  T_star = fun_t(dline, ct)
  ax[1].errorbar(mean_D_all, mean_T_all, yerr=std_T_all, fmt='o', label='Mean ± SD',color='k')
  ax[1].plot(dline, T_star, '-', label=f'Fit (ct={ct:.2f})',color = c)
  ax[1].set_xlim([0, 0.8])
  ax[1].set_ylim([0, 1.6])
  ax[1].set_xlabel('Distance (m)')
  ax[1].set_ylabel('Duration (s)')
  ax[1].legend()
  ax[1].spines['top'].set_visible(False)
  ax[1].spines['right'].set_visible(False)

  f.suptitle('Averaged Main Sequence:'+ get_condition_text(cond), fontsize=16)
  f.tight_layout(rect=[0, 0, 1, 0.96])
  plt.show(block=True)
  
  save_figure(f, f'group_avg_{cond}.pdf')

print("Cell 7 (plot digitized) complete.")
# %%
