#%%
#todo: replace rf with lr
datafolder_string_zara  = r"/Users/zarwareenkhan/Library/CloudStorage/OneDrive-Personal/Undergrad 2024 Shared Folder"
datafolder_string_jer   = r"/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Undergrad 2024 Shared Folder/"
datafolder_string_jess  = r"C:\Users\JC\University of Calgary\Jeremy Wong - Undergrad 2024 Shared Folder"
import libreach as lr
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy as sp
# folder files for each person
import jh_subjects as jh

class mainSeq():
  D = []
  V = []
  T = []
  def __init__(self,D=[],V=[],T=[]):
    self.D = D
    self.V = V
    self.T = T
    
subnames      = ['202501293','20250128']
folderstring  = ['2025-01-29b','2025-01-28']
sr_datacollection = 30
datafolder_string = datafolder_string_jer #one of the four above

num_conditions = 16
cts = np.zeros((len(subnames),num_conditions))
ps = np.zeros((len(subnames),num_conditions))

#define target distances hardcoded here. 
tgts = np.array([.02,.05,.1,.2,.3,.45,.6])
#define edges intermediate to tgts array in loop
edges_d = np.zeros((tgts.shape[0]+1))
edges_d[0] = 0.0
for i in range(1,len(tgts)):
  edges_d[i] = (tgts[i-1]+tgts[i])/2

edges_d[-1] = tgts[-1]+.15

mlist_pkvel = list()
mlist_tdur = list()
mlist_dist = list()

# each mlist_pkvel should have 4 conditions

def fullfile_for_date_and_recordingnum(datestring, trialnum):
  model_type = "SIMPLE_HOLISTIC"
  return os.path.join(datafolder_string,datestring,"recordings","recording_"+str(trialnum),model_type, "xyz_"+model_type+"_labelled.csv")

# choose the colors you would like for each condition
def color_from_condition(cond):
  if cond == 'l':
    return '#deebf7'
  elif cond == 's':
    return '#efedf5'
  elif cond == 'm':
    return '#bcbddc'
  elif cond == 'b':
    return '#756bb1'

#%%
isub = 0
subj = subnames[isub]
#initialize a pd dataframe to hold all subjects' data
table = pd.DataFrame(columns = ['isub','subj','cond','p','ct','ms'])

for isub, subj in enumerate(subnames):
  filenums  = jh.filenums_for_subject(subj)
  conds     = jh.conditions_for_subject(subj)
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
    
    pathname_cur = fullfile_for_date_and_recordingnum(folderstring[isub],filenums[ifl])
    print(f'Main Sequence (speed, duration): iteration {ifl}; filename {pathname_cur}.')
    pddata_cur = pd.read_csv(pathname_cur)
    fname_rot = os.path.join(datafolder_string,'cached',subnames[isub] + 'rot.mat')
    if os.path.exists(fname_rot):
      dat = sp.io.loadmat(fname_rot)
      R   = dat["R"]
      x0  = dat["x0"]

    reach_trial = lr.ReachBody(pddata_cur, subnames[isub], "recording_"+filenums[ifl], R, x0, sr_datacollection)
    cached_folder_string = os.path.join(datafolder_string,"cached")
    reach_trial.click_add_reach_starts_ends(cached_folder=cached_folder_string)

    distancelist, durationlist, peakspeedlist, indlist_middle_mvmnt_start_end = reach_trial.mainsequence(cached_folder = cached_folder_string)

    # filter out outliers
    peakspeedmax  = 2.0
    durationmax   = 2.0
    peakspeedlist[0,peakspeedlist[0,:]>peakspeedmax] = np.nan
    durationlist[0,durationlist[0,:]>durationmax] = np.nan

    # i use a 'mainSeq' object to wrap the D, V, T together. 
    ms = mainSeq(D = distancelist[0,:], V = peakspeedlist[0,:],T = durationlist[0,:])
    ct,p,stats,fun_v,fun_t = lr.fit_ct(ms,normT = 1.0, normV = 1.0)
    ps[isub,ifl] = p
    cts[isub,ifl] = ct

    # discretize distancelist 
    digitized = np.digitize(distancelist[0,:],edges_d)
    
    vels.append(np.array([np.mean(peakspeedlist[0,digitized==i]) for i in range(1,edges_d.shape[0]+1)]))
    durs.append([np.mean(durationlist[0,digitized==i]) for i in range(1,edges_d.shape[0]+1)])
    dists.append(np.array([np.mean(distancelist[0,digitized==i]) for i in range(1,edges_d.shape[0]+1)]))

    # plot the main sequence data, then the fit on top.
    
    c = color_from_condition(cond)
    dline = np.linspace(0,0.7,100)
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
    f.suptitle(subj)
    # remote top and right box lines
    ax[row_plt,0].spines['top'].set_visible(False)
    ax[row_plt,0].spines['right'].set_visible(False)
    ax[row_plt,1].spines['top'].set_visible(False)
    ax[row_plt,1].spines['right'].set_visible(False)
    
    
    #
    table.loc[len(table)] = [isub,subj,cond,p,ct,ms]
    
  print("subject complete")
  plt.show(block=True)
    
# display cts.
np.set_printoptions(suppress=True)
print(np.round(cts.T,2))

# %% plot error bars from mlist_vel
conditionlist = ['l','s','m','b']

Kvs = list()
Kts = list()
for i_c in range(len(conditionlist)):
  
  N = len(subnames)
  c = color_from_condition(conditionlist[i_c])
  
  # make mlistrows be the rows of the table where the condition is conditionlist[i]
  mlist_ = table.loc[table.cond == conditionlist[i_c]]

  f1,ax1 = plt.subplots(len(mlist_),2,figsize=(10,5))

  for j in range(0,len(mlist_)):
    D = mlist_.iloc[j].ms.D
    V = mlist_.iloc[j].ms.V
    T = mlist_.iloc[j].ms.T
    ax1[j,0].scatter(D,V,color=c)
    ax1[j,1].scatter(D,T,color=c)
    ms = mainSeq(D = D, V = V,T = T)

    A = D**(3/4)
    b = V

    results = smf.ols('b ~ A -1', data = pd.DataFrame({'A':A,'b':b})).fit()
    Kv      = results.params.A # note: this Kv is independently fit, until new gains for movement are generated.
    Kvs.append(Kv)
    A=D**(1/4)
    b=T
    results = smf.ols('b ~ A -1', data = pd.DataFrame({'A':A,'b':b})).fit()
    Kt      = results.params.A # note: this Kt is independently fit, as above.
    Kts.append(Kt)
    dline = np.concatenate((np.linspace(0,0.02,100),np.linspace(0.03,.9,100)))
    V_star = Kv*dline**(3/4)
    T_star = Kt*dline**(1/4)
    ax1[j,0].plot(dline,V_star,'-',color=c)
    ax1[j,1].plot(dline,T_star,'-',color=c)

    ax1[j,0].set_ylim([0,2])
    ax1[j,0].set_xlim([0,.8])
    ax1[j,0].set_ylabel('Peak speed (m/s)')
    ax1[j,0].set_xlabel('Distance (m)')

    ax1[j,1].set_ylim([0,1.6])
    ax1[j,1].set_xlim([0,.8])
    ax1[j,1].set_ylabel('Duration (s)')
    ax1[j,1].set_xlabel('Distance (m)')

    # remove top and right box lines
    ax1[j,0].spines['top'].set_visible(False)
    ax1[j,0].spines['right'].set_visible(False)
    ax1[j,1].spines['top'].set_visible(False)
    ax1[j,1].spines['right'].set_visible(False)

    # for label in (ax1[0].get_xticklabels() + ax1[0].get_yticklabels()+ax1[1].get_xticklabels()+ax1[1].get_yticklabels()):
    
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels

    # ax1[j,0].set_xticks([0,.1,.2,.3,.4])
    # ax1[j,0].set_yticks([0,.5,1,1.5,2])
    # ax1[j,1].set_xticks([0,.1,.2,.3,.4])
    # ax1[j,1].set_yticks([0,.5,1,1.5,2])

    # set dimensions of the subplots 
    
    # f1.savefig('figures/means_vt.pdf')
    # give the plot a title
  f1.suptitle('Main Sequence for ' + conditionlist[i_c])
  f1.tight_layout()
  plt.show(block=True)
    



# %%
f2,ax2 = plt.subplots(1,2,figsize=(10,5))
for i in range(4,7):
  N = len(subnames)
  c = color_from_condition(conditionlist[i_c])
  D = mlist_D[i]
  V = mlist_vel[i]
  T = mlist_t[i]
  ax2[0].errorbar(np.nanmean(D,axis=0),np.nanmean(V,axis=0),yerr=np.nanstd(V,axis=0),color=c,linestyle='none',capsize=3,marker='o')
  ax2[1].errorbar(np.nanmean(D,axis=0),np.nanmean(T,axis=0),yerr=np.nanstd(T,axis=0),color=c,linestyle='none',capsize=3,marker='o')

  ms = mainSeq(D = np.nanmean(D,axis=0), V = np.nanmean(V,axis=0),T = np.nanmean(T,axis=0))

  A = np.nanmean(D,axis=0)**(3/4)
  b = np.nanmean(V,axis=0)

  results = smf.ols('b ~ A -1', data = pd.DataFrame({'A':A,'b':b})).fit()
  Kv      = results.params.A # note: this Kv is independently fit, until new gains for movement are generated.

  A = np.nanmean(D,axis=0)**(1/4)
  b = np.nanmean(T,axis=0)
  results = smf.ols('b ~ A -1', data = pd.DataFrame({'A':A,'b':b})).fit()
  Kt      = results.params.A # note: this Kt is independently fit, as above.

  dline = np.concatenate((np.linspace(0,0.02,100),np.linspace(0.03,.75,100)))
  V_star = Kv*dline**(3/4)
  T_star = Kt*dline**(1/4)
  ax2[0].plot(dline,V_star,'-',color=c)
  ax2[1].plot(dline,T_star,'-',color=c)


ax2[0].set_ylim([0,2])
ax2[0].set_xlim([0,.4])
ax2[0].set_ylabel('Peak speed (m/s)')
ax2[0].set_xlabel('Distance (m)')

ax2[1].set_ylim([0,1.6])
ax2[1].set_xlim([0,.4])
ax2[1].set_ylabel('Duration (s)')
ax2[1].set_xlabel('Distance (m)')

# remove top and right box lines
ax2[0].spines['top'].set_visible(False)
ax2[0].spines['right'].set_visible(False)
ax2[1].spines['top'].set_visible(False)
ax2[1].spines['right'].set_visible(False)

for label in (ax2[0].get_xticklabels() + ax2[0].get_yticklabels()+ax2[1].get_xticklabels()+ax2[1].get_yticklabels()):
  label.set_fontsize(18)
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels

ax2[0].set_xticks([0,.2,.4,.6,.8])
ax2[0].set_yticks([0,.5,1,1.5,2])
ax2[1].set_xticks([0,.2,.4,.6,.8])
ax2[1].set_yticks([0,.5,1,1.5,2])


# set dimensions of the subplots 
f2.tight_layout()
# f2.savefig('figures/means_acc_vt.pdf')

plt.show(block=True)

# %%
