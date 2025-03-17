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

def filenums_for_subject(sub_num):
  #switchstatement on fname, string compare between known hardcoded values
  fnums = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']  
  if sub_num == '20250128':
    print("")
    #do nothing
    
  elif sub_num == '202501292':
    print("")
    #do nothing
  elif sub_num == '202501293':
    fnums = ['17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']
  return fnums


def conditions_for_subject(sub_num):
  #switchstatement on fname, string compare between known hardcoded values
  conds = ['l','l','l','l', 's','s','s','s', 'm','m','m','m', 'b','b','b','b']
  if sub_num == '20250128':
    print("")
    #do nothing
    
  elif sub_num == '202501292':
    conds = ['l','l','l','l', 'm','m','m','m', 's','s','s','s', 'b','b','b','b']
    #do nothing
  elif sub_num == '202501293':
    conds = ['l','l','l','l', 's','s','s','s', 'm','m','m','m', 'b','b','b','b']
    #do nothing
  
  return conds