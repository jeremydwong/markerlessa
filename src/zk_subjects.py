class subinfo():
  name = ''
  foldername = ''
  fnums = []
  conds = []
  def __init__(self, name, foldername, fnums, conds):
    self.name = name
    self.foldername = foldername
    self.fnums = fnums
    self.conds = conds
    # if you wanted to add a "slide" attribute, you could do that here

  def __str__(self):
    return f"Subject Name: {self.name}, Folder: {self.foldername}, File Numbers: {self.fnums}, Conditions: {self.conds}"

def get_figures_folder_for_project():
  return 'zk-figures'

def get_subject_info_list():
  
  sub_num = '20250128'
  foldername = '2025-01-28'
  fnums = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']  
  conds = ['l','l','l','l', 's','s','s','s', 'm','m','m','m', 'b','b','b','b']
  sub1 = subinfo(sub_num, foldername, fnums, conds)
  
  sub_num = '202501292'
  foldername = '2025-01-29b'
  conds = ['l','l','l','l', 'm','m','m','m', 's','s','s','s', 'b','b','b','b']
  sub2 = subinfo(sub_num, foldername, fnums, conds)
  
  sub_num = '202501293'
  foldername = '2025-01-29b'
  fnums = ['17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']
  conds = ['l','l','l','l', 's','s','s','s', 'm','m','m','m', 'b','b','b','b']
  sub3 = subinfo(sub_num, foldername, fnums, conds)
  
  return [sub1, sub3] # add each subject 

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
