class subinfo():
  name = ''
  foldername = ''
  fnums = []
  conds = []
  def __init__(self, name, foldername, fnums, conds, movetype=[]):
    self.name = name
    self.foldername = foldername
    self.fnums    = fnums
    self.conds    = conds
    self.movetype = movetype # 0, 1, 2
    # if you wanted to add a "slide" attribute, you could do that here

  def __str__(self):
    return f"Subject Name: {self.name}, Folder: {self.foldername}, File Numbers: {self.fnums}, Conditions: {self.conds}"

def get_figures_folder_for_project():
  return 'jh-figures'

def get_subject_info_list():

  sub_num = '20250128'
  foldername = '2025-01-28'
  fnums = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']  
  conds = ['l','l','l','l', 's','s','s','s', 'm','m','m','m', 'b','b','b','b']
  fls   = [  0,  0,  0,  1,  0,  0,  0,  1,   0,  0,  0,  1,   0,  0,  0,  1]
  sub1 = subinfo(sub_num, foldername, fnums, conds, movetype=fls)
  
  sub_num = '202501292'
  foldername = '2025-01-29b'
  fnums = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']  
  conds = ['l','l','l','l', 'm','m','m','m', 's','s','s','s', 'b','b','b','b']
  fls   = [  0,  0,  0,  1,  0,  0,  0,  1,   0,  0,  0,  1,   0,  0,  0,  1]
  sub2 = subinfo(sub_num, foldername, fnums, conds, movetype=fls)

  sub_num = '202501293'
  foldername = '2025-01-29b'
  fnums = ['17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']
  conds = ['l','l','l','l', 's','s','s','s', 'm','m','m','m', 'b','b','b','b']
  sub3 = subinfo(sub_num, foldername, fnums, conds, movetype=fls)
    
  sub_num = '202502031'
  foldername = '2025-02-03'
  fnums = ['1', '3', '4', '6', '7', '8', '10', '11', '12' '13', '14', '15', '16', '17', '19']
  conds = ['l','l','l','l', 'm','m','m','m', 'b','b','b','b', 's','s','s','s']
  sub4 = subinfo(sub_num, foldername, fnums, conds, movetype=fls)
  
  sub_num = '202502032'
  foldername = '2025-02-03'
  conds = ['l','l','l','l', 's','s','s','s', 'b','b','b','b', 'm','m','m','m']
  fnums = ['20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '35', '36']
  sub5 = subinfo(sub_num, foldername, fnums, conds, movetype=fls)

  sub_num = '202502192'
  foldername = '2025-02-19'
  conds = ['l','l','l','l', 'm','m','m','m', 's','s','s','s', 'b','b','b','b']
  fnums = ['20', '21', '22', '24', '25', '26', '27', '28', '29' '30', '31', '32', '33', '35', '36', '37']
  sub6 = subinfo(sub_num, foldername, fnums, conds, movetype=fls)

  sub_num = '202502193'
  foldername = '2025-02-19'
  conds = ['l','l','l','l', 'm','m','m','m', 'b','b','b','b', 's','s','s','s']
  fnums = ['38', '39', '40', '41', '42', '43', '44', '45', '46' '48', '49', '50', '51', '53', '54', '55']
  sub7 = subinfo(sub_num, foldername, fnums, conds, movetype=fls)

  sub_num = '20250319'
  foldername = '2025-03-19'
  conds = ['l','l','l','l', 'm','m','m','m', 'b','b','b','b', 's','s','s','s']
  fnums = ['1', '2', '3', '4', '5', '6', '7', '8', '9' '10', '11', '12', '13', '14', '15', '17']  
  sub8 = subinfo(sub_num, foldername, fnums, conds, movetype=fls)

  return [sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8] # add each subject

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
