import os
import re
import shutil
from config import root_dir

path = root_dir + 'log/GoogleNet_mixed3a_mixed4a_mixed4c_mixed4e_mixed5b:fire engine:blue_green_red_yellow_0'

create_path = root_dir + 'log/Random50-50'

activation_path = path + '/activations/'
cav_path = path + '/cavs/'
tcav_path = path + '/tcavs/'

act = os.listdir(activation_path)
random_act_path = [activation_path + element for element in act if element[:11] == 'acts_random']
cav = os.listdir(cav_path)
random_cav_path = [cav_path + element for element in cav if element[:6] == 'random']
tcav = os.listdir(tcav_path)
random_tcav_path = [tcav_path + element for element in tcav if re.match(r'.+:.+:random',element)]

if not os.path.exists(create_path + '/activations'):
  os.mkdir(create_path + '/activations')
for rand_act in random_act_path:
  if not os.path.exists(create_path + '/activations/' + rand_act.split('/')[-1]):
    shutil.copyfile(rand_act, create_path + '/activations/' + rand_act.split('/')[-1])
    
if not os.path.exists(create_path + '/cavs'):
  os.mkdir(create_path + '/cavs')
for rand_cav in random_cav_path:
  if not os.path.exists(create_path + '/cavs/' + rand_cav.split('/')[-1]):
    shutil.copyfile(rand_cav, create_path + '/cavs/' + rand_cav.split('/')[-1])
    
if not os.path.exists(create_path + '/tcavs'):
  os.mkdir(create_path + '/tcavs')
for rand_tcav in random_tcav_path:
  if not os.path.exists(create_path + '/tcavs/' + rand_tcav.split('/')[-1]):
    shutil.copyfile(rand_tcav, create_path + '/tcavs/' + rand_tcav.split('/')[-1])