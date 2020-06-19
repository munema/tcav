import os
import re
import shutil
from config import root_dir
from tcav.utils import pickle_load, pickle_dump

path = root_dir + 'log/3layer-mnist/random50'
create_path = path

activation_path = path + '/activations/'
cav_path = path + '/cavs/'
tcav_path = path + '/tcavs/'

if path != create_path:

  is_act = True
  try:
    act = os.listdir(activation_path)
    random_act_path = [activation_path + element for element in act if element[:11] == 'acts_random']
  except:
    is_act = False

  is_cav = True
  try:
    cav = os.listdir(cav_path)
    random_cav_path = [cav_path + element for element in cav if element[:6] == 'random']
  except:
    is_cav = False

  is_tcav = True
  try:
    tcav = os.listdir(tcav_path)
    random_tcav_path = [tcav_path + element for element in tcav if re.match(r'.+:.+:.+:random',element)]
  except:
    is_tcav = False

  if not os.path.exists(create_path):
    os.mkdir(create_path)
  if is_act:
    if not os.path.exists(create_path + '/activations'):
      os.mkdir(create_path + '/activations')
    for rand_act in random_act_path:
      if not os.path.exists(create_path + '/activations/' + rand_act.split('/')[-1]):
        shutil.copyfile(rand_act, create_path + '/activations/' + rand_act.split('/')[-1])

  if is_cav:
    if not os.path.exists(create_path + '/cavs'):
      os.mkdir(create_path + '/cavs')
    for rand_cav in random_cav_path:
      if not os.path.exists(create_path + '/cavs/' + rand_cav.split('/')[-1]):
        shutil.copyfile(rand_cav, create_path + '/cavs/' + rand_cav.split('/')[-1])

  if is_tcav:
    if not os.path.exists(create_path + '/tcavs'):
      os.mkdir(create_path + '/tcavs')
    for rand_tcav in random_tcav_path:
      if not os.path.exists(create_path + '/tcavs/' + rand_tcav.split('/')[-1]):
        shutil.copyfile(rand_tcav, create_path + '/tcavs/' + rand_tcav.split('/')[-1])


random_path = create_path

# activation_path = random_path + '/activations/'
# cav_path = random_path + '/cavs/'
tcav_path = random_path + '/tcavs/'

results_path = os.listdir(tcav_path)
all_results = []
non_dup_results = []
lst = []
lst2 = []
for result in results_path:
  positive_num = int(re.match(r'.+:.+:.+:random500_(\d+)_random500_(\d+)',result).group(1))
  negative_num = int(re.match(r'.+:.+:.+:random500_(\d+)_random500_(\d+)',result).group(2))
  if result.split(':')[1] == 'mnist_red':
    if positive_num == 0:
      lst.append(result)
    if positive_num == 1:
      lst2.append(result)
  all_results.append(pickle_load(tcav_path+result))
  if positive_num < negative_num:
    non_dup_results.append(pickle_load(tcav_path+result))


print(len(all_results))
print(len(non_dup_results))

import collections
c = collections.Counter(lst)
d = collections.Counter([l.split(':')[0] for l in lst])

print(d)
d2 = collections.Counter([l.split(':')[0] for l in lst2])

print(d)
print(d2)
#pickle_dump(all_results, random_path + '/_results_all')
#pickle_dump(non_dup_results,random_path + '/_results_non_dup' )