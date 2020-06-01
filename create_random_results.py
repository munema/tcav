import os
import re
import shutil
from config import root_dir
from tcav.utils import pickle_load, pickle_dump

random_path = root_dir + 'log/Random50-50'

activation_path = random_path + '/activations/'
cav_path = random_path + '/cavs/'
tcav_path = random_path + '/tcavs/'

results_path = os.listdir(tcav_path)
all_results = []
non_dup_results = []

for result in results_path:
  positive_num = int(re.match(r'.+:.+:random500_(\d+)_random500_(\d+)',result).group(1))
  negative_num = int(re.match(r'.+:.+:random500_(\d+)_random500_(\d+)',result).group(2))


  all_results.append(pickle_load(tcav_path+result))
  if positive_num < negative_num:
    non_dup_results.append(pickle_load(tcav_path+result))


print(len(all_results))
print(len(non_dup_results))

pickle_dump(all_results, random_path + '/Results_all')
pickle_dump(non_dup_results,random_path + '/Results_non_dup' )
