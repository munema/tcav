import os
import re
import shutil
from config import root_dir
from tcav.utils import pickle_load, pickle_dump

path = root_dir + 'log/2layers-mnist-number/random100'
target = 'mnist-4'
#bottleneck = 'conv1'

# activation_path = path + '/activations/'
# cav_path = path + '/cavs/'
tcav_path = path + '/tcavs/'
results_path = os.listdir(tcav_path)
all_results = []
non_dup_results = []
for result in results_path:
  positive_num = int(re.match(r'.+:.+:.+:random500_(\d+)_random500_(\d+)',result).group(1))
  negative_num = int(re.match(r'.+:.+:.+:random500_(\d+)_random500_(\d+)',result).group(2))
  if result.split(':')[1] == target:
    all_results.append(pickle_load(tcav_path+result))
    if positive_num < negative_num:
      non_dup_results.append(pickle_load(tcav_path+result))


print(len(all_results))
print(len(non_dup_results))

pickle_dump(all_results, path + '/' + target + '_results_all')
pickle_dump(non_dup_results,path + '/' + target + '_results_non_dup' )