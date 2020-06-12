import os
import re
import shutil
from config import root_dir
from tcav.utils import pickle_load, pickle_dump

root_path = '/home/tomohiro/code/tcav/tcav/dataset/for_tcav'
output_dir = '/home/tomohiro/code/tcav/random'
NUM = 141

path = os.listdir(root_path)

for result in path:
    try:
        num = int(re.match(r'random500_(\d+)',result).group(1))
    except:
        continue
    if num >= NUM:
        if not os.path.exists(output_dir + '/' + result):
            shutil.copytree(root_path + '/' + result,output_dir + '/' + result)