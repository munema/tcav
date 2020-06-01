import os
import re
import shutil
import random
from config import root_dir

class_lst = [['Apple Crimson Snow', 'apple_red'], ['Apple Golden 2', 'apple_yellow'], ['Apple Granny Smith', 'apple_green'], ['Apple Red Yellow 1', 'apple_red_yellow'], ['Avocado', 'avocado_green'], ['Avocado ripe', 'avocado_black'], ['Banana Lady Finger', 'banana_yellow'], ['Banana Red', 'banana_red'], ['Blueberry', 'blueberry_blue'], ['Cantaloupe 1','cantaloupe_yellow'], ['Cantaloupe 2', 'cantaloupe_green'], ['Cherry 2', 'cherry_red'], ['Cherry Wax Yellow', 'cherry_yellow'], ['Corn', 'corn_yellow'], ['Corn Husk', 'corn_green'], ['Grape White 2', 'grape_white_green'], ['Grape Pink', 'grape_red'], ['Lemon', 'lemmon_yellow'], ['Lemon Meyer', 'lemmon_orange'], ['Limes', 'limes_green'], ['Onion Red Peeled', 'onion_paple'], ['Onion White', 'onion_white_brown'], ['Pepper Green', 'pepper_green'], ['Pepper Red', 'pepper_red'], ['Pepper Yellow', 'pepper_orange']]

source_train_path = root_dir + 'tcav/dataset/for_train/Fruit-Images-Dataset/Training/'
source_test_path = root_dir + 'tcav/dataset/for_train/Fruit-Images-Dataset/Test/'
train_path = root_dir + 'tcav/dataset/for_train/Fruit-Images-Dataset/train'
test_path = root_dir + 'tcav/dataset/for_train/Fruit-Images-Dataset/test'
for_tcav_path = root_dir + 'tcav/dataset/for_tcav'
num_for_tcav_data = 100

if not os.path.exists(train_path):
  os.mkdir(train_path)
if not os.path.exists(test_path):
  os.mkdir(test_path)



for cls in class_lst:
  trn_files = os.listdir(source_train_path + cls[0])
  random.shuffle(trn_files)
  tst_files = os.listdir(source_test_path + cls[0])
  random.shuffle(tst_files)
  
  if not os.path.exists(for_tcav_path + '/' + cls[1]):
    os.mkdir(for_tcav_path + '/' + cls[1])
  if not os.path.exists(train_path + '/' + cls[1]):
    os.mkdir(train_path + '/' + cls[1])    
  if not os.path.exists(test_path + '/' + cls[1]):
    os.mkdir(test_path +'/' + cls[1])    
  
  for i, trn_file in enumerate(trn_files):
    if i < num_for_tcav_data//2:
      shutil.copyfile(source_train_path + cls[0] + '/' + trn_file, for_tcav_path + '/' + cls[1] + '/' + trn_file)
    else:
       shutil.copyfile(source_train_path + cls[0] + '/' + trn_file, train_path + '/' + cls[1] + '/' + trn_file)

  for i, tst_file in enumerate(tst_files):
    if i < num_for_tcav_data//2:
      shutil.copyfile(source_test_path + cls[0] + '/' + tst_file, for_tcav_path + '/' + cls[1] + '/' + tst_file)
    else:
       shutil.copyfile(source_test_path + cls[0] + '/' + tst_file, test_path + '/' + cls[1] + '/' + tst_file)