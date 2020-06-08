from PIL import Image, ImageDraw, ImageFilter, ImageOps
import os
import numpy as np
import pathlib
import shutil
from config import root_dir

dataset_path = root_dir + 'tcav/dataset/google-512/dataset/'
#output_path = root_dir + 'tcav/dataset/for_tcav/'
colors = ['black','blue','brown','green','grey','orange','pink','purple','red','white','yellow']

def change_suffix(file_name, to_suffix):
    st = pathlib.PurePath(file_name).stem
    parent = str(pathlib.Path(file_name).parent)
    to_name = parent + '/' + st + to_suffix
    shutil.move(file_name, to_name)

# change extention  to png
to_suffix = '.png'
for color in colors:
    for path in os.listdir(dataset_path + color + '+color'):
        file_name = dataset_path + color + '+color/' + path
        change_suffix(file_name, to_suffix)
    for path in os.listdir(dataset_path + color + '+bolder'):
        file_name = dataset_path + color + '+bolder/' + path
        change_suffix(file_name, to_suffix)

for color in colors:
    if not os.path.exists(dataset_path + color):
        os.mkdir(dataset_path + color)

for color in colors:
    for i in range(len(os.listdir(dataset_path + color + '+color'))):
        if os.path.exists(dataset_path + color + '+color/' + str(i) + '.png') and os.path.exists(dataset_path + color + '+bolder/' + str(i) + '.png'):
            src = np.array(Image.open(dataset_path + color + '+color/' + str(i) + '.png'))
            mask = np.array(Image.open(dataset_path + color + '+bolder/' + str(i) + '.png'))
            mask = mask / 255
            mask = mask.astype(np.uint8)
            dst = src.copy()
            if len(src.shape) == 2:
                 dst *= mask
            else:
                dst[:,:,0] *= mask
                dst[:,:,1] *= mask
                dst[:,:,2] *= mask
            img = Image.fromarray(dst.astype(np.uint8)).resize((200, 200))
            img.save(dataset_path + color + '/' + str(i) + '.png')