import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import argparse
import shutil

# input path
input_path = '/home/tomohiro/code/tcav/tcav/dataset/for_tcav/-mnist-'

# define color list
color_lst = {}
color_lst['blue'] = [0,0,255]
color_lst['yellow'] = [255,255,0]
color_lst['red'] = [255,0,0]
color_lst['purple'] = [128,0,128]
color_lst['green'] = [0,128,0]

# add noize on RGB
noise_std = 20

len_color = len(color_lst)
for i, color in enumerate(color_lst):
    i+=5
    files = os.listdir(input_path + str(i))
    print(input_path[:-7] + color + input_path[-7:] + str(i))
    if not os.path.exists(input_path[:-7] + color + input_path[-7:] + str(i)):
        os.mkdir(input_path[:-7] + color + input_path[-7:] + str(i))
    for f in files:
        img = Image.open(input_path + str(i) + '/' + f).convert('L')
        rgb = color_lst[color] + np.random.normal(0, noise_std, 3)
        rgb = np.where(rgb <= 255, rgb, 255)
        rgb = np.where(rgb >= 0, rgb, 0)
        color_img = ImageOps.colorize(img, black=(0, 0, 0), white=rgb)
        color_img.save(input_path[:-7] + color + input_path[-7:] + str(i) + '/' + f)
