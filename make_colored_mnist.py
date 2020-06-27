import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import argparse
import shutil

# parser = argparse.ArgumentParser(description='Create Colored (Red, Blue, Yelllow, Green, Purple) MNIST dataset.')
# parser.add_argument('--output_dir', type=str,
#                     help='Name for the directory where we will create the data.')
# parser.add_argument('--noise_std', type=int,
#                     help='Noise std on normal distribution to add RGB')

# number or color
output_label = 'number'

# output path
output_path = '/home/tomohiro/code/tcav/tcav/dataset/for_train/Colored_mnist' + '/' + output_label
#output_path = '/home/tomohiro/code/tcav/tmp/Colored_mnist'


if not os.path.exists(output_path + '/train'):
    os.mkdir(output_path + '/train')

if not os.path.exists(output_path + '/test'):
    os.mkdir(output_path + '/test')

# mnist load
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# output size
WIDTH = 200
HEIGHT = 200

# add noize on RGB
noise_std = 20



# define color list
color_lst = {}
color_lst['blue'] = [0,0,255]
color_lst['yellow'] = [255,255,0]
color_lst['red'] = [255,0,0]
color_lst['purple'] = [128,0,128]
color_lst['green'] = [0,128,0]

len_color = len(color_lst)
for i in range(10):
    train_inx = np.where(y_train == i)[0]
    np.random.seed(seed = 32)
    np.random.shuffle(train_inx)
    len_train = len(train_inx) // len_color

    test_inx = np.where(y_test == i)[0]
    np.random.seed(seed = 32)
    np.random.shuffle(test_inx)
    len_test = len(test_inx) // len_color
    for j, color in enumerate(color_lst):
        if output_label == 'color':
            out_path = color
        elif output_label == 'number':
            out_path = str(i)
        else:
            print('output_label must be color or number')
            sys.exist()
        if not os.path.exists(output_path + '/train/mnist_' + out_path):
            os.mkdir(output_path + '/train/mnist_' + out_path)
        if not os.path.exists(output_path + '/test/mnist_' + out_path):
            os.mkdir(output_path + '/test/mnist_' + out_path)

        c_train_inx = train_inx[j*len_train:(j+1)*len_train]
        c_test_inx = test_inx[j*len_test:(j+1)*len_test]
        c_train = x_train[c_train_inx]
        for k in range(len_train):
            img = Image.fromarray(c_train[k])
            rgb = color_lst[color] + np.random.normal(0, noise_std, 3)
            rgb = np.where(rgb <= 255, rgb, 255)
            rgb = np.where(rgb >= 0, rgb, 0)
            color_img = ImageOps.colorize(img, black=(0, 0, 0), white=rgb).resize((WIDTH,HEIGHT))
            color_img.save(f'{output_path}/train/mnist_{out_path}/{i}_{color}_{k}.jpg')

        c_test_inx = test_inx[j*len_test:(j+1)*len_test]
        c_test_inx = test_inx[j*len_test:(j+1)*len_test]
        c_test = x_test[c_test_inx]
        for k in range(len_test):
            img = Image.fromarray(c_test[k])
            rgb = color_lst[color] + np.random.normal(0, noise_std, 3)
            rgb = np.where(rgb <= 255, rgb, 255)
            rgb = np.where(rgb >= 0, rgb, 0)
            color_img = ImageOps.colorize(img, black=(0, 0, 0), white=rgb).resize((WIDTH,HEIGHT))
            color_img.save(f'{output_path}/test/mnist_{out_path}/{i}_{color}_{k}.jpg')


# move for tcav
move_path = '/home/tomohiro/code/tcav/tcav/dataset/for_tcav'
#move_path = '/home/tomohiro/code/tcav/tmp/for_tcav'
if output_label == 'color':
    rang = color_lst
    _rang = np.arange(10)
elif output_label == 'number':
    rang = np.arange(10)
    _rang = color_lst
for r in rang:
    lst = os.listdir(output_path + '/train/mnist_' + str(r))
    for i in _rang:
        if output_label == 'color':
            i_lst = [s for s in lst if s.startswith(str(i))]
            out_path = r
        elif output_label == 'number':
            i_lst = [s for s in lst if i in s]
            out_path = str(r)

        if not os.path.exists(f'{move_path}/mnist_{out_path}'):
            os.mkdir(f'{move_path}/mnist_{out_path}')
        for file in i_lst[:50]:
            # print(f'{output_path}/train/{color}/{file}')
            shutil.move(f'{output_path}/train/mnist_{out_path}/{file}',f'{move_path}/mnist_{out_path}/{file}')


# move for tcav (test → shape concept)
# (_, _), (x_test, y_test) = mnist.load_data()
# move_path = '/home/tomohiro/code/tcav/tcav/dataset/for_tcav'
# for i in range(10):
#     if not os.path.exists(f'{move_path}/mnist_{i}'):
#         os.mkdir(f'{move_path}/mnist_{i}')
#     inx = np.where(y_test == i)[:100][0]
#     for j,k in enumerate(inx):
#         img = Image.fromarray(x_test[k])
#         img = img.resize((WIDTH,HEIGHT))
#         img.save(f'{move_path}/mnist_{i}/{j}.jpg')
