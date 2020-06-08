#%%
import os
from PIL import Image
import numpy as np

height = 200
width = 200
num_img = 100
dataset_path = '/home/tomohiro/code/tcav/tcav/dataset/for_tcav'
noise_std = 20

blank = np.zeros((height, width, 3))
color_lst = {}
color_lst['blue'] = [255,0,0]
#color_lst['white'] = [255,255,255]
color_lst['yellow'] = [0,255,255]
color_lst['red'] = [0,0,255]
#color_lst['black'] = [0,0,0]
color_lst['purple'] = [128,0,128]
color_lst['green'] = [0,128,0]

for i, color in enumerate(color_lst):
    blank = np.zeros((height, width, 3))
    if not os.path.exists(dataset_path + '/smiple-' + color):
        os.mkdir(dataset_path + '/simple-' + color)
    for j in range(num_img):
        np.random.seed(seed=i+j)
        # mean, std, len
        noise = np.random.normal(0, noise_std, 3)
        noised_color = color_lst[color][::-1] + noise
        noised_color = np.where(noised_color < 0, 0, noised_color)
        noised_color = np.where(noised_color > 255, 255, noised_color)
        colored = np.uint8(blank + noised_color)
        img = Image.fromarray(colored)
        img.save(dataset_path + '/simple-' + color + f'/{j}.jpg')



# %%
