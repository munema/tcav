from PIL import Image, ImageDraw
from config import root_dir
import matplotlib.pyplot as plt
import numpy as np
import random
import os

def get_h_line_img(leng = 0.8, angle = 0.3,dx = 0, dy = 0, w = 8,SIZE = 200):
    im = Image.new('RGB', (SIZE, SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    l = leng*SIZE//2
    h = angle*SIZE//2
    dx *=  SIZE//2
    dy *=  SIZE//2
    draw.line((-l + SIZE//2 + dx, - h + SIZE//2 + dy, l + SIZE//2 + dx, h + SIZE//2 + dy), fill=(255, 255, 255), width=w)
    return im

def get_v_line_img(leng = 0.8, angle = 0.3,dx = 0, dy = 0, w = 8,SIZE = 200):
    im = Image.new('RGB', (SIZE, SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    h = leng*SIZE//2
    l = angle*SIZE//2
    dx *=  SIZE//2
    dy *=  SIZE//2
    draw.line((-l + SIZE//2 + dx, - h + SIZE//2 + dy, l + SIZE//2 + dx, h + SIZE//2 + dy), fill=(255, 255, 255), width=w)
    return im
  
def get_circle_img(leng = 0.8,dx = 0, dy = 0, w = 8,SIZE = 200):
    im = Image.new('RGB', (SIZE, SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    l1 = leng*SIZE
    l2 = (1 - leng)*SIZE
    h1 = l1
    h2 = l2
    dx *=  SIZE//2
    dy *=  SIZE//2
    draw.arc((l2 + dx*SIZE, h2 + dy, l1 + dx, h1 + dy), start=0, end=360, fill=(255, 255, 255), width=w)
    return im
  
def get_half_circle_img(leng = 0.8, start = 0,dx = 0, dy = 0, w = 8,SIZE = 200):
    im = Image.new('RGB', (SIZE, SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    l1 = leng*SIZE
    l2 = (1 - leng)*SIZE
    h1 = l1
    h2 = l2
    dx *=  SIZE//2
    dy *=  SIZE//2
    st = start
    ed = st + 180
    if ed > 360:
        ed -= 360
    draw.arc((l2 + dx*SIZE, h2 + dy, l1 + dx, h1 + dy), start=st, end=ed, fill=(255, 255, 255), width=w)
    return im
  
def get_sharp_img(leng = 0.2, sharp = 0.2,dx = 0, dy = 0, rotate = 10,w = 8,SIZE = 200):
    im = Image.new('RGB', (SIZE, SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    l = leng * SIZE//2
    h = sharp*SIZE//2
    plus = leng * 20
    points = (-l + SIZE//2,- h + SIZE//2), (l + SIZE//2, SIZE//2), (-l + SIZE//2, h + SIZE//2)
    #draw.line(((-l + SIZE//2,- h + SIZE//2), (l + SIZE//2, SIZE//2), (-l + SIZE//2, h + SIZE//2)), fill=(255, 255, 255),width=w)
    draw.polygon(points, outline=(0, 0, 0), fill=(255, 255, 255))
    tan = h/np.linalg.norm(np.array(points[0])-np.array(points[1]))
    w = w/tan
    _points = (-l + SIZE//2 - w,- h + SIZE//2), (l + SIZE//2 - w, SIZE//2), (-l + SIZE//2 - w, h + SIZE//2)
    draw.polygon(_points, outline=(0, 0, 0), fill=(0, 0, 0))
    return im.rotate(rotate)
  
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def get_4_concat(im1, im2, im3, im4):
    dst = Image.new('RGB', (im1.width + im1.width, im1.height + im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    dst.paste(im3, (im1.width, 0))
    dst.paste(im4, (im1.width, im1.height))
    return dst.resize((im1.width,im1.height))
  
  
  
multi_shape_concept = ['roundness','straight','sharpness']
single_shape_concept = ['v_line','h_line','circle','half_circle','sharp']
save_path = root_dir + 'tcav/dataset/for_tcav/'

np.random.seed(42)
random.seed(42)

max_examples = 200

for concept in multi_shape_concept:
  if concept == 'roundness':
      for i in range(max_examples):
        im_lst = []
        for j in range(2):
          _min = 0.8
          _max = 1
          _leng = (_max - _min)*np.random.rand() + _min
          _min = 10
          _max = 20  
          _w = np.int((_max - _min)*np.random.rand() + _min)
          im_lst.append(get_circle_img(leng = _leng, w=_w))
        for j in range(2):
          _min = 0.8
          _max = 1
          _leng = (_max - _min)*np.random.rand() + _min 
          _min = 10
          _max = 20  
          _w = np.int((_max - _min)*np.random.rand() + _min)
          _min = 0
          _max = 360  
          _start = (_max - _min)/np.random.rand() + _min       
          im_lst.append(get_half_circle_img(leng = _leng, start = _start, w = _w))

        random.shuffle(im_lst)
        
        im = get_4_concat(im_lst[0],im_lst[1],im_lst[2],im_lst[3])
        if not os.path.exists(save_path + '/' + concept):
            os.mkdir(save_path + '/' + concept)
        im.save(save_path + '/' + concept + '/' + str(i) + '.jpg')
    
  elif concept == 'straight':
      for i in range(max_examples):
        im_lst = []
        for j in range(4):
          _min = 0.8
          _max = 1
          _leng = (_max - _min)*np.random.rand() + _min
          _min = 10
          _max = 20  
          _w = np.int((_max - _min)*np.random.rand() + _min)
          _min = -0.3
          _max = 0.3
          _angle = np.int((_max - _min)*np.random.rand() + _min)
          if j < 2:
              im_lst.append(get_h_line_img(leng = _leng, angle= _angle,w=_w))
          else:
              im_lst.append(get_v_line_img(leng = _leng, angle= _angle,w=_w))

        random.shuffle(im_lst)
        
        im = get_4_concat(im_lst[0],im_lst[1],im_lst[2],im_lst[3])
        if not os.path.exists(save_path + '/' + concept):
            os.mkdir(save_path + '/' + concept)
        im.save(save_path + '/' + concept + '/' + str(i) + '.jpg')      
  elif concept == 'sharpness':
      for i in range(max_examples):
        im_lst = []
        for j in range(4):
          _min = 0.2
          _max = 0.3
          _leng = (_max - _min)*np.random.rand() + _min
          _min = 15
          _max = 20  
          _w = np.int((_max - _min)*np.random.rand() + _min)
          _min = 0.3
          _max = 0.6
          _sharp = (_max - _min)*np.random.rand() + _min
          _min = 0
          _max = 360
          _rotate = (_max - _min)*np.random.rand() + _min
          im_lst.append(get_sharp_img(leng = _leng, sharp= _sharp,w=_w,rotate=_rotate))
          
        im = get_4_concat(im_lst[0],im_lst[1],im_lst[2],im_lst[3])
        if not os.path.exists(save_path + '/' + concept):
            os.mkdir(save_path + '/' + concept)
        im.save(save_path + '/' + concept + '/' + str(i) + '.jpg')         
    

['v_line','h_line','circle','half_circle','sharp']    
for concept in single_shape_concept:
  if concept == 'circle':
      for i in range(max_examples):
        _min = 0.8
        _max = 1
        _leng = (_max - _min)*np.random.rand() + _min
        _min = 10
        _max = 20  
        _w = np.int((_max - _min)*np.random.rand() + _min)
        im = get_circle_img(leng = _leng, w=_w)
        if not os.path.exists(save_path + '/' + concept):
            os.mkdir(save_path + '/' + concept)
        im.save(save_path + '/' + concept + '/' + str(i) + '.jpg')
        
  elif concept == 'half_circle':
      for i in range(max_examples):
        _min = 0.8
        _max = 1
        _leng = (_max - _min)*np.random.rand() + _min 
        _min = 10
        _max = 20  
        _w = np.int((_max - _min)*np.random.rand() + _min)
        _min = 0
        _max = 360  
        _start = (_max - _min)/np.random.rand() + _min       
        im = get_half_circle_img(leng = _leng, start = _start, w = _w)
        if not os.path.exists(save_path + '/' + concept):        
            os.mkdir(save_path + '/' + concept)
        im.save(save_path + '/' + concept + '/' + str(i) + '.jpg')

  elif concept == 'v_line':
      for i in range(max_examples):
        _min = 0.8
        _max = 1
        _leng = (_max - _min)*np.random.rand() + _min
        _min = 10
        _max = 20  
        _w = np.int((_max - _min)*np.random.rand() + _min)
        _min = -0.3
        _max = 0.3
        _angle = np.int((_max - _min)*np.random.rand() + _min)
        im = get_v_line_img(leng = _leng, angle= _angle,w=_w)
        if not os.path.exists(save_path + '/' + concept):
            os.mkdir(save_path + '/' + concept)
        im.save(save_path + '/' + concept + '/' + str(i) + '.jpg')

  elif concept == 'h_line':
      for i in range(max_examples):
        _min = 0.8
        _max = 1
        _leng = (_max - _min)*np.random.rand() + _min
        _min = 10
        _max = 20  
        _w = np.int((_max - _min)*np.random.rand() + _min)
        _min = -0.3
        _max = 0.3
        _angle = np.int((_max - _min)*np.random.rand() + _min)
        im = get_h_line_img(leng = _leng, angle= _angle,w=_w)
        if not os.path.exists(save_path + '/' + concept):
            os.mkdir(save_path + '/' + concept)
        im.save(save_path + '/' + concept + '/' + str(i) + '.jpg')
    
  
  elif concept == 'sharp':
      for i in range(max_examples):
        _min = 0.2
        _max = 0.3
        _leng = (_max - _min)*np.random.rand() + _min
        _min = 15
        _max = 20  
        _w = np.int((_max - _min)*np.random.rand() + _min)
        _min = 0.3
        _max = 0.6
        _sharp = (_max - _min)*np.random.rand() + _min
        _min = 0
        _max = 360
        _rotate = (_max - _min)*np.random.rand() + _min
        im = get_sharp_img(leng = _leng, sharp= _sharp,w=_w,rotate=_rotate)
  
        if not os.path.exists(save_path + '/' + concept):
            os.mkdir(save_path + '/' + concept)
        im.save(save_path + '/' + concept + '/' + str(i) + '.jpg')


