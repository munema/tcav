#%%
from PIL import Image, ImageFilter, ImageMath, ImageOps
import numpy as np
import cv2
#%%
img = Image.open('/home/tomohiro/code/tcav/tcav/dataset/for_tcav/ambulance/00h13oqC.jpg')
img = Image.open('/home/tomohiro/code/tcav/tcav/dataset/google-512/dataset/green/173.png')

#%%
# ボツ
h, s, v = img.convert("HSV").split()
p = 0.6
color_p = 240
color_p *= 255/360
lower = (color_p - 10)*255/360
higher = (color_p + 10)*255/360
img_bad_colored = Image.merge(
    "HSV",
    (
        h.point(lambda x: x if (x > lower and x < higher) else color_p),
        s.point(lambda x: max(x,p*255)),
        v.point(lambda x: max(x,p*255))
    )
).convert("RGB")
img_bad_colored
# img_bad_colored.save('/home/tomohiro/code/tcav/tmp/img_bad_colored_example.jpg')
# %%
img = Image.open('/home/tomohiro/code/tcav/tcav/dataset/for_tcav/apple_red/3_100.jpg')
#img = Image.open('/home/tomohiro/code/tcav/tcav/dataset/for_tcav/ambulance/00h13oqC.jpg')
#img = Image.open('/home/tomohiro/code/tcav/tcav/dataset/google-512/dataset/green/11.png')
gray = ImageOps.grayscale(img)
img_colored = ImageOps.colorize(gray, black=(255, 255, 0), white=(255, 255, 255))
img_colored
#img_colored.save('/home/tomohiro/code/tcav/tmp/img_colored_example.jpg')
# %%
# ボツ
# im = np.array(Image.open('/home/tomohiro/code/tcav/tcav/dataset/for_tcav/ambulance/00h13oqC.jpg'))

# im_R = im.copy()
# im_R[:, :, (1, 2)] = 0
# im_G = im.copy()
# im_G[:, :, (0, 2)] = 0
# im_B = im.copy()
# im_B[:, :, (0, 1)] = 0

# # 横に並べて結合（どれでもよい）
# im_RGB = np.concatenate((im_R, im_G, im_B), axis=1)
# # im_RGB = np.hstack((im_R, im_G, im_B))
# # im_RGB = np.c_['1', im_R, im_G, im_B]

# pil_img_RGB = Image.fromarray(im_RGB)
# pil_img_RGB

#%%
# img = Image.open('/home/tomohiro/code/tcav/tcav/dataset/for_tcav/ambulance/00h13oqC.jpg')
# h, s, v = img.convert("HSV").split()
# np.mean(np.array(h))
# img = Image.open('/home/tomohiro/code/tcav/tcav/dataset/for_tcav/ambulance/00h13oqC.jpg')
# img
# %%
int(0.8)

# %%
def change_color_img(img_path, color, min_sv = 0.6):
    img = Image.open(img_path)
    h, s, v = img.convert("HSV").split()
    p = min_sv
    color_dct = {}
    color_dct['blue'] = {}
    color_dct['blue']['angle'] = 240
    color_dct['blue']['range'] = 25
    color_dct['yellow'] = {}
    color_dct['yellow']['angle'] = 60
    color_dct['yellow']['range'] = 15
    color_dct['red'] = {}
    color_dct['red']['angle'] = 0
    color_dct['red']['range'] = 10
    color_dct['green'] = {}
    color_dct['green']['angle'] = 120
    color_dct['green']['range'] = 30

    color_name_lst = list(color_dct.keys())
    if color not in color_name_lst:
        color
        print(r'Color Selection ERROR : You can choose {color_name_lst}')
        sys.exit()

    color_p = color_dct[color]['angle']
    lower = color_p - color_dct[color]['range']
    # fix lower because red color's lower is minus
    if lower < 0:
        lower += 360
    higher = color_p + color_dct[color]['range']

    img_colored = Image.merge(
        "HSV",
        (
            h.point(lambda x: x if (x > lower and x < higher) else color_p),
            s.point(lambda x: max(x,int(p*255))),
            v.point(lambda x: max(x,int(p*255)))
        )
    ).convert("RGB")

    return img_colored


# %%
path = '/home/tomohiro/code/tcav/tcav/dataset/google-512/dataset/green/173.png'
change_color_img(path, 'red', min_sv = 0.5)

# %%
