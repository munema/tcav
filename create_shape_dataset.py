from PIL import Image, ImageDraw
from config import root_dir

# 保留

save_path = root_dir + 'tcav/dataset/for_tcav/'

im = Image.new('RGB', (512, 512), (128, 128, 128))
draw = ImageDraw.Draw(im)

draw.line((0, im.height, im.width, 0), fill=(255, 0, 0), width=8)
draw.rectangle((100, 100, 200, 200), fill=(0, 255, 0))
draw.ellipse((250, 300, 450, 400), fill=(0, 0, 255))

# im.save('test.jpg', quality=95)
