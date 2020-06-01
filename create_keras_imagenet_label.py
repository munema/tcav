import json
from config import root_dir
import pandas as pd

json_path = root_dir + 'tcav/frozen_models/keras_imagenet_class_index.json'
output_path = root_dir + 'tcav/frozen_models/keras_imagenet_label_string.txt'

df_s = pd.read_json(json_path)

with open(output_path, mode='w') as f:
  for name in df_s.T[1].values:
    f.write(name + '\n')
