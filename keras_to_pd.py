# from keras import backend as Kでも動作する
from tensorflow.keras import backend as K
import tensorflow as tf
import os
import pathlib


IMG_HEIGHT = 299
IMG_WIDTH = 299

# モデルの名前(保存名)
output_path = 'tcav/frozen_models/inceptionv3.pb'

# requirement : tensorflow 1.15
assert tf.__version__ == '1.15.2', 'Tensorflow version Error. You need 1.15.2 version'
assert output_path[-3:] == '.pb', 'Extension: output file extention is .pb'

if os.path.exists(output_path) == False:
  os.system('git clone -b r1.15 --single-branch https://github.com/tensorflow/tensorflow.git')
  
  # モデルをロード (Imagenetで事前学習済みのInceptionV3をロード)
  model = tf.keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=(IMG_HEIGHT,IMG_WIDTH,3), pooling=None, classes=1000)

  # outputのノード名が必要なのでprintして確認する
  print(model.output.op.name)
  output_node_name = model.output.op.name
  #ファイル名を.ckptとしてモデルを保存
  saver = tf.train.Saver()
  saver.save(K.get_session(), 'frozen_model.ckpt')

  # convert frozen model from .ckpt to .pb
  os.system('python tensorflow/tensorflow/python/tools/freeze_graph.py --input_meta_graph=frozen_model.ckpt.meta --input_checkpoint=frozen_model.ckpt --output_graph={0}  --output_node_names={1} --input_binary=true'.format(output_path,output_node_name))

  os.system('rm -rf frozen_model.ckpt*')
  os.system('rm checkpoint')


# .pbファイルをロードしてテキストファイルにネットワークの名前を書き込む
def writeTensorsName(pb_file, txt_file):
    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    path = pathlib.Path(txt_file)
    with path.open(mode='w') as f:
        # for op in graph.get_operations():
        #     f.write(op.name+'\n')

       for op in graph.get_operations():
           for t in op.values():
                f.write(t.name+'\n')

output_name_path = output_path[:-3] + '_name'


if os.path.exists(output_name_path) == False:
  writeTensorsName(output_path,output_name_path)