import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils
import tcav.utils_plot as utils_plot # utils_plot requires matplotlib
import os
import tensorflow as tf
from config import root_dir, model_to_run, bottlenecks, concepts, version, num_random_exp, max_examples, run_parallel, num_workers, is_cav_on, make_random,true_cav
import sys

# target
target = str(sys.argv[1])

# function to create project name

def list_string(list):
    str = list[0]
    for s in list[1:]:
        str += '_' + s
    return str

def create_project_name(model, bottlenecks, target, concepts, version):
    name = model_to_run
    name += '_'+ list_string(bottlenecks)
    name += ':'+ target
    name += ':'+ list_string(concepts)
    name += '_'+ str(version)
    return name

print ('REMEMBER TO UPDATE YOUR_PATH (where images, models are)!')
# the name of the parent directory that results are stored (only if you want to cache)
project_name = create_project_name(model_to_run, bottlenecks, target, concepts, version)
save_folder = model_to_run
if make_random:
    save_folder += '/random' + str(max_examples)
working_dir = root_dir + 'log/'  + save_folder
# where activations are stored (only if your act_gen_wrapper does so)
activation_dir =  working_dir+ '/activations/'
# where CAVs are stored.
# You can say None if you don't wish to store any.
if is_cav_on:
    cav_dir = working_dir + '/cavs/'
    utils.make_dir_if_not_exists(cav_dir)
else:
    cav_dir = None
# where TCAVs are stored.
tcav_dir = working_dir + '/tcavs/'
# where the images live.
source_dir = root_dir + 'tcav/dataset/for_tcav/'

utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(tcav_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs.
alphas = [0.1]

print('TCAV dataset path is {}'.format(source_dir))
print('Results is saved at {}'.format(working_dir))

sess = utils.create_session()

#=================================================================================
# GRAPH_PATH = root_dir + "tcav/frozen_models/tensorflow_inception_graph.pb"
# GRAPH_PATH = root_dir + 'tcav/frozen_models/inceptionv3.pb'
# GRAPH_PATH = root_dir + 'tcav/frozen_models/fruit_4layers_cnn_2d.pb'
GRAPH_PATH = root_dir + 'tcav/frozen_models/mnist_3layers_cnn-2d.pb'

# LABEL_PATH = root_dir + "tcav/dataset/tensorflow_imagenet_label_strings.txt"
# LABEL_PATH = root_dir + "tcav/dataset/keras_imagenet_label_strings.txt"
# LABEL_PATH = root_dir + 'tcav/dataset/fruit25_label_string.txt'
LABEL_PATH = root_dir + 'tcav/dataset/colored_mnist'

# mymodel = model.GoogleNetWrapper_public(sess,GRAPH_PATH,LABEL_PATH)
# mymodel = model.KerasInceptionV3Wrapper_public(sess,GRAPH_PATH,LABEL_PATH)
# mymodel = model.KerasCNNWrapper_public(sess,GRAPH_PATH,LABEL_PATH)
mymodel = model.KerasMnistCnnWrapper_public(sess,GRAPH_PATH,LABEL_PATH)
#=================================================================================

act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=max_examples)

tf.logging.set_verbosity(tf.logging.INFO)
## only running num_random_exp = 10 to save some time. The paper number are reported for 500 random runs.
mytcav = tcav.TCAV(sess,
                   target,
                   concepts,
                   bottlenecks,
                   act_generator,
                   alphas,
                   cav_dir=cav_dir,
                   tcav_dir=tcav_dir,
                   num_random_exp=num_random_exp,
                   project_name=project_name,
                   make_random=make_random,
                   true_cav=true_cav)#10)
print ('This may take a while... Go get coffee!')
results = mytcav.run(run_parallel=run_parallel, num_workers=num_workers)
print ('done!')

# utils_plot.plot_results(results, num_random_exp=num_random_exp)