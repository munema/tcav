"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
""" Downloads models and datasets for imagenet

    Content downloaded:
        - Imagenet images for the zebra class.
        - Full Broden dataset(http://netdissect.csail.mit.edu/)
        - Inception 5h model(https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception5h.py)
        - Mobilenet V2 model(https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

    Functionality:
        - Downloads open source models(Inception and Mobilenet)
        - Downloads the zebra class from imagenet, to illustrate a target class
        - Extracts three concepts from the Broden dataset(striped, dotted, zigzagged)
        - Structures the data in a format that can be readily used by TCAV
        - Creates random folders with examples from Imagenet. Those are used by TCAV.

    Example usage:

    python download_and_make_datasets.py --source_dir=YOUR_FOLDER --number_of_images_per_folder=50 --number_of_random_folders=10
"""
import subprocess
import os
import argparse
from tensorflow.io import gfile
import imagenet_and_broden_fetcher as fetcher
import tensorflow as tf
import logging

def make_concepts_targets_and_randoms(source_dir, number_of_images_per_folder, number_of_random_folders):
    
    logging.basicConfig(filename=source_dir+'/logger.log', level=logging.INFO)
    
    # Run script to download data to source_dir
    if not gfile.exists(source_dir):
        gfile.makedirs(source_dir)
    if not gfile.exists(os.path.join(source_dir,'broden1_224/')) or not gfile.exists(os.path.join(source_dir,'inception5h')):
        subprocess.call(['bash' , 'FetchDataAndModels.sh', source_dir])
        
        
    # make targets from imagenet
    imagenet_dataframe = fetcher.make_imagenet_dataframe("/home/tomohiro/code/tcav/tcav/tcav_examples/image_models/imagenet/imagenet_url_map.csv")
    all_class = imagenet_dataframe["class_name"].values.tolist()

    # Determine classes that we will fetch
    imagenet_classes = ['fire engine']
    broden_concepts = ['striped', 'dotted', 'zigzagged']
    random_except_concepts = ['zebra','fire engine']
    except_words = ['cat', 'shark', 'apron', 'dogsled','dumbbell','ball','bus']
    for e_word in except_words:
        random_except_concepts.extend([element for element in all_class if e_word == str(element)[-len(e_word):]])
    
    tf.logging.info('imagenet_classe %s' % imagenet_classes)
    tf.logging.info('concepts %s' % broden_concepts)
    tf.logging.info('random_except_concepts %s' % random_except_concepts)

    for image in imagenet_classes:
        fetcher.fetch_imagenet_class(source_dir, image, number_of_images_per_folder, imagenet_dataframe)
    # Make concepts from broden
    for concept in broden_concepts:
        fetcher.download_texture_to_working_folder(broden_path=os.path.join(source_dir, 'broden1_224'),
                                                   saving_path=source_dir,
                                                   texture_name=concept,
                                                   number_of_images=number_of_images_per_folder)

    # Make random folders. If we want to run N random experiments with tcav, we need N+1 folders.
    # (変更) 除外するクラスを指定
    fetcher.generate_random_folders(
        working_directory=source_dir,
        random_folder_prefix="random500",
        number_of_random_folders=number_of_random_folders+1,
        number_of_examples_per_folder=number_of_images_per_folder,
        imagenet_dataframe=imagenet_dataframe,
        random_except_concepts = random_except_concepts
    )

def make_targets(source_dir, number_of_images_per_folder):
    
    # make targets from imagenet
    imagenet_dataframe = fetcher.make_imagenet_dataframe("/home/tomohiro/code/tcav/tcav/tcav_examples/image_models/imagenet/imagenet_url_map.csv")
    all_class = imagenet_dataframe["class_name"].values.tolist()

    # Determine classes that we will fetch
    imagenet_classes = ['soccer ball']
    
    for image in imagenet_classes:
        fetcher.fetch_imagenet_class(source_dir, image, number_of_images_per_folder, imagenet_dataframe)


def make_randoms(source_dir, number_of_images_per_folder, number_of_random_folders):
    
    logging.basicConfig(filename=source_dir+'/logger.log', level=logging.INFO)
    
    # Run script to download data to source_dir
    if not gfile.exists(source_dir):
        gfile.makedirs(source_dir)
    if not gfile.exists(os.path.join(source_dir,'broden1_224/')) or not gfile.exists(os.path.join(source_dir,'inception5h')):
        subprocess.call(['bash' , 'FetchDataAndModels.sh', source_dir])
        
        
    # make targets from imagenet
    imagenet_dataframe = fetcher.make_imagenet_dataframe("/home/tomohiro/code/tcav/tcav/tcav_examples/image_models/imagenet/imagenet_url_map.csv")
    all_class = imagenet_dataframe["class_name"].values.tolist()

    # Determine classes that we will fetch
    imagenet_classes = ['sorrel']
    broden_concepts = ['striped', 'dotted', 'zigzagged']
    random_except_concepts = ['zebra','fire engine']
    except_words = ['cat', 'shark', 'apron', 'dogsled','dumbbell','ball','bus']
    for e_word in except_words:
        random_except_concepts.extend([element for element in all_class if e_word == str(element)[-len(e_word):]])


    # Make random folders. If we want to run N random experiments with tcav, we need N+1 folders.
    # (変更) 除外するクラスを指定
    fetcher.generate_random_folders(
        working_directory=source_dir,
        random_folder_prefix="random500",
        number_of_random_folders=number_of_random_folders+1,
        number_of_examples_per_folder=number_of_images_per_folder,
        imagenet_dataframe=imagenet_dataframe,
        random_except_concepts = random_except_concepts
    )


if __name__ == '__main__':
    tf.get_logger().setLevel('INFO')
    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument('--source_dir', type=str,
                        help='Name for the directory where we will create the data.')
    parser.add_argument('--number_of_images_per_folder', type=int,
                        help='Number of images to be included in each folder')
    parser.add_argument('--number_of_random_folders', type=int,
                        help='Number of folders with random examples that we will generate for tcav')

    args = parser.parse_args()
    # create folder if it doesn't exist
    if not gfile.exists(args.source_dir):
        gfile.makedirs(os.path.join(args.source_dir))
        print("Created source directory at " + args.source_dir)
    # # Make data
    # make_concepts_targets_and_randoms(args.source_dir, args.number_of_images_per_folder, args.number_of_random_folders)
    # print("Successfully created data at " + args.source_dir)

    # Make random
    # make_randoms(args.source_dir, args.number_of_images_per_folder, args.number_of_random_folders)
    # print("Successfully created data at " + args.source_dir)

    # # Make target data
    make_targets(args.source_dir, args.number_of_images_per_folder)
    print("Successfully created data at " + args.source_dir)