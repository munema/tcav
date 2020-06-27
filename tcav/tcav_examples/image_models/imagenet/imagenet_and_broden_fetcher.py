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

"""
Used to download images from the imagenet dataset and to move concepts from the Broden dataset, rearranging them
in a format that is TCAV readable. Also enables creation of random folders from imagenet

Usage for Imagenet
  imagenet_dataframe = pandas.read_csv("imagenet_url_map.csv")
  fetch_imagenet_class(path="your_path", class_name="zebra", number_of_images=100,
                      imagenet_dataframe=imagenet_dataframe)
                    
Usage for broden:
First, make sure you downloaded and unzipped the broden_224 dataset to a location of your interest. Then, run:
  download_texture_to_working_folder(broden_path="path_were_you_saved_broden", 
                                      saving_path="your_path",
                                      texture_name="striped",
                                       number_of_images=100)
                                      
Usage for making random folders:
  imagenet_dataframe = pandas.read_csv("imagenet_url_map.csv")
  generate_random_folders(working_directory="your_path",
                            random_folder_prefix="random_500",
                            number_of_random_folders=11,
                            number_of_examples_per_folder=100,
                            imagenet_dataframe=imagenet_dataframe)

"""
import pandas as pd
import urllib.request
import os
import shutil
import PIL
from PIL import Image
import tensorflow as tf
import socket
import random
import string
import numpy as np

kImagenetBaseUrl = "http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid="
kBrodenTexturesPath = "broden1_224/images/dtd/"
kMinFileSize = 10000

def randomname(n):
   return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

####### Helper functions
def change_near_color_img(img_path, min_sv = 0.6):
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

    color_distance = {'blue' : np.abs(color_dct['blue']['angle'] - np.mean(h)),
                      'yellow' : np.abs(color_dct['yellow']['angle'] - np.mean(h)),
                      'red' : np.abs(color_dct['red']['angle'] - np.mean(h)),
                      'green' : np.abs(color_dct['green']['angle'] - np.mean(h)),
                      }

    color = min(color_distance, key=color_distance.get)
    if color_distance[color] >= color_dct[color]['range']:
      print('not match',np.mean(h))
      return None
    print(color,color_dct[color]['angle'],np.mean(h))

    color_p = color_dct[color]['angle']
    lower = color_p - color_dct[color]['range']
    # fix lower because red color's lower is minus
    if lower < 0:
        lower += 360
    higher = color_p + color_dct[color]['range']

    color_rand = np.random.normal(color_p, color_dct[color]['range']/2)
    img_colored = Image.merge(
        "HSV",
        (
            h.point(lambda x: x if (x > lower and x < higher) else max(0,min(color_rand,359))),
            s.point(lambda x: max(x,int(p*255))),
            v.point(lambda x: max(x,int(p*255)))
        )
    ).convert("RGB")

    return color, img_colored


def change_color_img(img_path, color, min_sv = 0.6):
    img = Image.open(img_path)
    h, s, v = img.convert("HSV").split()
    p = min_sv
    color_dct = {}
    color_dct['blue'] = {}
    color_dct['blue']['angle'] = 225
    color_dct['blue']['range'] = 15
    color_dct['yellow'] = {}
    color_dct['yellow']['angle'] = 55
    color_dct['yellow']['range'] = 5
    color_dct['red'] = {}
    color_dct['red']['angle'] = 0
    color_dct['red']['range'] = 10
    color_dct['green'] = {}
    color_dct['green']['angle'] = 120
    color_dct['green']['range'] = 30
    color_dct['purple'] = {}
    color_dct['purple']['angle'] = 280
    color_dct['purple']['range'] = 10

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
    color_rand = np.random.normal(color_p, color_dct[color]['range']/2)
    if color_rand < 0:
      color_rand += 360
    elif color_rand > 360:
      color_rand -= 360

    # nomalization [0,360]→[0,255]
    color_rand *= 255/360
    higher *= 255/360
    lower *= 255/360

    img_colored = Image.merge(
        "HSV",
        (
            h.point(lambda x: x if (x > lower and x < higher) else color_rand),
            s.point(lambda x: max(x,int(p*255))),
            v.point(lambda x: max(x,int(p*255)))
        )
    ).convert("RGB")

    return img_colored

""" Makes a dataframe matching imagenet labels with their respective url.

Reads a csv file containing matches between imagenet synids and the url in
which we can fetch them. Appending the synid to kImagenetBaseUrl will fetch all
the URLs of images for a given imagenet label

  Args: path_to_imagenet_classes: String. Points to a csv file matching
          imagenet labels with synids.

  Returns: a pandas dataframe with keys {url: _ , class_name: _ , synid: _}
"""
def make_imagenet_dataframe(path_to_imagenet_classes):
  urls_dataframe = pd.read_csv(path_to_imagenet_classes)
  urls_dataframe["url"] = kImagenetBaseUrl + urls_dataframe["synid"]
  return urls_dataframe

""" Downloads an image.

Downloads and image from a image url provided and saves it under path.
Filters away images that are corrupted or smaller than 10KB

  Args:
    path: Path to the folder where we're saving this image.
    url: url to this image.

  Raises:
    Exception: Propagated from PIL.image.verify()
"""
# (変更)取得データ数(examples_selected)計算, ファイル名の重複を無くすためにランダムな名前にする(まだ読み込みないファイルが存在する...)
def download_image(path, url, examples_selected):
  image_name = url.split("/")[-1]
  image_name = image_name.split("?")[0]
  # image_prefix = image_name.split(".")[0]
  name = randomname(8)
  # saving_path = os.path.join(path, image_prefix +str(randint) + ".jpg")
  saving_path = os.path.join(path, name + ".jpg")
  urllib.request.urlretrieve(url, saving_path)
  try:
    # Throw an exception if the image is unreadable or corrupted
    Image.open(saving_path).verify()
    # Remove images smaller than 10kb, to make sure we are not downloading empty/low quality images
    if tf.io.gfile.stat(saving_path).length < kMinFileSize:
      tf.io.gfile.remove(saving_path)
    # (変更)全て成功したときに+1
    else:
      examples_selected += 1
      print(examples_selected)
  # PIL.Image.verify() throws a default exception if it finds a corrupted image.
  except Exception as e:
    tf.io.gfile.remove(
        saving_path
    )  # We need to delete it, since urllib automatically saves them.
    raise e
  return examples_selected


# (追加) color image取得
def download_near_color_image(pre_partition_name, color_lst, url, examples_selected):
  image_name = url.split("/")[-1]
  image_name = image_name.split("?")[0]
  name = randomname(8)
  saving_path = '/home/tomohiro/code/tcav/trash/tmp.jpg'
  urllib.request.urlretrieve(url, saving_path)
  try:
    Image.open(saving_path).verify()

    if tf.io.gfile.stat(saving_path).length < kMinFileSize:
      tf.io.gfile.remove(saving_path)
    # (変更)全て成功したときに+1
    else:
      color, img = change_near_color_img(saving_path)
      examples_selected[color] += 1
      img.save(os.path.join(pre_partition_name + color, name + ".jpg"))
      tf.io.gfile.remove(
        saving_path
      )
  # PIL.Image.verify() throws a default exception if it finds a corrupted image.
  except Exception as e:
    tf.io.gfile.remove(
        saving_path
    )  # We need to delete it, since urllib automatically saves them.
    raise e
  return color, examples_selected

def download_color_image(partition_name, color, url, examples_selected, original_save = False):
  image_name = url.split("/")[-1]
  image_name = image_name.split("?")[0]
  name = randomname(8)
  saving_path = '/home/tomohiro/code/tcav/trash/0.jpg'
  urllib.request.urlretrieve(url, saving_path)
  try:
    Image.open(saving_path).verify()

    if tf.io.gfile.stat(saving_path).length < kMinFileSize:
      tf.io.gfile.remove(saving_path)
    # (変更)全て成功したときに+1
    else:
      if original_save:
        if not os.path.exists('/home/tomohiro/code/tcav/tmp/imagenet_' + color):
          os.mkdir('/home/tomohiro/code/tcav/tmp/imagenet_' + color)
        origin_img_path = '/home/tomohiro/code/tcav/tmp/imagenet_' + color + '/' + name + '_origin.jpg'
        Image.open(saving_path).save(origin_img_path)
      img = change_color_img(saving_path, color)
      examples_selected += 1
      img.save(os.path.join(partition_name, name + ".jpg"))
      tf.io.gfile.remove(
        saving_path
      )
  # PIL.Image.verify() throws a default exception if it finds a corrupted image.
  except Exception as e:
    tf.io.gfile.remove(
        saving_path
    )  # We need to delete it, since urllib automatically saves them.
    raise e
  return color, examples_selected


""" For a imagenet label, fetches all URLs that contain this image, from the main URL contained in the dataframe

  Args:
    imagenet_dataframe: Pandas Dataframe containing the URLs for different
      imagenet classes.
    concept: A string representing Imagenet concept(i.e. "zebra").

  Returns:
    A list containing all urls for the imagenet label. For example
    ["abc.com/image.jpg", "abc.com/image2.jpg", ...]

  Raises:
    tf.errors.NotFoundError: Error occurred when we can't find the imagenet
    concept on the dataframe.
"""
def fetch_all_urls_for_concept(imagenet_dataframe, concept):
  if imagenet_dataframe["class_name"].str.contains(concept).any():
    all_images = imagenet_dataframe[imagenet_dataframe["class_name"] ==
                                    concept]["url"].values[0]
    bytes = urllib.request.urlopen(all_images)
    all_urls = []
    for line in bytes:
      all_urls.append(line.decode("utf-8"))
    return all_urls
  else:
    raise tf.errors.NotFoundError(
        None, None, "Couldn't find any imagenet concept for " + concept +
        ". Make sure you're getting a valid concept")


####### Main methods
""" For a given imagenet class, download images from the internet

  Args:
    path: String. Path where we're saving the data. Creates a new folder with
      path/class_name.
    class_name: String representing the name of the imagenet class.
    number_of_images: Integer representing number of images we're getting for
      this example.
    imagenet_dataframe: Dataframe containing the URLs for different imagenet
      classes.

  Raises:
    tf.errors.NotFoundError: Raised when imagenet_dataframe is not provided


"""
def fetch_imagenet_class(path, class_name, number_of_images, imagenet_dataframe):
  if imagenet_dataframe is None:
    raise tf.errors.NotFoundError(
        None, None,
        "Please provide a dataframe containing the imagenet classes. Easiest way to do this is by calling make_imagenet_dataframe()"
    )
  # To speed up imagenet download, we timeout image downloads at 5 seconds.
  socket.setdefaulttimeout(5)

  tf.logging.info("Fetching imagenet data for " + class_name)
  concept_path = os.path.join(path, class_name)
  if os.path.exists(concept_path):
    num_downloaded = len(os.listdir(concept_path))
    print(concept_path)
  else:
    tf.io.gfile.makedirs(concept_path)
    num_downloaded = 0
  tf.logging.info("Saving images at {}, now length {}".format(concept_path,num_downloaded))

  # Check to see if this class name exists. Fetch all urls if so.
  all_images = fetch_all_urls_for_concept(imagenet_dataframe, class_name)
  for image_url in all_images:
    if "flickr" not in image_url:
      try:
        num_downloaded = download_image(concept_path, image_url, num_downloaded)
        # num_downloaded += 1

      except Exception as e:
        tf.logging.info("Problem downloading imagenet image. Exception was " +
                        str(e) + " for URL " + image_url)
    if num_downloaded >= number_of_images:
      break

  # If we reached the end, notify the user through the console.
  if num_downloaded < number_of_images:
    print("You requested " + str(number_of_images) +
          " but we were only able to find " +
          str(num_downloaded) +
          " good images from imageNet for concept " + class_name)
  else:
    print("Downloaded " + str(number_of_images) + " for " + class_name)


"""Moves all textures in a downloaded Broden to our working folder.

Assumes that you manually downloaded the broden dataset to broden_path.


  Args:
  broden_path: String.Path where you donwloaded broden.
  saving_path: String.Where we'll save the images. Saves under
    path/texture_name.
  texture_name: String representing DTD texture name i.e striped
  number_of_images: Integer.Number of images to move
"""
def download_texture_to_working_folder(broden_path, saving_path, texture_name,
                                       number_of_images):
  # Create new experiment folder where we're moving the data to
  texture_saving_path = os.path.join(saving_path, texture_name)
  tf.io.gfile.makedirs(texture_saving_path)

  # Get images from broden
  broden_textures_path = os.path.join(broden_path, kBrodenTexturesPath)
  tf.logging.info("Using path " + str(broden_textures_path) + " for texture: " +
                  str(texture_name))
  for root, dirs, files in os.walk(broden_textures_path):
    # Broden contains _color suffixed images. Those shouldn't be used by tcav.
    texture_files = [
        a for a in files if (a.startswith(texture_name) and "color" not in a)
    ]
    number_of_files_for_concept = len(texture_files)
    tf.logging.info("We have " + str(len(texture_files)) +
                    " images for the concept " + texture_name)

    # Make sure we can fetch as many as the user requested.
    if number_of_images > number_of_files_for_concept:
      raise Exception("Concept " + texture_name + " only contains " +
                      str(number_of_files_for_concept) +
                      " images. You requested " + str(number_of_images))

    # We are only moving data we are guaranteed to have, so no risk for infinite loop here.
    save_number = number_of_images
    while save_number > 0:
      for file in texture_files:
        path_file = os.path.join(root, file)
        texture_saving_path_file = os.path.join(texture_saving_path, file)
        tf.io.gfile.copy(
            path_file, texture_saving_path_file,
            overwrite=True)  # change you destination dir
        save_number -= 1
        # Break if we saved all images
        if save_number <= 0:
          break


""" Creates folders with random examples under working directory.

They will be named with random_folder_prefix as a prefix followed by the number
of the folder. For example, if we have:

    random_folder_prefix = random500
    number_of_random_folders = 3

This function will create 3 folders, all with number_of_examples_per_folder
images on them, like this:
    random500_0
    random500_1
    random500_2


  Args:
    random_folder_prefix: String.The prefix for your folders. For example,
      random500_1, random500_2, ... , random500_n.
    number_of_random_folders: Integer. Number of random folders.
    number of examples_per_folder: Integer. Number of images that will be saved
      per folder.
    imagenet_dataframe: Pandas Dataframe containing the URLs for different
      imagenet classes.
"""
def generate_random_folders(working_directory, random_folder_prefix,
                            number_of_random_folders,
                            number_of_examples_per_folder, imagenet_dataframe):
  socket.setdefaulttimeout(3)
  imagenet_concepts = imagenet_dataframe["class_name"].values.tolist()
  for partition_number in range(number_of_random_folders):
    partition_name = random_folder_prefix + "_" + str(partition_number)
    partition_folder_path = os.path.join(working_directory, partition_name)
    if os.path.exists(partition_folder_path):
      examples_selected = len(os.listdir(partition_folder_path))
    else:
      tf.io.gfile.makedirs(partition_folder_path)
      examples_selected = 0

    # Select a random concept
    tf.logging.info('{} number of examples {}'.format(partition_name,examples_selected))
    while examples_selected < number_of_examples_per_folder:
      random_concept = random.choice(imagenet_concepts)
      try:
        urls = fetch_all_urls_for_concept(imagenet_dataframe, random_concept)
      except:
        continue
      for url in urls:
        # We are filtering out images from Flickr urls, since several of those were removed
        if "flickr" not in url:
          try:
            examples_selected = download_image(partition_folder_path, url, examples_selected)
            # examples_selected += 1
            if (examples_selected) % 10 == 0:
              tf.logging.info("Downloaded " + str(examples_selected) + "/" +
                              str(number_of_examples_per_folder) + " for " +
                              partition_name)
            break  # Break if we successfully downloaded an image
          except:
              pass # try new url

# (追加) imagenetから色ベースで画像を収集 (hsv変換)
def fetch_imagenet_class_color(working_directory, number_of_examples_per_folder, imagenet_dataframe, folder_prefix, color_lst):
  if imagenet_dataframe is None:
    raise tf.errors.NotFoundError(
        None, None,
        "Please provide a dataframe containing the imagenet classes. Easiest way to do this is by calling make_imagenet_dataframe()"
    )
  # To speed up imagenet download, we timeout image downloads at 5 seconds.
  socket.setdefaulttimeout(3)
  imagenet_concepts = imagenet_dataframe["class_name"].values.tolist()

  examples_selected = {}
  for partition in color_lst:
    partition_name = folder_prefix + "-" + partition
    partition_folder_path = os.path.join(working_directory, partition_name)
    if os.path.exists(partition_folder_path):
      examples_selected[partition] = len(os.listdir(partition_folder_path))
    else:
      tf.io.gfile.makedirs(partition_folder_path)
      examples_selected[partition] = 0

  pre_partition_name = str(os.path.join(working_directory, folder_prefix + "-"))

  while min(examples_selected.values()) < number_of_examples_per_folder:
    random_concept = random.choice(imagenet_concepts)
    urls = fetch_all_urls_for_concept(imagenet_dataframe, random_concept)
    for url in urls:
      # We are filtering out images from Flickr urls, since several of those were removed
      if "flickr" not in url:
        try:
          img_color, examples_selected = download_near_color_image(pre_partition_name, color_lst, url, examples_selected)
          break  # Break if we successfully downloaded an image
        except:
            pass # try new url



# (追加) imagenetから色ベースで画像を収集 (色指定) (hsv変換)
def fetch_imagenet_class_color_fixed(working_directory, number_of_examples_per_folder, imagenet_dataframe, folder_prefix, color_lst):
  if imagenet_dataframe is None:
    raise tf.errors.NotFoundError(
        None, None,
        "Please provide a dataframe containing the imagenet classes. Easiest way to do this is by calling make_imagenet_dataframe()"
    )
  # To speed up imagenet download, we timeout image downloads at 5 seconds.
  socket.setdefaulttimeout(3)
  imagenet_concepts = imagenet_dataframe["class_name"].values.tolist()

  for partition in color_lst:
    print(partition)
    partition_name = folder_prefix + "-" + partition
    partition_folder_path = os.path.join(working_directory, partition_name)
    if os.path.exists(partition_folder_path):
      examples_selected = len(os.listdir(partition_folder_path))
    else:
      tf.io.gfile.makedirs(partition_folder_path)
      examples_selected = 0

    while examples_selected < number_of_examples_per_folder:
      random_concept = random.choice(imagenet_concepts)
      urls = fetch_all_urls_for_concept(imagenet_dataframe, random_concept)
      for url in urls:
        # We are filtering out images from Flickr urls, since several of those were removed
        if "flickr" not in url:
          try:
            print(examples_selected)
            img_color, examples_selected = download_color_image(partition_folder_path, partition, url, examples_selected,True)
            break  # Break if we successfully downloaded an image
          except:
              pass # try new url