#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Dense, Dropout, Flatten

model = tf.keras.applications.vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=20)
#%%
model.summary()

# %%
