#opyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CIFAR100 small images classification dataset."""

import os
from tqdm import trange
import scipy.misc as sc
import numpy as np
from PIL import Image
from keras import backend
from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.datasets.cifar100.load_data')
def load_data(label_mode='fine'):
  
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    
    path = get_file(
      dirname,
      origin=origin,
      untar=True,
      file_hash=
      '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7')

    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if backend.image_data_format() == 'channels_last':  
    	x_train = x_train.transpose(0, 2, 3, 1)
    	x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)




def make_and_save(data, label, folder):
    num = 10000
    os.mkdir(folder)
    classes = np.unique(label)
    for i in classes:
        if  not os.path.isdir(folder+ '/' + str(i) ):
            os.makedirs(folder+ '/' + str(i) )
    count = data.shape[0]
    for i in trange(count):
        img = data[i:i+1,:,:,:].squeeze(axis = 0)
#         print(img)
        cls = label[i][0]
#         im = Image.fromarray(((img).astype('float32')))
        saveas = folder + '/' + str(cls) + '/' + str(num+i)+'_'+str(cls)+ '.png'
        sc.imsave(saveas, img) 
#         raise SystemError




(x_train, y_train), (x_test, y_test) = load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)
                
make_and_save(x_test, y_test, 'test')
