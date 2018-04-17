
'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from random import sample
from scipy.misc import imread
from keras.layers.normalization import BatchNormalization
import numpy as np
import os
import glob
from tqdm import tqdm
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def square(x):
    return x**2


get_custom_objects().update({'square': Activation(square)})
batch_size = 10
num_classes = 2
epochs = 100
num_steps = 10000 // batch_size
data_augmentation = False
#num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_gilbert_trained_model.h5'
IMAGE_DIR = '/media/data_cifs/cluster_projects/objectContourDetector/data/GILBERT/'
IMAGE_ROOT = os.path.join(IMAGE_DIR,
                            'all_images')
TRAIN_IMAGE_FILE = os.path.join(IMAGE_DIR,
                                  'train.txt')
TEST_IMAGE_FILE = os.path.join(IMAGE_DIR,
                                 'test.txt')
TRAIN_IMAGES = []
TEST_IMAGES = []
TRAIN_EXT = 'png'
TEST_EXT = 'png'

def load_split_images():
    ''' Load train and test images to a global list'''
    with open(TRAIN_IMAGE_FILE,'r') as f:
        for i, img in enumerate(f.readlines()):
            TRAIN_IMAGES.append('%s.%s'%(img.strip('\n'),
                                          TRAIN_EXT))
    print "%s train images loaded"%(i)

    with open(TEST_IMAGE_FILE) as f:
        for i, img in enumerate(f.readlines()):
            TEST_IMAGES.append('%s.%s'%(img.strip('\n'),
                                          TEST_EXT))
    print "%s test images loaded"%(i)

def preprocess_image(img):
    '''Perform all image preprocessing here.
    Currently just scaling to 0,255'''
    min_px, max_px = 0,255
    img = (img - min_px)/(max_px - min_px)
    return img

def get_batch_images_labels(split='train'):
    '''Load a batch of images from "split"'''
    if split == 'train':
        batch_fn = sample(TRAIN_IMAGES, batch_size)
    else:
        batch_fn = sample(TEST_IMAGES, batch_size)
    labels = [[1,0] if 'neg' in img else [0,1]
                        for img in batch_fn]
    labels = np.array(labels)
    imgs = [imread('%s/%s'%(IMAGE_ROOT,i))
                for i in batch_fn]
    imgs = [preprocess_image(i) for i in imgs]
    imgs = np.array(imgs)
    imgs = np.expand_dims(imgs, axis=3)
    return imgs, labels

def test_image_read():
    '''Test if image i/o works well'''
    batch = get_batch_images()
    import ipdb; ipdb.set_trace()

def get_Alexnet():
    alnet_weights = np.load('Alexnet_cc.npy')
    alnet_biases = np.zeros((32,))
    return alnet_weights, alnet_biases

def load_simple_convnet():
    model = Sequential()
    #Conv1 definition
    model.add(Conv2D(32, (11,11),
                input_shape=[256,256,3]))
    conv1 = model.layers[0]
    w,b = get_Alexnet()
    conv1.set_weights((w,b))
    model.add(Activation(square))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Conv2 definition
    model.add(Conv2D(12, (10,10)))
    model.add(Activation('relu'))
    #Readout conv definition
    model.add(Conv2D(2, (1,1)))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

def get_optimizer():
    opt = keras.optimizers.Nadam(lr=1e-3)
    return opt

def compile_model(model, opt):
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
    return model

def train_model(model):
    n_iters = epochs*len(TRAIN_IMAGES)/batch_size
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        'data/GILBERT/curvy_continuity_1.0/train',
        batch_size=batch_size,
        class_mode='binary'
    )
    model.fit_generator(
        train_generator,
        steps_per_epoch=num_steps,
        epochs=epochs
    )

def main():
    load_split_images()
    model = load_simple_convnet()
    opt = get_optimizer()
    model = compile_model(model, opt)
    train_model(model)
    import ipdb; ipdb.set_trace()

if __name__=='__main__':
    main()
