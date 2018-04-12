import _init_paths
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from utils.blob import im_list_to_blob

#Training script for cedn on gilbert stimuli

def getCaffeNet(prototxt, caffemodel=None):
    if caffemodel:
        net = caffe.Net(prototxt, caffemodel)
    else:
        net = caffe.Net(prototxt)
    return net

def getTransformer(net):
    transformer = caffe.io.Transformer({'data':
                                            net.blobs['data'].data.shape})
    transformer.set_mean('data',np.load(mean_path))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data',255.)
    return transformer

def loadImage(img_path):
    im = caffe.io.load_image(img_path)
    return im

def getSolver(solverPath):
    solver = caffe.AdamSolver(solverPath)
    return solver
