import numpy as np
import caffe
import glob
import sys
from tqdm import tqdm

'''Script to load weights from caffemodel
   and store them in a numpy file '''

OUT_DIR = '.'
def get_net_params(prototxt, caffemodel):
    net = caffe.Net(prototxt, caffemodel)
    params = net.params
    weight_keys = params.keys()
    print "Successfully loaded model"
    return params, weight_keys

def get_drewstyle_weights(w, b, layer):
    conv_w_b = {layer:(w,b)}
    return conv_w_b


def store_conv_layerwise(params, weight_keys):
    for layer in tqdm(weight_keys):
        w, b = params.get(layer)
        w_np = w.data
        b_np = b.data
        conv_w_b = get_drewstyle_weights(w_np,
                                           b_np,
                                           layer)
        np.save('%s/VGG_PASCAL_%s.npy'%(
                OUT_DIR,
                layer),
                conv_w_b)

def main():
    prototxt, caffemodel = sys.argv[1], sys.argv[2]
    OUT_DIR = sys.argv[3]
    params, weight_keys = get_net_params(prototxt, caffemodel)
    store_conv_layerwise(params, weight_keys)

if __name__=="__main__":
    main()
