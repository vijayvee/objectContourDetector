import caffe
import numpy as np

zeros = np.zeros((8,3,256,256))
net = caffe.Net('code/vgg-16-encoder-decoder-contour-gap.prototxt')
net.blobs['data'].data[...] = zeros
out = net.forward()
print "Success!"
import ipdb; ipdb.set_trace()
