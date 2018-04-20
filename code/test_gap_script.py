import caffe
import numpy as np
import lmdb

# env = lmdb.open('data/gilbert/train_lmdb', readonly=True)
# with env.begin() as txn:
#     raw_datum = txn.get(b'00000000')
#
# datum = caffe.proto.caffe_pb2.Datum()
# datum.ParseFromString(raw_datum)
# flat_x = np.fromstring(datum.data, dtype=np.uint8)
# x = flat_x.reshape(datum.channels, datum.height, datum.width)
# y = datum.label
net = caffe.Net('code/vgg-16-encoder-decoder-contour-gap.prototxt')
# net.blobs['data'].data[...] = x
out = net.forward()
print "Success!"
import ipdb; ipdb.set_trace()
