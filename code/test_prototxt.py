import caffe
import sys
net = caffe.Net(sys.argv[1])
print 'Successfully initialized!'
