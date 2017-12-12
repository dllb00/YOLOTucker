from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

proto_input = '/opt/caffe/YOLOTucker/tiny-yolo-voc.prototext'
weight_input = '/opt/caffe/YOLOTucker/tiny-yolo-voc.caffemodel'

proto_output = '/opt/caffe/YOLOTucker/tiny-yolo-voc-lra.prototext'
weight_output = '/opt/caffe/YOLOTucker/tiny-yolo-voc-lra.caffemodel'

from proto_lra import proto_lra
from weight_lra import weight_lra

lra_map = {'conv1': (3,32), 'conv2': (16,32), 'conv3': (32, 64)} #, 'ip1': 64, 'ip2': 10
#lra_map = {'conv1_scale': (3,16),'conv2_scale': (16,32),'conv3_scale': (32,64),'conv4_scale': (64,128),'conv5_scale': (128,256) ,'conv6_scale': (256,512),'conv7_scale': (512,1024),'conv8_scale': (1024,1024),'conv9': (1024,125)}
#lra_map = {'conv1': (3,16),'conv2': (16,32),'conv3': (32,64),'conv4': (64,128),'conv5': (128,256) ,'conv6': (256,512),'conv7': (512,1024),'conv8': (1024,1024),'conv9': (1024,125)}

if __name__ == "__main__":
        proto_lra(proto_input, proto_output, lra_map)
        weight_lra(proto_input, weight_input, proto_output, weight_output, lra_map)