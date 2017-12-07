from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

proto_input = "~/YOLOTucker/tiniy-yolo-caffe/tiny_yolo_voc.prototxt"
weight_input = "~/YOLOTucker/tiniy-yolo-caffe/tiny_yolo_voc.caffemodel"

proto_output = "~/YOLOTucker/tiniy-yolo-caffe/tiny_yolo_voc_lra.prototxt"
weight_output = "~/YOLOTucker/tiniy-yolo-caffe/tiny_yolo_voc_lra.caffemodel"

from proto_lra import proto_lra
from weight_lra import weight_lra

lra_map = {'conv1': (3,32), 'conv2': (16,32), 'conv3': (32, 64)} #, 'ip1': 64, 'ip2': 10

if __name__ == "__main__":
        proto_lra(proto_input, proto_output, lra_map)
        weight_lra(proto_input, weight_input, proto_output, weight_output, lra_map)