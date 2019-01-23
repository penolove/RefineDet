import argparse
import logging

import caffe
from eyewitness.dataset_util import BboxDataSet
from eyewitness.evaluation import BboxMAPEvaluator

from naive_detector import RefineDetDetectorWrapper


parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('--gpu_id', type=int, default=0)

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parser.parse_args()

    # initialize object_detector
    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    dataset_folder = 'VOC2007'
    dataset_VOC_2007 = BboxDataSet(dataset_folder, 'VOC2007')
    params = {
        'labelmap_file': 'data/VOC0712/labelmap_voc.prototxt',
        'model_path': 'models/VGGNet/VOC0712/refinedet_vgg16_320x320/',
        'model_name': 'VOC0712_refinedet_vgg16_320x320_final.caffemodel',
        'image_size': 320,
    }
    object_detector = RefineDetDetectorWrapper(params, threshold=0.6)
    bbox_map_evaluator = BboxMAPEvaluator(test_set_only=False)
    # which will lead to ~0.73
    print(bbox_map_evaluator.evaluate(object_detector, dataset_VOC_2007))
