import argparse
import os

import arrow
import cv2
import time
import caffe
from eyewitness.config import (IN_MEMORY, BBOX)
from eyewitness.image_id import ImageId
from eyewitness.image_utils import (ImageProducer, swap_channel_rgb_bgr, ImageHandler)
from eyewitness.result_handler.db_writer import BboxPeeweeDbWriter
from eyewitness.result_handler.line_detection_result_handler import LineAnnotationSender
from naive_detector import RefineDetDetectorWrapper
from peewee import SqliteDatabase
from PIL import Image


# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--save_fig', action='store_true')
parser.add_argument(
    '--db_path', type=str, default='::memory::',
    help='the path used to store detection result records'
)


class InMemoryImageProducer(ImageProducer):
    def __init__(self, video_path, interval_s):
        self.vid = cv2.VideoCapture(video_path)
        self.interval_s = interval_s
        if not self.vid.isOpened():
            raise IOError("Couldn't open webcam or video")

    def produce_method(self):
        return IN_MEMORY

    def produce_image(self):
        while True:
            # clean buffer hack: for Linux V4L capture backend with a internal fifo
            for iter_ in range(5):
                self.vid.grab()
            _, frame = self.vid.read()
            yield Image.fromarray(swap_channel_rgb_bgr(frame))
            time.sleep(self.interval_s)


def image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image"""
    site_domain = os.environ.get('site_domain')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, drawn_image_path)


def line_detection_result_filter(detection_result):
    """
    used to check if sent notification or not
    """
    return any(i.label == 'person' for i in detection_result.detected_objects)


if __name__ == '__main__':
    args = parser.parse_args()
    # image producer from webcam
    image_producer = InMemoryImageProducer(0, interval_s=args.interval_s)

    # gpu preparation
    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    params = {
        'labelmap_file': 'data/VOC0712/labelmap_voc.prototxt',
        'model_path': 'models/VGGNet/VOC0712/refinedet_vgg16_320x320/',
        'model_name': 'VOC0712_refinedet_vgg16_320x320_final.caffemodel',
        'image_size': 320,
    }

    # object detector
    object_detector = RefineDetDetectorWrapper(params, threshold=0.6)

    # detection result handlers
    result_handlers = []

    # update image_info drawn_image_path, insert detection result
    database = SqliteDatabase(args.db_path)
    bbox_sqlite_handler = BboxPeeweeDbWriter(database)
    result_handlers.append(bbox_sqlite_handler)

    # setup your line channel token and audience
    channel_access_token = os.environ.get('CHANNEL_ACCESS_TOKEN')
    if channel_access_token:
        line_annotation_sender = LineAnnotationSender(
            channel_access_token=channel_access_token,
            image_url_handler=image_url_handler,
            detection_result_filter=line_detection_result_filter,
            detection_method=BBOX,
            update_audience_period=10,
            database=database)
        result_handlers.append(line_annotation_sender)

    for image in image_producer.produce_image():
        image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
        bbox_sqlite_handler.register_image(image_id, {})
        detection_result = object_detector.detect(image, image_id)

        # draw and save image, update detection result
        drawn_image_path = "detected_image/%s_%s.%s" % (
            image_id.channel, image_id.timestamp, image_id.file_format)
        ImageHandler.draw_bbox(image, detection_result.detected_objects)
        ImageHandler.save(image, drawn_image_path)
        detection_result.image_dict['drawn_image_path'] = drawn_image_path

        for result_handler in result_handlers:
            result_handler.handle(detection_result)
