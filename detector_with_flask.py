import argparse
import os

import caffe
from eyewitness.config import BBOX
from eyewitness.flask_server import BboxObjectDetectionFlaskWrapper
from eyewitness.result_handler.db_writer import BboxPeeweeDbWriter
from eyewitness.result_handler.line_detection_result_handler import LineAnnotationSender
from naive_detector import RefineDetDetectorWrapper
from peewee import SqliteDatabase


# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument(
    '--db_path', type=str, default='::memory::',
    help='the path used to store detection result records'
)
parser.add_argument(
    '--interval_s', type=int, default=3, help='the interval of image generation'
)
parser.add_argument(
    '--detector_host', type=str, default='localhost', help='the ip address of detector'
)
parser.add_argument(
    '--detector_port', type=int, default=5566, help='the port of detector port'
)
parser.add_argument(
    '--drawn_image_dir', type=str, default=None,
    help='the path used to store drawn images'
)


def image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image"""
    site_domain = os.environ.get('site_domain')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, drawn_image_path)


def line_detection_result_filter(detection_result, threshold=0.6):
    """
    used to check if sent notification or not
    """
    return any(i.label == 'person' and i.score > threshold
               for i in detection_result.detected_objects)


if __name__ == '__main__':
    args = parser.parse_args()
    detection_threshold = 0.7

    # initialize object_detector
    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    params = {
        'labelmap_file': 'data/VOC0712/labelmap_voc.prototxt',
        'model_path': 'models/VGGNet/VOC0712/refinedet_vgg16_320x320/',
        'model_name': 'VOC0712_refinedet_vgg16_320x320_final.caffemodel',
        'image_size': 320,
    }
    object_detector = RefineDetDetectorWrapper(params, threshold=detection_threshold)

    # detection result handlers
    result_handlers = []

    # update image_info drawn_image_path, insert detection result
    database = SqliteDatabase(args.db_path)
    bbox_sqlite_handler = BboxPeeweeDbWriter(database)
    result_handlers.append(bbox_sqlite_handler)

    # setup your line channel token and audience
    channel_access_token = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
    if channel_access_token:
        line_annotation_sender = LineAnnotationSender(
            channel_access_token=channel_access_token,
            image_url_handler=image_url_handler,
            detection_result_filter=line_detection_result_filter,
            detection_method=BBOX,
            update_audience_period=10,
            database=database)
        result_handlers.append(line_annotation_sender)

    flask_wrapper = BboxObjectDetectionFlaskWrapper(
        object_detector, bbox_sqlite_handler, result_handlers,
        database=database, drawn_image_dir=args.drawn_image_dir,
        detection_threshold=detection_threshold, collect_feedback_period=172800)

    params = {'host': args.detector_host, 'port': args.detector_port, 'threaded': False}
    flask_wrapper.app.run(**params)
