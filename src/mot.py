import time
import os
import argparse
import numpy as np
from PIL import Image
import cv2 

from cfg.config import config
from detector.detect_image import Detector
from tracker.classify_image import FeatureExtractor
from tracker.tracker import MOTTracker
from utils import common, utils

class MOT():
    def __init__(self, detector_model, extractor_model, labels, threshold, top_k):
        self.detector = Detector(detector_model, labels)
        self.extractor = FeatureExtractor(extractor_model, labels)
        self.tracker = MOTTracker()
        
        # load labels
        self.labels = utils.load_labels(labels)

        self.threshold = threshold
        self.top_k = top_k

        # Average fps over last 30 frames.
        self.fps_counter = common.avg_fps_counter(30)

    def run_mot(self, cv2_im):
        # image information
        height, width, _ = cv2_im.shape

        start_time = time.monotonic()
        # Detect bounding boxes
        self.detector.set_input(cv2_im)
        objs = self.detector.get_detections()
        # run object detector
        bboxes = []
        for i in range(len(objs)):
            x0 = int(objs[i].bbox.xmin * width)
            y0 = int(objs[i].bbox.ymin * height)
            x1 = int(objs[i].bbox.xmax * width)
            y1 = int(objs[i].bbox.ymax * height)

            bboxes.append([x0, y0, x1, y1])
        
        #print("Number of bounding boxes : {}".format(len(detections)))

        # run feature extractor
        self.extractor.set_input(cv2_im)
        embeddings = self.extractor.get_features(bboxes)
        
        end_time = time.monotonic()
        # tracking
        detections = []
        for i in range(len(objs)):
            element = []  # np.array([])
            element.append(objs[i].bbox.xmin)
            element.append(objs[i].bbox.ymin)
            element.append(objs[i].bbox.xmax)
            element.append(objs[i].bbox.ymax)
            element.append(embeddings[i]) 
            detections.append(element)

        detections = np.array(detections)
        trkdata = []
        text_lines = ['']
        
        if detections.any():
            trkdata = self.tracker.update(detections)
            text_lines = [
                'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
                'FPS: {} fps'.format(round(next(self.fps_counter)))]
        
        if len(objs) != 0:
            # draw bounding box
            cv2_im = utils.append_objs_to_img(cv2_im, objs, self.labels, text_lines, trkdata)
        
        return cv2_im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_model', help='path to detector tflite model',
                        default=os.path.join(config.WEIGHT_DIR, 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'))
    parser.add_argument('--extractor_model', help='path to features extractor tflite model',
                        default=os.path.join(config.WEIGHT_DIR, 'osnet_x0_25_msmt17_quant_tpu.tflite'))
    parser.add_argument('--labels', help='load categories object',
                        default=os.path.join(config.WEIGHT_DIR, 'coco_labels.txt'))
    parser.add_argument('--threshold', type=float, help='classifier score threshold',
                        default=0.4)
    parser.add_argument('--top_k', type=int, help='number of categories with heighest score to display',
                        default=3)
    parser.add_argument('--input_data', required=True, help=
                        'URI to input stream\n'
                        '1) video file (e.g. /data/group3.mp4)\n'
                        '2) USB or V4L2 camera (e.g. /dev/video0)\n',
                        default='/dev/video1')

    args = parser.parse_args()

    mot = MOT(args.detector_model, args.extractor_model, args.labels, args.threshold, args.top_k)
    
    # load input video
    print('Starting Video...')
    cap = cv2.VideoCapture(args.input_data)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count == 1024:
           break 

        cv2_im = cv2.resize(frame, (640, 480))

        # run multiple objects tracking
        cv2_im = mot.run_mot(cv2_im)
        count += 1

        # show image
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()