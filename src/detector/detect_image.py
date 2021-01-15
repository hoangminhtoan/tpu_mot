import collections
import numpy as np
import cv2 
from PIL import Image 

from utils import common

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

class Detector():
    def __init__(self, detector_model, labels):
        super(Detector, self).__init__()

        self.labels = labels

        #load model
        print('Loading detector model ...')
        self.interpreter = common.make_interpreter(detector_model)
        self.interpreter.allocate_tensors()

    def set_input(self, cv2_im):
        self.pil_image = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))
        common.set_input(self.interpreter, self.pil_image)
        self.interpreter.invoke()

    def get_detections(self, score_threshold=0.4, top_k=3):
        """Returns list of detected objects."""
        boxes = common.output_tensor(self.interpreter, 0, 'boxes')
        class_ids = common.output_tensor(self.interpreter, 1, 'class_ids')
        scores = common.output_tensor(self.interpreter, 2, 'scores') # features

        def make(i):
            ymin, xmin, ymax, xmax = boxes[i]
            return Object(
                id=int(class_ids[i]),
                score=scores[i],
                bbox=BBox(xmin=np.maximum(0.0, xmin),
                    ymin=np.maximum(0.0, ymin),
                    xmax=np.minimum(1.0, xmax),
                    ymax=np.minimum(1.0, ymax)))

        return [make(i) for i in range(top_k) if scores[i] >= score_threshold]





