import collections
import numpy as np
import operator
from PIL import Image
import cv2

from utils import common

Category = collections.namedtuple('Category', ['id', 'score'])

class FeatureExtractor():
    def __init__(self, extractor_model, labels):
        super(FeatureExtractor, self).__init__()

        self.labels = labels 
        self.feature_dim = 512
        print('Loading features extractor model ...')
        self.interpreter = common.make_interpreter(extractor_model)
        self.interpreter.allocate_tensors()

    def set_input(self, cv2_im):
        self.cv2_im = cv2_im
        self.pil_image = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))

        #common.set_input(self.interpreter, self.pil_image)
        #self.interpreter.invoke()

    def get_features(self, detections):
        imgs = self.multi_crops(self.cv2_im, detections)
        
        if len(imgs) == 0:
            return np.empty((0, self.feature_dim))

        self.embeddings = []

        for img in imgs:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            common.set_input(self.interpreter, pil_img)
            self.interpreter.invoke()
            embedding_out = common.output_tensor(self.interpreter, 0, 'score')
            self.embeddings.append(embedding_out)
        
        embeddings = np.concatenate(self.embeddings).reshape(-1, self.feature_dim)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings

    def multi_crops(self, frame, bboxes):
        imgs = [] 
        for box in bboxes:
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            img = frame[y0:y1, x0:x1]
            imgs.append(img)

        return imgs

    '''
    def get_output(self, score_threshold=0.4, top_k=3):

        """Returns no more than top_k categories with score >= score_threshold."""
        scores = common.output_tensor(self.interpreter, 0, 'score')
        categories = [
            Category(i, scores[i])
            for i in np.argpartition(scores, -top_k)[-top_k:]
            if scores[i] >= score_threshold
        ]
        return sorted(categories, key=operator.itemgetter(1), reverse=True)
    '''

    
    