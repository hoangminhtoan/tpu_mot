import enum
import cv2
import re

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

def visualize_image(cv2_im, objs, labels, text_lines, trkdata):
    height, width, _ = cv2_im.shape

    for trk in trkdata:
        x0, y0, x1, y1, trackID = trk[0].item(), trk[1].item(), trk[2].item(), trk[3].item(), trk[4].item()
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)

        overlap = 0
        ob = None
        for obj in objs:
            dx0, dy0, dx1, dy1 = list(obj.bbox)
            dx0, dy0, dx1, dy1 = int(dx0*width), int(dy0*height), int(dx1*width), int(dy1*height)
            
            area = (min(dx1, x1) - max(dx0, x0)) * (min(dy1, y1) - max(dy0, y0))
            if area > overlap:
                overlap = area
                ob = obj

        # Relative cooridnates
        percent = int(100 * ob.score)
        label = '{}% {} ID:{}'.format(percent, labels.get(ob.id, ob.id), int(trackID))

        for i, line in enumerate(text_lines):
            cv2_im = cv2.putText(cv2_im, line, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                             
    return cv2_im