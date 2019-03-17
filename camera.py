import cv2
import numpy as np
from GUITWO import SabKuch

class VideoCamera(object):
    def __init__(self):

        #self.face_cascade = cv2.CascadeClassifier('face-har.xml')
        #self.eye_cascade = cv2.CascadeClassifier('haar-eye.xml')
        self.video = cv2.VideoCapture(0)
        # cap = 
        self.word = ""
        self.action = 10

    def __del__(self):
        self.video.release()

    def get_word(self):
        return self.word
    
    def get_frame(self):
        ret, frame = self.video.read()
        en = cv2.imencode('.jpg', frame)[1].tostring()
        img,self.word,self.action = SabKuch(en)
        # print ('self: ' + self.word)

        de = cv2.imdecode(np.fromstring(img, np.uint8),1)


        ret, jpeg = cv2.imencode('.jpg', de)
        return jpeg.tobytes(), self.word, self.action

