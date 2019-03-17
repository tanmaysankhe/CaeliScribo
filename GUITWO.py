import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import load_model
# from collections import deque

import cv2




import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
# sess = tf.Session(config = config)


#M = cv2.getRotationMatrix2D((320,180), 180, 1.0)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'new_fingertip'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1





# Letters lookup
#letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
#11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
#21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}
letters = { i: chr(i+ord('A')-1) for i in range (1,28)}
letters[27] = "-"
# print(letters)


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# cap = cv2.VideoCapture('http://192.168.0.101:4747/video')
# cap = cv2.VideoCapture('http://192.168.43.123:4747/video')

# Define prediction variables
prediction1 = 26
prediction2 = 26

fingertip_tracking_list = []
# In[10]:

character_recognition_counter = 29
black_image = np.zeros((360, 640),dtype=np.uint8)



with detection_graph.as_default():
  sess = tf.Session(graph=detection_graph, config = config)
  if True:

    #loading charcater recognition model MNIST
        # mlp_model = load_model('emnist_mlp_model.h5')
    cnn_model = load_model('model.h5')
    # mlp_model._make_predict_function()
    cnn_model._make_predict_function()


    action, deque = [-1], [-1]

    def SabKuch(en):
      global action, deque
      alphabet_drawn = ""
      global prediction1, prediction2,fingertip_tracking_list, letters, black_image, character_recognition_counter, label_map, categories, category_index
      image_np = cv2.imdecode(np.fromstring(en, np.uint8),1)
      #image_np = cv2.flip(image_np, 1)

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      """vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      """
      image_np = cv2.resize(image_np, (640,360))
      #image_np = cv2.warpAffine(image_np, M, (640,360)) 
      coordinates = vis_util.return_coordinates(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=1,
                        min_score_thresh=0.80)
      
      finger_count = len(coordinates)
      # print( action )
      
      if deque[-1] == finger_count:
        deque.append(finger_count)
      else:
        deque = [finger_count]

      if len(deque) == 5 and action[0] != 3:
        action[0] = deque[-1]
      elif len(deque) == 5 and action[0] == 3:
        action.append(deque[-1])


      if action[0] == 1:
        if len(coordinates) != 0:

          coordinates = coordinates[0]
          column = (int(coordinates[0]) + int(coordinates[1]) ) / 2
          row = ( int(coordinates[2]) + int(coordinates[3]) ) / 2
          #print ( row, column)
          #cv2.circle(image_np,(int(coordinates[2]), int(coordinates[0])), 5, (0,255,0), -1)
          #cv2.circle(image_np,(int(coordinates[3]), int(coordinates[1])), 5, (0,0,255), -1)
          fingertip_tracking_list.append( (row,column) )

      
      

      elif action[0] == 2:
        # cv2.imwrite(chr(key-32) + "_" + str(character_recognition_counter) + "haha.jpg", black_image)
        #########################
        blackboard_gray = black_image
        blur1 = cv2.medianBlur(blackboard_gray, 5)
        blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
        thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #cv2.imshow("thresh", thresh1)
        _, blackboard_cnts,_ = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(blackboard_cnts) >= 1:
            cnt = sorted(blackboard_cnts, key = cv2.contourArea, reverse = True)[0]
            # areas = [cv2.contourArea(c) for c in blackboard_cnts]
            # max_index = np.argmax(areas)
            # cnt=contours[max_index]
            # print(cv2.contourArea(cnt))

            if True:
                x, y, w, h = cv2.boundingRect(cnt)
                alphabet = blackboard_gray[y-10:y + h + 10, x-10:x + w + 10]
                newImage = cv2.resize(alphabet, (28, 28))
                newImage = np.array(newImage)
                newImage = newImage.astype('float32')/255

                # prediction1 = mlp_model.predict(newImage.reshape(1,28,28))[0]
                # prediction1 = np.argmax(prediction1)

                prediction2 = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
                prediction2 = np.argmax(prediction2)
                alphabet_drawn = str(letters[int(prediction2)+1])
                # print("prediction1 : ",str(letters[int(prediction1)+1]))
                # print("prediction : ", str(letters[int(prediction2)+1]))

        # Empty the points deque and the blackboard
        # points = deque(maxlen=512)
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

        #########################
        character_recognition_counter += 1
        fingertip_tracking_list = []
        black_image = np.zeros((360, 640), dtype=np.uint8)

        # print ("Saving" + chr(key-32) + "_" + str(character_recognition_counter) )

      #cv2.imshow('window', image_np)
      # elif action == 3:
      #   break  

      if action[0] == 3:
        if len(action) == 1:
          pass
        else:
          print(action[-1])
          deque, action = [-1], [-1]

      if action[0] == 4:
        fingertip_tracking_list = []
        black_image = np.zeros((360, 640),dtype=np.uint8) 

      if action[0] == 5:
        pass
      #print (coordinates if not None else "")
      
       # Put the result on the screen
      # cv2.putText(image_np, "Multilayer Perceptron : " + str(letters[int(prediction1)+1]), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
      #cv2.putText(image_np, "Convolution Neural Network:  ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      cv2.putText(image_np, "" + alphabet_drawn, (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
      # cv2.imshow ("Saving this", black_image)
      
      for z in range(1, len(fingertip_tracking_list) ):
        #cv2.circle(image_np,(int(fingertip_tracking_list[z][0]), int(fingertip_tracking_list[z][1])), 5, (0,0,255), -1)
        cv2.line (image_np, (int(fingertip_tracking_list[z-1][0]), int(fingertip_tracking_list[z-1][1])), (int(fingertip_tracking_list[z][0]), int(fingertip_tracking_list[z][1])), (0,0,255), 5)
        cv2.line (black_image, (int(fingertip_tracking_list[z-1][0]), int(fingertip_tracking_list[z-1][1])), (int(fingertip_tracking_list[z][0]), int(fingertip_tracking_list[z][1])), (255,255,255), 5)
      

      # cv2.imshow("Image",image_np)
      en = cv2.imencode('.jpg', image_np)[1].tostring()

      return en, alphabet_drawn,action[0]
      

      
