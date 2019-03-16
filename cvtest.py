import cv2
import numpy as np 
from cvpen import pointer
import mouseControl as mc

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	frame = cv2.resize(frame, (1350//2, 750//2))
	key = cv2.waitKey(1)
	if not ret or key == ord('q'):
		break
	frame,x,y = pointer(frame)
	mc.move((int(x)-60)*1350//615,(int(y)-60)*750//375)
	cv2.imshow("frame", frame)

	if key == ord('f'):
		mc.click((int(x)-60)*1350//615,(int(y)-60)*750//375)
	

cv2.destroyAllWindows()
cap.release()