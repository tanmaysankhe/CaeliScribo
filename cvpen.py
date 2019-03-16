import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
lower_yellow = np.array([20,100,100]) #Lower bound of hsv color to detect
upper_yellow = np.array([30,255,180]) #Upper bound of hsv color to detect

lost_count = 0
x_pt, y_pt = 1000, 1000

def pointer(img):
    global lost_count, x_pt, y_pt, counter, upper_yellow, lower_yellow, font
    #as per documentation
    hsv =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.inRange(hsv,lower_yellow,upper_yellow)
    mask = cv2.erode(mask,kernel, iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    res=cv2.bitwise_and(img,img,mask=mask)

    _,cnts,_=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 5:
            lost_count = 0
            x_pt = int(x)
            y_pt = int(y)
            cv2.circle(img, (int(x), int(y)), int(radius),(255,255,255), 2)
            cv2.circle(img, center, 5, (216, 100, 138), -1)
            lost_count += 1
            if lost_count > 20:
                x_pt, y_pt = 1000, 1000
    else:
        lost_count += 1
        if lost_count > 20:
            x_pt, y_pt = 1000, 1000
    
    return img, x_pt, y_pt
