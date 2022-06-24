import cv2
import numpy as np
import time

## initialize classifiers for face and eye detection ###############################################
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
####################################################################################################

## initialize blob detection #######################################################################
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)
####################################################################################################

## main loop (capture image, find faces, find eyes, find pupils, display) ##########################
cap = cv2.VideoCapture(0)
while(True):
    __, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)    
    for (x,y,w,h) in faces:
        gray_face = gray[y:y+h,x:x+w]
        face = img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_face)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        threshold = 80
        for(ex,ey,ew,eh) in eyes:
            if ey > y+eh/2:
                pass
            eyecenter = ex + ew/2
            # if eyecenter<w/2:
            eye = face[ex:(ex+ew), ey:(ey+eh)]
            cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
            keypoints = detector.detect(eye)
            cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255))
            # else:
            #     eye = img[y:y + h + eh, x:x + h + ew]
            #     cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
            
            
            #cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)


    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
####################################################################################################