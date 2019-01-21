import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('opencv-4/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv-4/data/haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    edges = cv2.Canny(frame,100,200)
    #cv2.imshow('Edges',edges)

    #imggrey1 = cv2.cvtColor(imggrey, cv2.COLOR_GRAY2BGR)
    img = frame 
    

    #ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    

    for (x,y,w,h) in faces:
        cv2.rectangle(edges,(x,y),(x+w,y+h),(255,255,255),30)
        roi_gray = edges[y:y+h, x:x+w]
        roi_color = edges[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(51,255,255),10)

    
    cv2.imshow('imggrey+blocks', edges)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
