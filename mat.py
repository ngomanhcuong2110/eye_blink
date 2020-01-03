import cv2
import numpy as np
import dlib
import math
cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def eye_blink(point,lm_face):
    left_point = (landmarks.part(point).x, landmarks.part(point).y)
    right_point = (landmarks.part(point+3).x, landmarks.part(point+3).y)
    center_top_r = (landmarks.part(point+1).x, landmarks.part(point+1).y)
    center_top_l = (landmarks.part(point+2).x, landmarks.part(point+2).y)
    center_bottom_r = (landmarks.part(point+5).x, landmarks.part(point+5).y)
    center_bottom_l = (landmarks.part(point+4).x, landmarks.part(point+4).y)
    center_top=(int(center_top_r[0]/2+center_top_l[0]/2),int(center_top_r[1]/2+center_top_l[1]/2))
    center_bottom=(int(center_bottom_r[0]/2+center_bottom_l[0]/2),int(center_bottom_r[1]/2+center_bottom_l[1]/2))
    hor_line = cv2.line(frame, left_point, right_point, (0,0, 255), 2)
        #ver_line1 = cv2.line(frame, center_top_r, center_bottom_l, (0, 0, 255), 2)
        #ver_line2 = cv2.line(frame, center_top_l, center_bottom_r, (0, 0, 255), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (255, 0, 255), 2)
    hor_len=math.hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))
    ver_len=math.hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
    ratio=ver_len/hor_len
    print("ti le= ",ratio)
    return ratio
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   
    faces=detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
       
        landmarks = predictor(gray, face)
        
        ratio=(eye_blink(36,landmarks)+eye_blink(42,landmarks))/2
        if ratio>5.7:
            fontface = cv2.FONT_HERSHEY_COMPLEX
            fontscale = 4
            fontcolor = (255,0,0)
            cv2.putText(frame, "Blinking", (50,150), fontface, fontscale, fontcolor) 
           
           # cv2.putText(frame,,(50,150),font,0.7,(0,255,0))
            

        cv2.imshow("vid",frame)
    k=cv2.waitKey(10)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
