import cv2
import numpy as np
import dlib
import math
cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def eye_blink(point,lm_face):
    left_point = (lm_face.part(point).x, lm_face.part(point).y)
    right_point = (lm_face.part(point+3).x, lm_face.part(point+3).y)
    center_top_r = (lm_face.part(point+1).x, lm_face.part(point+1).y)
    center_top_l = (lm_face.part(point+2).x, lm_face.part(point+2).y)
    center_bottom_r = (lm_face.part(point+5).x, lm_face.part(point+5).y)
    center_bottom_l = (lm_face.part(point+4).x, lm_face.part(point+4).y)
    center_top=(int(center_top_r[0]/2+center_top_l[0]/2),int(center_top_r[1]/2+center_top_l[1]/2))
    center_bottom=(int(center_bottom_r[0]/2+center_bottom_l[0]/2),int(center_bottom_r[1]/2+center_bottom_l[1]/2))

    #hor_line = cv2.line(frame, left_point, right_point, (0,0, 255), 2)
        #ver_line1 = cv2.line(frame, center_top_r, center_bottom_l, (0, 0, 255), 2)
        #ver_line2 = cv2.line(frame, center_top_l, center_bottom_r, (0, 0, 255), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (255, 0, 255), 2)
    hor_len=math.hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))
    ver_len=math.hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
    ratio=ver_len/hor_len
    print("ti le= ",ratio)
    return ratio
def gaze_ratio(point, lm_face):
    left_eye=np.array([(lm_face.part(36).x,lm_face.part(36).y),
                           (lm_face.part(37).x,lm_face.part(37).y),
                           (lm_face.part(38).x,lm_face.part(38).y),
                           (lm_face.part(39).x,lm_face.part(39).y),
                           (lm_face.part(40).x,lm_face.part(40).y),
                           (lm_face.part(41).x,lm_face.part(41).y)],np.int32)
    cao,rong,sau=frame.shape
    mask=np.zeros((cao,rong),np.uint8)
        
        
    cv2.polylines(mask,[left_eye],True,255,2)
    cv2.fillPoly(mask,[left_eye],255)
    left_eye1=cv2.bitwise_and(gray,gray,mask=mask)
    min_x=np.min(left_eye[:,0])
    max_x=np.max(left_eye[:,0])
    min_y=np.min(left_eye[:,1])
    max_y=np.max(left_eye[:,1])       
        
    gray_eye=left_eye1[min_y : max_y,min_x:max_x]
    eye=cv2.resize(gray_eye,None,fx=5,fy=5)
         
        
    rt,thresh=cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY)
    rong_mat_trai,dai_mat_trai=gray_eye.shape
    nuatrai_mattrai=gray_eye[0:rong_mat_trai,0:int(dai_mat_trai/2)]
    nuaphai_mattrai=gray_eye[0:rong_mat_trai,int(dai_mat_trai/2):dai_mat_trai]
    nuatraitrang=cv2.countNonZero(nuatrai_mattrai)
    nuaphaitrang=cv2.countNonZero(nuaphai_mattrai)
   # print("nuatraitrang: ",nuatraitrang)
    #print("nuaphaitrang: ",nuaphaitrang)
    #cv2.imshow("nuatrai",nuatrai_mattrai)
    #cv2.imshow("nuaphai",nuaphai_mattrai)
    gaze_ratio = nuatraitrang/nuaphaitrang
    #cv2.putText(frame, str(gaze_ratio), (70, 350), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
    
    
    
    thresh=cv2.resize(thresh,None,fx=5,fy=5)
    cv2.imshow("eye",eye)
    cv2.imshow("thresh",thresh)
    cv2.imshow("left_eye",left_eye1)
    #cv2.polylines(frame,[left_eye],True,255,2)
    return gaze_ratio


    
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   
    faces=detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
       
        landmarks = predictor(gray, face)
        
        ratio=eye_blink(36,landmarks)/2+eye_blink(42,landmarks)/2
        if ratio>5.7:
            fontface = cv2.FONT_HERSHEY_COMPLEX
            fontscale = 4
            fontcolor = (255,0,0)
            cv2.putText(frame, "Blinking", (50,150), fontface, fontscale, fontcolor) 
           
           # cv2.putText(frame,,(50,150),font,0.7,(0,255,0))
        gaze_ratio1=gaze_ratio(36,landmarks)/2+gaze_ratio(42,landmarks)/2
        if gaze_ratio1 >=1.01:
            cv2.putText(frame, "RIGHT", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
        
        elif 0.95 <= gaze_ratio1 < 1.01:
            cv2.putText(frame, "CENTER", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
        else:
        
            cv2.putText(frame, "LEFT", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
        
                    

        cv2.imshow("vid",frame)
    k=cv2.waitKey(10)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
