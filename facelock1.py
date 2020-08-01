import cv2
import os
import numpy

alkdalkalda

face =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    if not os.path.exists('datacollection'):
        os.makedirs('datacollection')
except OSError:
    print('directorry not created')
def face_extraction(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,1.5,5)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        crop =img[y:y+h, x:x+w]
    return crop
cam = cv2.VideoCapture(0)
c = 0
while(True):
    checker,frame = cam.read()
    if face_extraction(frame) is not None:
        c +=1
        face1 = cv2.resize(face_extraction(frame),(200,200))
        face2 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
        name = './datacollection/count' + str(c) + '.jpg'
        print(name)
        cv2.imwrite(name,face2)
        cv2.imshow("frame ",face2)
    else:
        print('not found')
        pass
    key = cv2.waitKey(1)
    if key == 27 or c == 400 :
        break
cam.release()
cv2.destroyAllWindows()
