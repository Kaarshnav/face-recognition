import cv2
import numpy as np
import time 

name = input("Enter your name:-")
num = int(input("Enter the no. of photos :-"))

face_data = []


# Instantiate the cascade_classifier with file name 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True and num:
    time.sleep(0.5)
    ret , frame = cap.read() # Status , Frame
    if not ret:
        continue
    # Find All the faces in the frame
    faces = face_cascade.detectMultiScale(frame , 1.3 ,5) # Frame , scaling factor , neighbors
    faces = sorted(faces , key = lambda x: x[2]*x[3] , reverse = True)
    faces = faces[:1]

        
    print(faces)

    for face in faces:
        x,y,w,h = face
        face_selection = frame[y:y+h , x:x+w]
        print(face_selection.shape)
        cv2.imshow("Face_selection", face_selection)
        face_selection = cv2.resize(face_selection,(100,100))
        print(face_selection.shape)
        face_data.append(face_selection)
        
                
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5) # Frame , start pos , end pos , color , thickness

        num-=1
		
    cv2.imshow("Feed" , frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

print(len(face_data))
face_data = np.array(face_data)
print(face_data.shape)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
np.save(("dataset/"+name ),face_data)
cap.release()
cv2.destroyAllWindows()
