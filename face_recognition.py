import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

print(os.listdir('dataset'))

files = [faces for faces in  os.listdir('dataset') if faces.endswith('.npy') ]
names = [faces[:-4] for faces in files]
print(names)

face_data = []

for filename in files:
    data = np.load('dataset/' + filename,allow_pickle=True)
    print(data.shape)
    face_data.append(data)
    
face_data = np.array(face_data)
print(face_data.shape)
print(names)
face_data = np.concatenate(face_data ,axis = 0)
print(face_data.shape)
names = np.repeat(names , 10)
print(names.shape)
names = names.reshape((names.shape[0],1))
print(names.shape)

dataset = np.hstack((face_data, names))
print(dataset.shape)

knn = KNeighborsClassifier(n_neighbors = 5)

#Training

knn.fit(dataset[:, :-1] , dataset[:,-1])


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read() # Status , Frame
    if not ret:
        continue
    # Find All the faces in the frame
    faces = face_cascade.detectMultiScale(frame , 1.3 ,5) # Frame , scaling factor , neighbors

        
    print(faces)

    for face in faces:
        x,y,w,h = face
        face_selection = frame[y:y+h , x:x+w]
        print(face_selection.shape)
        # cv2.imshow("Face_selection", face_selection)
        face_selection = cv2.resize(face_selection,(100,100))
        print(face_selection.shape)
        face_cropped = face_selection.reshape((1,-1))
        print(face_cropped.shape)
        
        pred = knn.predict(face_cropped)
        print(face_selection.shape)
        
                
        cv2.rectangle(frame,(x,y),(x+w,y+h),(220,255,22),2) # Frame , start pos , end pos , color , thickness
        cv2.putText(frame , pred[0] , (x,y),cv2.FONT_HERSHEY_SIMPLEX , 1 ,(255,240,235),1)
		
    cv2.imshow("Feed" , frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





