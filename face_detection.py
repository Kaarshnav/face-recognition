import cv2


# Instantiate the cascade_classifier with file name 
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
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5) # Frame , start pos , end pos , color , thickness
		
    cv2.imshow("Feed" , frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
