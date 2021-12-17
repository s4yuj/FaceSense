import numpy as np
import face_recognition as fr
import cv2
from faces import newface, readfaces
import time

# img = cv2.imread("047ff66b-de6e-4596-935e-89fdd15352e5.jpg")

# img = cv2.resize(img, (0,0), fx= 0.4, fy= 0.4)
# cv2.imshow('hello',img)
# img= cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
# cv2.imwrite('new_img.jpg', img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

ptime = 0
ctime = 0

known_face_encodings = []
known_face_names = []

i=0
while True:
    try:
        known_face_encodings.append(readfaces()[i][1])
        known_face_names.append(readfaces()[i][0])
        i+=1
    except IndexError:
        break

print(f'face list: {known_face_names}',end='\n\n')

ch=input('add new face?(y/n): ')
if ch.lower()=='y':
    newface(input(str("enter image name: ")), input(str(('enter face name: '))))

video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True: 

    ret, frame = video_capture.read()
    frame=cv2.flip(frame,1)
    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(frame, str(int(fps)) , (10, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print('program quit',end='\n\n')
