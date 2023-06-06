import numpy as np
import face_recognition as fr
import cv2
import mysql.connector as db
import pickle
import base64

mydb = db.connect(host = 'localhost', user = 'root', password = '-', database = 'faces')
cursor = mydb.cursor()

def readfaces():
    cursor.execute('select * from facedata')
    rec = cursor.fetchall()
    return rec

known_face_encodings = []
known_face_names = []    

for i in readfaces():
    known_face_names.append(i[0])
    known_face_encodings.append(pickle.loads(base64.b64decode(i[1])))

print(f'face list: {known_face_names}',end='\n\n')


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

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print('program quit',end='\n\n')