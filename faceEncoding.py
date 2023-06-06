import face_recognition as fr
import mysql.connector as db
import pickle
import base64

mydb = db.connect(host = 'localhost', user = 'root', password = '-', database = 'faces')
cursor = mydb.cursor()

def newface(img, name):
    face = fr.load_image_file(img)
    print('encoding...',end='\n\n')
    face_encoding = base64.b64encode(pickle.dumps(fr.face_encodings(face)[0]))
    print('encoded',end='\n\n')
    values= (name, face_encoding)
    cursor.execute('insert into facedata values(%s,%s)',values)
    mydb.commit()
    print('face added',end='\n\n')

newface(r'C:\Users\sayuj\Desktop\face_rec\vedika.jpg','vedika')