import pickle
import face_recognition as fr

def newface(img, name):
    face = fr.load_image_file(img)
    print('encoding...',end='\n\n')
    face_encoding = fr.face_encodings(face)[0]
    print('encoded',end='\n\n')
    faces=[]
    with open("faceRec.dat", 'rb') as f:
        try:
            while True:
                faces = (pickle.load(f))
        except EOFError:
            pass
    faces.append((name, face_encoding))
    pickle.dump(faces, open('faceRec.dat', 'wb'))
    print('face added',end='\n\n')

def readfaces():
    faces=[]
    with open("faceRec.dat", 'rb') as f:
        try:
            while True:
                faces = (pickle.load(f))
        except EOFError:
            pass
    return (faces)

# newface(input(str("enter image name: ")), input(str(('enter face name: '))))