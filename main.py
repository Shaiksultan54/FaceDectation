import face_recognition
import cv2
import numpy as np


imgQhadeer = face_recognition.load_image_file('ImagesBasic/Qhadeer.jpg')
imgQhadeer = cv2.cvtColor(imgQhadeer,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/Qhadeer Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgQhadeer)[0]
encodeQhadeer = face_recognition.face_encodings(imgQhadeer)[0]
cv2.rectangle(imgQhadeer,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeQhadeer],encodeTest)
faceDis = face_recognition.face_distance([encodeQhadeer],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Qhadeer',imgQhadeer)
cv2.imshow('Qhadeer Test',imgTest)
cv2.waitKey(0)

