# facerec.py
import cv2, sys, numpy, os
import pickle
import cv2
import os
import pickle
import numpy as np
import openface


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

def getRep(pimg):

    alignedFace = align.align( 96, pimg, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    if(alignedFace == None):
        alignedFace = pimg

    rep = net.forward(alignedFace)
    
    return rep

def infer(img):

    rep = getRep(img).reshape(1, -1)
    predictions = clf.predict_proba(rep).ravel()
    maxI = np.argmax(predictions)
    person = le.inverse_transform(maxI)
    confidence = predictions[maxI]
    
    return person,confidence

def prosama(facec):

    (x,y,w,h) = facec
    face = rgbIm[y:y+h, x:x+w]
    face_resize = cv2.resize(face, (96, 96))
    person, prob = infer(face_resize)
    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
    if (round(100*prob,2) > 80):
        cv2.putText(im,'%s - %.0f' % (person,round(100*prob,2)),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    else:
        cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

if __name__ == '__main__':

    align = openface.AlignDlib(os.path.join( dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    net = openface.TorchNeuralNet(os.path.join(openfaceModelDir,'nn4.small2.v1.t7'), 96, False)

    with open("./generated-embeddings/classifier.pkl", 'r') as f:
        (le, clf) = pickle.load(f)

    haar_file = 'haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)
        
    while True:
        (_, im) = webcam.read()
        rgbIm = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(rgbIm, 1.3, 5)

        for coor in faces:
            prosama(coor)

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break