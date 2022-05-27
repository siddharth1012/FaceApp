import numpy as np
import cv2
import sklearn
import pickle
from django.conf import settings 
import os
from imutils import face_utils
import dlib

STATIC_DIR = settings.STATIC_DIR


# face detection
face_detector_model = cv2.dnn.readNetFromCaffe(os.path.join(STATIC_DIR,'models/deploy.prototxt.txt'),
                                               os.path.join(STATIC_DIR,'models/res10_300x300_ssd_iter_140000.caffemodel'))
# feature extraction
face_feature_model = cv2.dnn.readNetFromTorch(os.path.join(STATIC_DIR,'models/openface.nn4.small2.v1.t7'))
# face recognition
face_recognition_model = pickle.load(open(os.path.join(STATIC_DIR,'models/machinelearning_face_person_identity.pkl'),
                                          mode='rb'))
# emotion recognition model
emotion_recognition_model = pickle.load(open(os.path.join(STATIC_DIR,'models/machinelearning_face_emotion.pkl'),mode='rb'))

# shape predictor
shape_predictor = dlib.shape_predictor(os.path.join(STATIC_DIR,'./models/shape_predictor_68_face_landmarks.dat'))
#face descriptor
shape_descriptor = dlib.face_recognition_model_v1(os.path.join(STATIC_DIR,'./models/dlib_face_recognition_resnet_model_v1.dat'))

def pipeline_model(path):
    # pipeline model
    img = cv2.imread(path)
    image = img.copy()
    h,w = img.shape[:2]
    # face detection
    img_blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,177,123),swapRB=False,crop=False)
    face_detector_model.setInput(img_blob)
    detections = face_detector_model.forward()
    
    # machcine results
    machinlearning_results = dict(face_detect_score = [], 
                                 face_name = [],
                                 face_name_score = [],
                                 emotion_name = [],
                                 emotion_name_score = [],
                                 count = [])
    count = 1
    if len(detections) > 0:
        for i , confidence in enumerate(detections[0,0,:,2]):
            if confidence > 0.5:
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                startx,starty,endx,endy = box.astype(int)

                cv2.rectangle(image,(startx,starty),(endx,endy),(0,255,0))

                # feature extraction
                face_roi = img[starty:endy,startx:endx]
                face_blob = cv2.dnn.blobFromImage(face_roi,1/255,(96,96),(0,0,0),swapRB=True,crop=True)
                face_feature_model.setInput(face_blob)
                vectors = face_feature_model.forward()

                # predict with machine learning
                face_name = face_recognition_model.predict(vectors)[0]
                face_score = face_recognition_model.predict_proba(vectors).max()
                # EMOTION 
                emotion_name = emotion_recognition_model.predict(vectors)[0]
                emotion_score = emotion_recognition_model.predict_proba(vectors).max()

                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/process.jpg'),image)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/roi_{}.jpg'.format(count)),face_roi)
                
                machinlearning_results['count'].append(count)
                machinlearning_results['face_detect_score'].append(confidence)
                machinlearning_results['face_name'].append(face_name)
                machinlearning_results['face_name_score'].append(face_score)
                machinlearning_results['emotion_name'].append(emotion_name)
                machinlearning_results['emotion_name_score'].append(emotion_score)
                
                count += 1
                
            
    return machinlearning_results

#detects key facial points of a face in images

def key_feature(path):
    
    img = cv2.imread(path)
    image = img.copy()
    h,w = img.shape[:2]
    # face detection
    img_blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,177,123),swapRB=False,crop=False)
    face_detector_model.setInput(img_blob)
    detections = face_detector_model.forward()
    count = 1
    if len(detections) > 0:
        for i , confidence in enumerate(detections[0,0,:,2]):
            if confidence > 0.5:
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                startx,starty,endx,endy = box.astype(int)
                
                #facial key feature
                face_detector = dlib.get_frontal_face_detector()
                face_roi = img[starty:endy,startx:endx]
                faces = face_detector(image)
                for box in faces:
                    pt1 = box.left(), box.top()
                    pt2 = box.right(), box.bottom()
                        
                    face_shape = shape_predictor(image,box)
                    face_shape_array = face_utils.shape_to_np(face_shape)
                    shape_descriptor.compute_face_descriptor(image,face_shape)
                    #print(face_shape_array)
                    for points in face_shape_array:
                        cv2.circle(image,tuple(points),3,(0,255,0))
                #prints image on screen as a result        
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/process.jpg'),image)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/roi_{}.jpg'.format(count)),face_roi)
                
                count += 1