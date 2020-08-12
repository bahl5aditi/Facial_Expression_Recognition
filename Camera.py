import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np

config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.15
session=tf.compat.v1.Session(config=config)

faceec=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
json_model=open('ConvNetV1_1_best_weights.json','r')
loaded_model_json=json_model.read()
json_model.close()
model_CNN =model_from_json(loaded_model_json)
model_CNN.load_weights('ConvNetV1_1_best_weights.hdf5')
model_CNN.make_predict_function()

json_model2=open('ConvNetV2_1_best_weights.json','r')
loaded_model_json2=json_model2.read()
json_model2.close()
model_CNN2 =model_from_json(loaded_model_json2)
model_CNN2.load_weights('ConvNetV2_1_best_weights.hdf5')
model_CNN2.make_predict_function()

json_model3=open('ConvNetV3_1_best_weights.json','r')
loaded_model_json3=json_model3.read()
json_model3.close()
model_CNN3 =model_from_json(loaded_model_json3)
model_CNN3.load_weights('ConvNetV3_1_best_weights.hdf5')
model_CNN3.make_predict_function()


json_model = open("ConvSIFTNET_1_model.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_SIFTNET = model_from_json(loaded_json_model)
model_SIFTNET.load_weights("ConvSIFTNET_1_best_weights.hdf5")

font=cv2.FONT_HERSHEY_SIMPLEX

class videoCamera(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    #returns camera frame with bounding boxes and predictions
    def get_frame(self):
        _,fr=self.video.read()
        gray_fr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        faces=faceec.detectMultiScale(gray_fr,1.3,5)
        EMOTIONS_LIST = ["Happy", "Angry",
                         "Disgust", "Sad",
                         "Surprise", "Fear",
                         "Neutral"]
        for (x,y,w,h)in faces:
            fc=gray_fr[y:y+h,x:x+w]
            roi=cv2.resize(fc,(48,48))

        pred1=model_CNN.predict(roi[np.newaxis,:,:,np.newaxis])
        pred2=model_CNN2.predict(roi[np.newaxis,:,:,np.newaxis])
        pred3=model_CNN3.predict(roi[np.newaxis,:,:,np.newaxis])

        pred=(pred1+pred2+pred3)/3
        pred=EMOTIONS_LIST[np.argmax(pred)]

        cv2.putText(fr,pred,(x,y),font,1,(255,255,0),2)
        cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _,jpeg=cv2.imencode('.jpg',fr)
        return jpeg.tobytes()
