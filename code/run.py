from scipy.signal import wiener
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import *
import numpy as np
import pandas as pd
import warnings
import cv2
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import load_model


def wiener_bilateral_filter(img):
    ar, ag, ab = cv2.split(img)
    Y1 = wiener(ar, (1,1))  
    Y2 = wiener(ag, (1,1))  
    Y3 = wiener(ab, (1,1))  
    image_ = cv2.merge((Y1, Y2, Y3))
    where_ab = isnan(image_)
    image_[where_ab] = 0
    img_1 = np.array(image_,dtype=np.uint8)
    bilateral = cv2.bilateralFilter(img_1,10,30,30)
    return bilateral

def main():
    
    c_ = 0
    cap = cv2.VideoCapture('testing_video.mp4')
    if (cap.isOpened()== False): 
        print("Error opening video  file")
    
    model_fe = load_model("Model save/model_AttenIncBiLSTM")
    model = load_model('Model save/model_GTAE')
    
    while(cap.isOpened()):   
        ret, frame = cap.read()
        if ret == True:
            Model = Mode(c_)
            image = cv2.resize(frame,(200,200))
            frame = cv2.resize(frame,(800,500))
            # wiener bilateralFilter
            wie_bi_img = wiener_bilateral_filter(image)
            wi_img = wie_bi_img.copy()
            wie_bi_img = wie_bi_img.reshape(1,wie_bi_img.shape[0],wie_bi_img.shape[1],wie_bi_img.shape[2])
            
            import Feature_extraction
            extracted_feat = Feature_extraction.Extracting(model_fe,wie_bi_img)
            extracted_feat = np.nan_to_num(extracted_feat) 
            
            import Feature_selection
            sel = np.load("Feature save/select_index.npy")
            selected_feat = extracted_feat[:,sel]
            selected_feat = selected_feat.reshape(selected_feat.shape[0],selected_feat.shape[1],1)

            pred = model.predict(selected_feat)
            
            if pred==0 :
                perd_cls = 'CalotTriangleDissection'
            if pred==1 :
                perd_cls = 'CleaningCoagulation'
            if pred==2 :
                perd_cls = 'ClippingCutting'
            if pred==3 :
                perd_cls = 'GallbladderDissection'
            if pred==4 :
                perd_cls = 'GallbladderPackaging'
            if pred==5 :
                perd_cls = 'GallbladderRetraction'
            if pred==6 :
                perd_cls = 'Preparation'
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50);fontScale = 1;thickness = 1
            color = (0, 255, 0)
            frame_image = cv2.putText(frame,perd_cls, org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('window_name', frame_image) 
            cv2.waitKey(200)
            c_+=1
        else:break

    x_test = np.load("Feature save/x_test.npy");
    ytest = np.load("Feature save/ytest.npy");
    
    import classifier
    pro_pred = classifier.predition('Model save/model_GTAE',x_test)

if __name__ == "__main__":
    main()

    