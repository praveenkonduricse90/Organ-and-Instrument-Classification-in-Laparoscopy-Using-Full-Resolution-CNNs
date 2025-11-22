from scipy.signal import wiener
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import pandas as pd
import warnings
import cv2
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    path = 'Dataset/Data/'
    classes_file = os.listdir(path)
    classes_file = [x for x in classes_file if x not in ['Thumbs.db','.DS_Store']]
    
    cls_lab = dict();l=0;
    for cl in classes_file:
        cls_lab.update({cl:l})
        l=l+1
    
    label = list()
    image_data = list()
    for cl_file in classes_file:
        root_file = path+cl_file+'/'
        img_file = os.listdir(root_file)
        img_file = [x for x in img_file if x not in ['Thumbs.db','.DS_Store']]
        for file in img_file:
            imagepath = root_file+file
            if ".jpg" in imagepath:
                print(imagepath)
                label.append(cls_lab[cl_file])
                image = cv2.imread(imagepath)
                image = cv2.resize(image,(200,200))
                ## wiener bilateralFilter
                wiener_bilateral_img = wiener_bilateral_filter(image)
                image_data.append(wiener_bilateral_img)
    
    image_data = np.array(image_data)
    label = np.array(label)
    
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(label)
    label_trans = lb.transform(label)
    
    print("Feature Extracting Processing.......")
    import Feature_extraction
    Feature_extraction.Attention_Inception_Bi_LSTM()
    extracted_feat = Feature_extraction.Extract(image_data)
    extracted_feat = np.nan_to_num(extracted_feat) 
    print("Feature Extracting completed")
    
    print("Feature Selection Processing.......")
    import Feature_selection
    sel = Feature_selection.AQUILA(extracted_feat,label)
    selected_feat = extracted_feat[:,sel]
    # np.save("Feature save/select_index.npy",sel)
    print("Feature Selection completed")
    
    print("Train Test Split")
    feature_selected = selected_feat.reshape(selected_feat.shape[0],selected_feat.shape[1],1)
    from sklearn.model_selection import train_test_split
    x_train,x_test,ytrain,ytest =train_test_split(feature_selected,label_trans, test_size=0.2)
    # np.save("Feature save/x_train.npy",x_train);np.save("Feature save/x_test.npy",x_test);
    # np.save("Feature save/ytrain.npy",ytrain);np.save("Feature save/ytest.npy",ytest);
    n_class= len(np.unique(label))
    
    print("Model Training.......")
    import classifier
    model = classifier.Proposed(x_train,ytrain,n_class, path='Model save/model_GTAE')
   
        
    
if __name__ == "__main__":
    main()

