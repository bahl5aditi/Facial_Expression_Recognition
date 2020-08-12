import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
import pickle
import cupy as cp
import vlfeat as cy

dic_name={1:'Fer2013',2:'CK+',3:'BigFer2013'}
Fer_error_imgs=[2172, 5275, 6459, 7630, 10424, 11287, 13149, 13403, 13989, 15895, 22199, 22928, 28602, 30003]

def Extractor(Images,method_detector="FAST",data_code=1):
    Detector=None
    Descriptor=None

    if(method_detector=="FAST"):
        Detector=cv2.FastFeatureDetector.create()
        Descriptor=cv2.xfeatures2d.SIFT_create()
    else:
        Descriptor=cv2.xfeatures2d.SIFT_create()
        Detector = cv2.xfeatures2d.SIFT_create()


    desc_seq=[]
    count=0
    for img in Images:
        kp=Detector.detect(img)
        kp,desc=Descriptor.compute(img,kp)

        if(len(kp)==0):
          count+=1
          continue
        print("IMAGE number"+str(count)+"has been extracted!")
        desc_seq.append(desc)
        count+=1

    print("Images Extracted")
    print("Start Concatenating")
    descriptors_data=cp.array(desc_seq[0])
    for remaining in desc_seq[1:]:
        descriptors_data=cp.vstack((descriptors_data,remaining))
    print("Descriptors shape"+str(descriptors_data.shape))
    descriptors_data=cp.asnumpy(descriptors_data)
    print("End Concatenating")
    print("descriptors ready to Cluster")

    filename = "Fer2013_SIFTfeature_SIFTDescriptors.npy"
    np.save(filename,descriptors_data)
    print(filename+" has been saved to disk")

def Vocabularyize(X_filename,data_name_code,K=2048,detector_name="SIFT"):

    X=np.load(X_filename)
    print("Clustering")
    K_model=MiniBatchKMeans(n_clusters=K,max_iter=300,batch_size=K*2,max_no_improvement=30,init_size=K*3).fit(X)
    print("Clustered")
    #save model to disk
    filename = "Fer2013_SIFTDetector_Kmean_model.sav"
    pickle.dump(K_model,open(filename,"wb"))
    print("Model saved")

def Vectorize_of_an_Img(img,Kmean,detector_method="SIFT"):

    if detector_method=="SIFT":
        Detector=cv2.xfeatures2d.SIFT_create()
        Descriptor=cv2.xfeatures2d.SIFT_create()
    else:
        Detector=cv2.FastFeatureDetector.create()
        Descriptor=cv2.xfeatures2d.SIFT_create()

    vector_2048=np.zeros(2048,dtype='uint8')
    kp=Detector.detect(img)
    kp,desc=Descriptor.compute(img,kp)

    if(len(kp)==0):
        return vector_2048
    predictions=Kmean.predict(desc)

    for pre in predictions:
        vector_2048[pre]+=1

    return vector_2048

def Histograms_All_Images(imgs,Kmean,detector_method="SIFT",data_name_code=1,):

    Stack=Vectorize_of_an_Img(imgs[0],Kmean)
    count=0
    print("Image number "+str(count)+" in Stack")
    for img in imgs[1:]:

        vector=Vectorize_of_an_Img(img,Kmean,detector_method)
        Stack=np.vstack((Stack,vector))
        count+=1
        print("Image number "+str(count)+" in Stack")
    print("Histogram Generated")
    filename = "Fer2013_SIFTDetector_Histogram.npy"

    np.save(filename,Stack)
    print("Saved Histogram as numpy array to Disk")

def Dense_SIFT_Extractor(images,data_name_code):
    kp,desc0=cy.vl_dsift(images[0],step=1,size=1,bounds=None,window_size=1,norm=True,
                           fast=True,float_descriptors=True,geometry=(4,4,8),
                           verbose=1)
    count=0
    print("Image number "+str(count)+" in Stack")
    kp, desc1 = cy.vl_dsift(images[1], step=1, size=1, bounds=None, window_size=1, norm=True,
                              fast=True, float_descriptors=True, geometry=(4, 4, 8),
                              verbose=1)
    count = 1
    print("Image number " + str(count) + " in Stack")

    concate=np.concatenate((desc0,desc1))
    for img in images[2:]:
        kp, desc = cy.vl_dsift(img, step=1, size=1, bounds=None, window_size=1, norm=True,
                                  fast=True, float_descriptors=True, geometry=(4, 4, 8),
                                  verbose=1)
        concate=np.concatenate((concate,desc))
        print(concate.shape)
        count += 1
        print("Image number " + str(count) + " in Stack")

    print("All Dense SIFT Descriptors Generated")

    filename = "Fer2013_DenseSIFF_Descriptors.npy"
    concate=np.reshape(concate,(images.shape[0],2025,128))
    np.save(filename,concate)
    print("Saved all Dense SIFT Descriptors as numpy array to Disk")




