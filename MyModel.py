import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization , Input ,concatenate
from keras.losses import categorical_crossentropy,categorical_hinge,hinge,squared_hinge
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

def fer_dataset():
    x=np.load('C:/Users/RAJEEV BAHL/Desktop/dataset/Fer_X.npy')
    y=np.load('C:/Users/RAJEEV BAHL/Desktop/dataset/Fer_Y.npy')
    x=np.expand_dims(x,-1)
    x = x / 255.0
    y = np.eye(7, dtype='uint8')[y]

    Split = np.load('C:/Users/RAJEEV BAHL/Desktop/dataset/Fer_Usage.npy')
    x_index = np.where(Split == 'Training')
    y_index = np.where(Split == 'PublicTest')
    z_index = np.where(Split == 'PrivateTest')
    #print("X_INDEX = ",x_index)
    #print("X_INDEX[0] = ", x_index[0][0])
    #print("X_INDEX[-1] = ",x_index[0][-1])
    X_Train = x[x_index[0][0]:x_index[0][-1]+1]
    X_Valid = x[y_index[0][0]:y_index[0][-1]+1]
    X_Test = x[z_index[0][0]:z_index[0][-1]+1]
    #print("Y_INDEX = ", y_index)
    #print("Y_INDEX[0] = ", y_index[0][0])
    #print("Y_INDEX[-1] = ",y_index[0][-1])
    Y_Train = y[x_index[0][0]:x_index[0][-1]+1]
    Y_Valid = y[y_index[0][0]:y_index[0][-1] + 1]
    Y_Test = y[z_index[0][0]:z_index[0][-1] + 1]
    return X_Train,X_Valid,X_Test,Y_Train,Y_Valid,Y_Test

def CNN3():

    num_features = 64
    num_labels = 7
    batch_size = 128
    length,width,height = 48, 48, 1
    epochs = 300

    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test = fer_dataset()
    model = Sequential()


    #ConvNet-1
    model.add(Conv2D(64, (3,3), padding='same', input_shape=(length, width, height)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #ConvNet-2
    model.add(Conv2D(128, (5,5),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #ConvNet-3
    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #ConvNet-4
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #Flatten
    model.add(Flatten())

    #Dense Fully Connected-1
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    #Dense Fully Connected-2
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    #Output Layer
    model.add(Dense(num_labels, activation='softmax'))
    opt=Adam(lr=0.0005)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()

    filepath="ConvNetV3_1_best_weights.hdf5"
    es=EarlyStopping(monitor="val_accuracy",patience=50,mode='max',verbose=1)
    cp=ModelCheckpoint(filepath,monitor="val_accuracy",verbose=1,save_best_only=True,mode='max')
    callbacks=[es,cp]
    model_json=model.to_json()
    with open("ConvNetV3_1_best_weights.json",'w')as json_file:
        json_file.write(model_json)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  #patience=2, min_lr=0.00001, mode='auto')
    model.fit_generator(data_generator.flow(X_Train, Y_Train,
                                            batch_size=batch_size),
                        steps_per_epoch=len(Y_Train) / batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_Valid, Y_Valid),
                        shuffle=True
                        )

    print("Model has been saved to disk ! Training time done !")

def CNN2():
    img_size=48
    batch_size=64
    epochs=300

    data_generator=ImageDataGenerator(horizontal_flip=True)
    X_Train,X_Valid,X_Test,Y_Train,Y_Valid,Y_Test=fer_dataset()

    model=Sequential()

    #ConvNet-1
    model.add(Conv2D(64,(3,3),padding='same', input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #ConvNet-2
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #ConvNet-3
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #ConvNet-4
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #Flattening
    model.add(Flatten())

    #Fully Connected Layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Dense(7,activation='softmax'))

    opt=Adam(lr=0.0005)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()

    filepath="ConvNetV2_1_best_weights.hdf5"
    model_json=model.to_json()
    with open('ConvNetV2_1_best_weights.json','w')as jsonfile:
        jsonfile.write(model_json)
    reduce_lr= ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2,min_lr=0.00001,mode='auto')
    mc=ModelCheckpoint(filepath,monitor='val_accuracy',mode='max',verbose=1,save_best_only=True)
    model.fit_generator(data_generator.flow(X_Train,Y_Train,batch_size=batch_size),validation_data=(X_Valid,Y_Valid),steps_per_epoch=len(Y_Train)/batch_size
                        ,epochs=epochs,verbose=1,callbacks=[reduce_lr,mc])

    print("Model has been saved to disk! Training time done!")



def CNN1():
    X_Train,X_Valid,X_Test,Y_Train,Y_Valid,Y_Test = fer_dataset()
    num_features = 32
    num_labels = 7
    batch_size = 128
    epochs = 300


    data_generator = ImageDataGenerator(
       featurewise_center=False,
       featurewise_std_normalization=False,
       rotation_range=10,
       width_shift_range=0.1,
       height_shift_range=0.1,
       zoom_range=.1,
       horizontal_flip=True)

    model = Sequential()
    model.add(Conv2D(num_features, kernel_size=(3,3), activation='relu', input_shape=(48, 48, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
 #   model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(2*num_features, kernel_size=(3, 3),activation='relu',  padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu',  padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])
    #model.fit(X_Train, Y_Train, validation_data=(X_Valid, Y_Valid), epochs=300, verbose=1)

    ##model.fit(X_Train, Y_Train, validation_data=(X_Valid, Y_Valid), epochs=epochs, verbose=1, batch_size=batch_size)

    filepath = "ConvNetV1_1_best_weights.hdf5"
    es = EarlyStopping(monitor="val_accuracy", patience=50, mode='max')
    cp = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode=True)
   #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  #patience=2, min_lr=0.00001, mode='auto')
    callbacks = [es, cp]
    model_json = model.to_json()
    with open("ConvNetV1_1_best_weights.json", 'w')as json_file:
        json_file.write(model_json)

    model.fit_generator(data_generator.flow(X_Train, Y_Train,
                                            batch_size=batch_size),
                        steps_per_epoch=len(Y_Train) / batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_Valid, Y_Valid),
                        shuffle=True
                        )
    print("Training time done !")

    model.predict(X_Test[:4])
    print("Actual result: ",Y_Test[:4])

def ExtractFeatures_Layer(dim):
    model=Sequential()
    model.add(Dense(4096,input_dim=dim,kernel_regularizer=l2(0.1)))
    model.add(Dropout(0.5))

    return model

def CNN_Layer(w,h,d):
    num_features=64
    model=Sequential()
    model.add(Conv2D(num_features,kernel_size=(3,3),activation='relu',input_shape=(w,h,d),data_format='channels_last',kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2*num_features,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(4*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(4*num_features,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    return model

def CNN_SIFT():
    num_labels=7
    batch_size=128
    epochs=300
    width,height,depth=48,48,1

    data_generator=ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    print("Loading Data")

    X_Train,X_Valid,X_Test,Y_Train,Y_Valid,Y_Test=fer_dataset()

    Split=np.load('C:/Users/RAJEEV BAHL/Desktop/dataset/Fer_Usage.npy')
    x_index, =np.where(Split=='Training')
    y_index, =np.where(Split=="PublicTest")

    X_SIFT=np.load('C:/Users/RAJEEV BAHL/PycharmProjects/FacialExpRecog/feature_extraction/Fer2013_SIFTDetector_Histogram.npy')
    X_SIFT=X_SIFT.astype('float64')
    X_SIFT_Train=X_SIFT[x_index[0]:x_index[-1]+1]
    X_SIFT_Valid=X_SIFT[y_index[0]:y_index[-1]+1]

    print("Data has been generated")
    print(X_SIFT_Train.shape[1])
    SIFT=ExtractFeatures_Layer(X_SIFT_Train.shape[1])
    CNN=CNN_Layer(width,height,depth)

    MergeModel=concatenate([CNN.output,SIFT.output])
    m=Dense(2048,activation='relu')(MergeModel)
    m=Dropout(0.5)(m)
    m=Dense(num_labels,activation='softmax')(m)

    model=Model(inputs=[CNN.input,SIFT.input],outputs=m)
    model.compile(loss=categorical_crossentropy,optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-7),metrics=['accuracy'])

    filepath='ConVSIFTNET_1_bets_weights.hdf5'
    es=EarlyStopping(monitor='val_accuracy',patience=50,mode='max')
    cp=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode="max")
    callbacks=[cp,es]

    model_json=model.to_json()
    with open('ConVSIFTNET_1_mdoel.json','w')as jsonfile:
        jsonfile.write(model_json)

    model.fit_generator(data_generator.flow([X_Train,X_SIFT_Train],Y_Train,batch_size=batch_size),
                        steps_per_epoch=len(Y_Train)/batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=([X_Valid,X_SIFT_Valid],Y_Valid),
                        shuffle=True)
    print('Model has been saved to Disk!Training time done!')



#CNN3()
#CNN2()
#CNN1()
CNN_SIFT()