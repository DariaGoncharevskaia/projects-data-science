from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
optimizer = Adam()
 
 
def load_train(path):
    datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)
    train_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345)
    #val_datagen_flow = datagen.flow_from_directory(
     #   target_size=(150, 150),
      #  batch_size=16,
       # class_mode='sparse',
       # subset='validation',
       # seed=12345)
 
    #features_train, target_train = next(train_datagen_flow)
    #features_test, target_test = next(val_datagen_flow)
    return (train_datagen_flow)
 
 
def create_model(input_shape):
    backbone = ResNet50(input_shape=(150, 150, 3),
                    weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    include_top=False) 
    #backbone.trainable = False
    model = Sequential()
    optimizer = Adam(lr=0.0001)
    model.add(backbone)
    model.add(GlobalMaxPooling2D())
    model.add(Dense(256,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(12, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
              metrics=['acc']) 
    return model
 
#def train_model(model, train_datagen_flow, val_datagen_flow, batch_size=None, epochs=3,
 #               steps_per_epoch=None, validation_steps=None):
    #features_train, target_train = train_datagen_flow
    #features_test, target_test = test_data
  #  model.fit(train_datagen_flow,
   #           validation_data=val_datagen_flow,
    #          batch_size=batch_size, epochs=epochs,
     #         steps_per_epoch=steps_per_epoch,
      #        validation_steps=validation_steps,
       #       verbose=2, shuffle=True)
 
    #return model

def train_model(model, train_datagen_flow, val_datagen_flow, epochs=5,
               steps_per_epoch=None, validation_steps=None, batch_size=None):
    if steps_per_epoch == None:
        steps_per_epoch = len(train_datagen_flow)
    if validation_steps == None:    
        validation_steps = len(val_datagen_flow)
    model.fit(train_datagen_flow, 
              validation_data= val_datagen_flow,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              batch_size = batch_size,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
 
    return model