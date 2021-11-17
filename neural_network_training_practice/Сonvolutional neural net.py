from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
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
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2))) 
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2))) 
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=12, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model
 
def train_model(model, train_datagen_flow, val_datagen_flow, batch_size=None, epochs=3,
                steps_per_epoch=None, validation_steps=None):
    #features_train, target_train = train_datagen_flow
    #features_test, target_test = test_data
    model.fit(train_datagen_flow,
              validation_data=val_datagen_flow,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
 
    return model