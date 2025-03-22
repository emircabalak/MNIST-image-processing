import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
import pandas as pd

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape(-1, 28, 28, 1)  
test_images = test_images.reshape(-1, 28, 28, 1)    

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_labels = to_categorical(train_labels,10)
test_labels = to_categorical(test_labels,10)

datagen = ImageDataGenerator(
    rotation_range=5,       
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2, 
    zoom_range=0.05,        
    fill_mode='nearest'     
)


model = Sequential()
model.add(Conv2D(32,(3,3),padding="same",activation="relu", input_shape= (28,28,1)))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))

earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=256),  
    epochs=10,
    validation_data=(test_images, test_labels),  
    callbacks=[earlystop]
)
model.save(r'C:\Users\emirc\Desktop\mnist\my_mnist_model2.keras')

loaded_model = load_model('my_mnist_model2.keras')
loaded_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

test_image = test_images[0]
test_image = test_image.reshape((1, 28, 28))
test_image = test_image.astype('float32') / 255.0

predictions = loaded_model.predict(test_image)

kayit = pd.DataFrame(model.history.history)
kayit.plot()