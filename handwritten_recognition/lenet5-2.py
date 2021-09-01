import tensorflow as tf

# MNISTデータベースのインポート
from tensorflow.keras.datasets import mnist
#from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import numpy as np
from tensorflow.python.keras.utils import np_utils

# リスト 8-2-(1)
#👇MNISTをインポートしたのち、データを整形
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.ndim)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# リスト 8-2-(3)
# CNNの定義と学習
np.random.seed(1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import time


model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),
                 padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='valid',
                 activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

startTime = time.time()
batchsize = 1000
num_epochs = 20
history = model.fit(x_train, y_train, batch_size=batchsize, epochs=num_epochs,
                    verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Computation time:{0:.3f} sec".format(time.time() - startTime))# リスト 8-1-(7)
print(model.summary())

#交差エントロピーと正答率の時間発展の可視化
import matplotlib.pyplot as plt

plt.figure(1, figsize=(10, 4))
plt.subplots_adjust(wspace=0.5)

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training', color='black')
plt.plot(history.history['val_loss'], label='Test',
color='cornflowerblue')
plt.ylim(0, 1.5)
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Test Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], label='Training', color='black')
plt.plot(history.history['val_acc'],label='Test', color='cornflowerblue')
plt.ylim(0.8, 1)
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
