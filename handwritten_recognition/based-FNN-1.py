import tensorflow as tf

# (1) データのインポート
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# (2)データの変形
import numpy as np
from tensorflow.python.keras.utils import np_utils
# (2-1)
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train = x_train / 255
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
# (2-2)
x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test = x_test / 255
y_test = np_utils.to_categorical(y_test, num_classes)

# (3) ネットワークの定義
np.random.seed(1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers import SGD
model = Sequential()

num_hidden_layers = 2
a = 8
num_first_hidden_units = num_classes * a ** num_hidden_layers
model.add(Dense(num_first_hidden_units, input_dim=784, activation='relu'))
for k in range(2, num_hidden_layers + 1):
    num_hidden_units = num_classes * a ** (num_hidden_layers + 1 - k)
    model.add(Dense(num_hidden_units, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

# (4) 学習
num_epochs = 100
batchsize = 1000

import time
startTime = time.time()
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batchsize,
                    verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0, batch_size=batchsize)

# (5) 計算結果の表示
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Computation time:{0:.3f} sec".format(time.time() - startTime))

print(model.summary())

# リスト 8-1-(6)
#交差エントロピーと正答率の時間発展の可視化
import matplotlib.pyplot as plt

plt.figure(1, figsize=(10, 4))
plt.subplots_adjust(wspace=0.5)

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training', color='black')
plt.plot(history.history['val_loss'], label='Test',
color='cornflowerblue')
plt.ylim(0, 4)
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Test Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], label='Training', color='black')
plt.plot(history.history['val_acc'],label='Test', color='cornflowerblue')
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
