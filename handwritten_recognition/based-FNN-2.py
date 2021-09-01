import tensorflow as tf

# (1) データのインポート
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# (2)データの変形
import numpy as np
from tensorflow.python.keras.utils import np_utils
# (2-1)
x_train = x_train.reshape(60000, 784)       # (2-1-1)
x_train = x_train.astype('float32')         # (2-1-2)
x_train = x_train / 255                     # (2-1-3)
num_classes = 10                            # (2-1-4)
y_train = np_utils.to_categorical(y_train, num_classes) # (2-1-5)
# (2-2)
x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test = x_test / 255
y_test = np_utils.to_categorical(y_test, num_classes)

# (3) ネットワークの定義
np.random.seed(1)                                        # (3-1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
model = Sequential()                                     # (3-2)
model.add(Dense(100, input_dim=784, activation='relu'))  # (3-3)
model.add(Dropout(0.75))
model.add(Dense(10, activation='softmax'))               # (3-4)
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])     # (3-5)

# (4) 学習
num_epochs = 100
batchsize = 100             # (4-1)

import time                  # (4-2)
startTime = time.time()      # (4-3)
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batchsize,
                    verbose=1, validation_data=(x_test, y_test))        # (4-4)
score = model.evaluate(x_test, y_test, verbose=0, batch_size=batchsize) # (4-5)

# (5) 計算結果の表示
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Computation time:{0:.3f} sec".format(time.time() - startTime))
print(model.summary())



#交差エントロピーと正答率の時間発展の可視化
import matplotlib.pyplot as plt

plt.figure(1, figsize=(10, 4))
plt.subplots_adjust(wspace=0.5)

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training', color='black')
plt.plot(history.history['val_loss'], label='Test',
color='cornflowerblue')
plt.ylim(0, 1)
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

"""
#100個のテストデータに対する正誤の表示
import matplotlib.pyplot as plt
def show_prediction():
    n_show = 100
    y = model.predict(x_test)  # (A)
    plt.figure(figsize=(8, 10))
    plt.gray()
    for i in range(n_show):
        plt.subplot(10, 10, i + 1)
        x = x_test[i, :]
        x = x.reshape(28, 28)
        plt.pcolor(1 - x)
        wk = y[i, :]
        prediction = np.argmax(wk)
        plt.text(22, 25.5, "%d" % prediction, color='orange',
                 fontsize=10)
        if prediction != np.argmax(y_test[i, :]):
            plt.plot([0, 27], [1, 1], color='orange', linewidth=5)
        plt.xlim(0, 27)
        plt.ylim(27, 0)
        plt.xticks([], "")
        plt.yticks([], "")

# -- メイン
show_prediction()
plt.show()


"""
