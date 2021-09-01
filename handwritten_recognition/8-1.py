#import numpy as np
import tensorflow as tf

# MNISTデータベースのインポート
from tensorflow.keras.datasets import mnist
#from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


#訓練データ初めの３つの可視化
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline #このコマンドは使えないらしい
#https://teratail.com/questions/54481

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]


plt.figure(1, figsize=(19, 113))
plt.subplots_adjust(wspace=0.25)
plt.gray()
for id in range(50):
    plt.subplot(5, 10, id + 1)
    img = x_train[id, :, :]
    plt.pcolor(255 - img)
    # plt.text(22, 26, "%d" % y_train[id],
    #          color='cornflowerblue', fontsize=16)
    plt.xlim(0, 27)
    plt.ylim(27, 0)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(bottom=False,
                left=False,
                right=False,
                top=False)
plt.show()


"""
#データの変形
from tensorflow.python.keras.utils import np_utils

x_train = x_train.reshape(60000, 784)  # (A)
x_train = x_train.astype('float32')   # (B)
x_train = x_train / 255               # (C)
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)  # (D)

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test = x_test / 255
y_test = np_utils.to_categorical(y_test, num_classes)

# リスト 8-1-(4)
#ネットワークの定義
np.random.seed(1)
#numpy.random.seed(seed=シードに用いる値) をシード (種) を指定することで、
#発生する乱数をあらかじめ固定することが可能です。
#乱数を用いる分析や処理で、再現性が必要な場合などに用いられます。
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential()
#「Sequentialクラスは入力と出力が必ず１つずつのネットワーク構成しか定義することができません。
#また、中間の層内でネットワークを分岐させるような構成も作れません。（層の線形スタック構成）」
#https://sinyblog.com/deaplearning/keras_how_to/
model.add(Dense(16, input_dim=784, activation='relu'))
#隠れ層のユニット数16、入力の次元？は784、活性化関数はReLU
model.add(Dense(10, activation='softmax'))
#出力層のユニット数10、活性化関数はソフトマックス
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(), metrics=['accuracy'])
#「lossという引数が損失関数で、名前を文字列で指定します」
#「optimizerには最適化方法を名前で指定します」
#http://marupeke296.com/IKDADV_DL_No2_Keras.html
#Adam(adaptive moment estimation)は2015年にKingmaらが考案した、確率的勾配法をより洗練させたアルゴリズム
#AdamはWikipediaの「確率的勾配降下法」のページにも説明がある。
#「metrics: 訓練時とテスト時にモデルにより評価される評価関数のリスト．
#一般的にはmetrics=['accuracy']を使うことになります．」
#https://keras.io/ja/models/model/#compile

# リスト 8-1-(5)
#学習
import time

startTime = time.time()
# shuffle_index = np.random.permutation(60000)
# x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
history = model.fit(x_train, y_train, epochs=100, batch_size=100,
                    verbose=1, validation_data=(x_test, y_test))  # (A)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Computation time:{0:.3f} sec".format(time.time() - startTime))


# リスト 8-1-(6)
#交差エントロピーと正答率の時間発展の可視化
plt.figure(1, figsize=(10, 4))
plt.subplots_adjust(wspace=0.5)

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='training', color='black')
plt.plot(history.history['val_loss'], label='test',
color='cornflowerblue')
plt.ylim(0, 10)
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], label='training', color='black')
plt.plot(history.history['val_acc'],label='test', color='cornflowerblue')
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()


# リスト 8-1-(7)
#96個のテストデータに対する正誤の表示
def show_prediction():
    n_show = 96
    y = model.predict(x_test)  # (A)
    plt.figure(2, figsize=(12, 8))
    plt.gray()
    for i in range(n_show):
        plt.subplot(8, 12, i + 1)
        x = x_test[i, :]
        x = x.reshape(28, 28)
        plt.pcolor(1 - x)
        wk = y[i, :]
        prediction = np.argmax(wk)
        plt.text(22, 25.5, "%d" % prediction, fontsize=12)
        if prediction != np.argmax(y_test[i, :]):
            plt.plot([0, 27], [1, 1], color='cornflowerblue', linewidth=5)
        plt.xlim(0, 27)
        plt.ylim(27, 0)
        plt.xticks([], "")
        plt.yticks([], "")

# -- メイン
show_prediction()
plt.show()


# リスト 8-1-(10)
# パラメータの可視化
#👇layers[0]にすると隠れ層、layers[1]にすると出力層
#👇get_weights()[0]にすると重みのパラメータ、get_weights()[1]にするとバイアスパラメータを参照できる
parameter_num = 10 #パラメータの数
w = model.layers[0].get_weights()[1]
#print(w.ndim)
#print(w.shape[0])
#print(w.shape[1])
print(w)

plt.figure(1, figsize=(12, 3)) #figure（全体）の大きさ？
plt.gray() # ウィンドウをグレースケールモードにする。
plt.subplots_adjust(wspace=0.35, hspace=0.5) #グラフ間の余白の設定
for i in range(parameter_num): # 👈do i = 0 , parameter_num - 1
    plt.subplot(2, 8, i + 1) #subplot(行数, 列数, プロット番号)
    w1 = w[:, i] #
    w1 = w1.reshape(28, 28)
    plt.pcolor(-w1)
    plt.xlim(0, 27)
    plt.ylim(27, 0)
    plt.xticks([], "")
    plt.yticks([], "")
    plt.title("%d" % i)
# end 'for' Pythonは字下げを止めるだけでループが閉じるらしい？😅
plt.show()
"""
