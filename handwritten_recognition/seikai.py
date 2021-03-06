import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np
# MNISTデータのダウンロード
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

NUM_CLASSES = 10    # モデルのクラス数

sess = tf.InteractiveSession()


def interence(imegs_placeholder, keep_prob):
    """ 予測モデルを作成する関数

    引数:
      images_placeholder: 画像のplaceholder
      keep_prob: dropout率のplaceholder

    返り値:
      y_conv: 各クラスの確率(のようなもの)

     with tf.name_scope("xxx") as scope:
         これでTensorBoard上に一塊のノードとし表示される
    """


    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
        inital = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(inital)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital)

    # 畳み込み層
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

    # プーリング層
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # 入力層を28*28*1に変形
    x_image = tf.reshape(imegs_placeholder, [-1, 28, 28, 1])

    # 畳み込み層1の作成
    with tf.name_scope("conv1") as scope:
        W_conv1 = weight_variable([3,3,1,16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # 畳み込み層2の作成
    with tf.name_scope("conv2") as scope:
        W_conv2 = weight_variable([3,3,16,16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # プーリング層1の作成
    with tf.name_scope("pool1") as scope:
        h_pool1 = max_pool_2x2(h_conv2)

    # 畳み込み層3の作成
    with tf.name_scope("conv3") as scope:
        W_conv3 = weight_variable([3,3,16,32])
        b_conv3 = bias_variable([32])
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    # 畳み込み層4の作成
    with tf.name_scope("conv4") as scope:
        W_conv4 = weight_variable([3,3,32,32])
        b_conv4 = bias_variable([32])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    # プーリング層2の作成
    with tf.name_scope("pool2") as scope:
        h_pool2 = max_pool_2x2(h_conv4)

    # 畳み込み層5の作成
    with tf.name_scope("conv5") as scope:
        W_conv5 = weight_variable([3,3,32,64])
        b_conv5 = bias_variable([64])
        h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

    # 畳み込み層6の作成
    with tf.name_scope("conv6") as scope:
        W_conv6 = weight_variable([3,3,64,64])
        b_conv6 = bias_variable([64])
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    # プーリング層3の作成
    with tf.name_scope("pool3") as scope:
        h_pool3 = max_pool_2x2(h_conv6)

    # 結合層1の作成
    with tf.name_scope("fc1") as scope:
        W_fc1 = weight_variable([4*4*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        # dropout1の設定
        h_fc_1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 結合層2の作成
    with tf.name_scope("fc2") as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数による正規化
    with tf.name_scope("softmax") as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv

def loss(logits, labels):
    """ lossを計算する関数
    引数:
      logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      cross_entropy: 交差エントロピーのtensor, float
    """
    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    # TensorBoardで表示するように指定
    tf.summary.scalar("cross_entropy", cross_entropy)   #v0.12から tf.summary.scalar() になった   # 参照: https://teratail.com/questions/68531
    return cross_entropy

def training(loss, learning_rate):
    """ 訓練のopを定義する関数

    引数:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数

    返り値:
      train_step: 訓練のop

    """

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    """正解率(accuracy)を計算する関数
    引数:
        logits: inference() の結果
        labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    返り値:
        accuracy: 正解率(float)
    """

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

if __name__=="__main__":
    with tf.Graph().as_default():
        # 式に用いる変数設定
        x_image = tf.placeholder("float", shape=[None, 784])    # 入力
        y_label = tf.placeholder("float", shape=[None, 10])
        W = tf.Variable(tf.zeros([784,10]))
        b = tf.Variable(tf.zeros([10]))
        #y_label = tf.nn.softmax(tf.matmul(x_image,W)+b)     # y=softmax(Wx+b)微分も勝手に行ってくれる
        keep_prob = tf.placeholder("float")
        #init_op = tf.global_variables_initializer()    # 変数の初期化(変数使用の際必ず必要)
        logits = interence(x_image, keep_prob) # inference()を呼び出してモデルを作成
        loss_value = loss(logits, y_label)     # loss()を呼び出して損失を計算
        train_op = training(loss_value,1e-4)   # training()を呼び出して訓練（1e-4は学習率）
        accur = accuracy(logits, y_label)      # accuracy()を呼び出して精度を計算
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        # TensorBoardで表示する値の設定
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("./tmp/data", graph=sess.graph)

        saver = tf.train.Saver()    # 全ての変数を保存して復元するらめの OP を追加

        init = tf.global_variables_initializer()
        sess.run(init) #変数を初期化して実行

        # 訓練の実行
        for step in range(20000)
            batch = mnist.train.next_batch(50)
            if step % 100 == 0:
                train_accury = sess.run(accur, feed_dict={x_image: batch[0], y_label: batch[1], keep_prob: 1.0})
                print("step%d, train_accury : %g"%(step, train_accury))
            sess.run(train_op, feed_dict={x_image: batch[0], y_label: batch[1], keep_prob:0.5})
            # 1step終わるたびにTensorBoardに表示する値を追加する
            summary_str = sess.run(summary_op, feed_dict={x_image: batch[0], y_label: batch[1], keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        #結果表示
        print("test accuracy : %g" %sess.run(accur, feed_dict={x_image: mnist.test.images, y_label: mnist.test.labels, keep_prob: 1.0}))
        saver.save(sess, ".\ckpt\model.ckpt") # 変数データ保存
