import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

import requests
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

import os
import cv2
import random
from sklearn.model_selection import StratifiedKFold
import keras
from sklearn.model_selection import KFold

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
DATADIR = "/Users/e195718/template/data/carpet2"
CATEGORIES = {"good", "scratch"}
IMG_SIZE = 384
training_data = []


def create_training_data():
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        print(path)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name),)  # 画像読み込み
                img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 画像のリサイズ
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)  # データをシャッフル
X_train = []  # 画像データ
y_train = []  # ラベル情報

# データセット作成
for feature, label in training_data:
    X_train.append(feature)
    y_train.append(label)

# numpy配列に変換
X_train = np.array(X_train)
y_train = np.array(y_train)


for i in range(0, 4):
    print("学習データのラベル：", y_train[i])
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.title(label = 'scratch' if y_train[i] == 1 else 'good')
    img_array = cv2.cvtColor(X_train[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img_array)

plt.show()
X_train = np.asarray(X_train) / 255


train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size=0.1, random_state=0)


pixels = 384
IMAGE_SIZE = (pixels, pixels)
dynamic_size = False

model_name = "efficientnetv2-s-21k"
model_handle = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/classification/2"


do_fine_tuning = True
print("Building model with", model_handle)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(model_handle, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(2,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01),activation='softmax')
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()


model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])


history = model.fit(train_X, train_y)


test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)

print(test_loss)
print(test_acc)


metrics = ['loss', 'accuracy']  # 使用する評価関数を指定

plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意

for i in range(len(metrics)):

    metric = metrics[i]

    plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.title(metric)  # グラフのタイトルを表示
    
    plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
    plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す
    
    plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
    plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
    plt.legend()  # ラベルの表示
    
plt.show()  # グラフの表示

metrics = ['loss', 'accuracy']  # 使用する評価関数を指定
 
plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意
 
for i in range(len(metrics)):
 
    metric = metrics[i]
 
    plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.title(metric)  # グラフのタイトルを表示
    
    plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
    plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す
    
    plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
    plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
    plt.legend()  # ラベルの表示
    
plt.show()