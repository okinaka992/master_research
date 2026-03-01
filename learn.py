#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import logging
#logging.getLogger("tensorflow").setLevel(logging.DEBUG)
import random
import matplotlib
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time
import pathlib
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers import LeakyReLU
from tensorflow.keras.utils import get_custom_objects
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import sys
import argparse
import os
import multiprocessing
from multiprocessing import Process

tf.config.threading.set_intra_op_parallelism_threads(1)

epochs = 1100
learning_rate = 0.0005
dropout_rate = 0

def learn(i):

    f_n_y = 's1494sep1000/s1494out{0}'.format(n1[i])

    #入力と正解の用意
    f = open('s1494indata')
    x_train=np.array([list(map(int,line.rstrip().split(","))) for line in f.readlines()], dtype=np.float16)      #入力
    f.close()
    f1 = open(f_n_y)
    y_train=np.array([list(map(int,line1.rstrip().split(","))) for line1 in f1.readlines()], dtype=np.float16)    #正解
    f1.close()

    n_out = len(y_train[0])
    #print(n_out)

    #各設定値
    for line in open("s1494info", "r"):
        data = line.split()
        n_in = int(data[0])       #入力層のニューロン数
        n_data = int(data[2])    #学習用データ数

    #"""

    model = Sequential()


    model.add(Dense(100, activation='tanh', input_dim=n_in))
    model.add(Dense(150, activation='tanh')) # 隠れ層
    model.add(Dense(n_out, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    t1 = time.time()

    result = model.fit(x_train, y_train, 
                       verbose=0,
                       epochs=epochs,
                       validation_data=(x_train, y_train),
                       batch_size=4)

    t2 = time.time()
    time1 = t2-t1
    print('  学習時間  ' + str(round(time1, 2)) +'秒')
    
    plt.plot(range(1, epochs+1), result.history['accuracy'], label="training")
    plt.plot(range(1, epochs+1), result.history['val_accuracy'], label="validation")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    m_n_t = 's1494sepmodel/s1494{0}.tflite'.format(i)
    
    export_dir = "/tmp/test_saved_model"

    #学習したモデルを保存
    #to tensorflow lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)   #KerasモデルをTFLite変換用のコンバータを作成。from_keras_model メソッドは、Kerasモデルオブジェクトを入力として受け取り、そのモデルをTFLite形式に変換するための設定や操作を行えるコンバータを返します
    converter.optimizations = [tf.lite.Optimize.DEFAULT]     # モデルのサイズの削減や性能を向上させるための最適化を有効化。⇒TFLiteモデルのサイズを縮小したり、推論速度を向上させたりします。
    converter.target_spec.supported_types = [tf.float16]     # TFLiteモデルで16ビット浮動小数点（FP16）を使用する設定を適用。⇒モデルサイズ、推論時のメモリ使用量の削減

    s1494tflite_model = converter.convert()     #これまで設定したコンバータオブジェクト (converter) を使用して、TFLite形式のモデルを生成。converter.convert() の戻り値は、TFLite形式に変換されたモデルをバイナリデータとして返します。このデータはメモリ上に存在し、後でファイルに保存することができます。

    tflite_models_dir = pathlib.Path("/tmp/test_saved_model")  # pathlib.Path を使って、/tmp/test_saved_model というディレクトリパスを表す Path オブジェクトを作成します。このディレクトリは、TFLiteモデルを保存するためのディレクトリです。
    tflite_models_dir.mkdir(exist_ok=True, parents=True)   # mkdir() メソッドを使って、ディレクトリを作成します。exist_ok=True は、ディレクトリが既に存在している場合にエラーを発生させないようにするためのオプションです。parents=True は、必要なすべての親ディレクトリを自動的に作成するためのオプションです。

    tflite_model_file = tflite_models_dir/"saved_model.tflite"  # tflite_models_dir ディレクトリ内の "saved_model.tflite" というファイルパスを表す Path オブジェクトを作成.この例では、/tmp/test_saved_model/saved_model.tflite というパスを表します
    tflite_model_file.write_bytes(s1494tflite_model)  # s1494tflite_model（バイナリ形式のTFLiteモデルデータ）を tflite_model_file に書き込みます。これにより、TFLiteモデルがファイルとして保存されます。

    open(m_n_t, 'wb').write(s1494tflite_model)   #学習済みのTFLiteモデル（s1494tflite_model）をファイルに保存.この場合、ファイル名は m_n_t になります。(バイナリ書き込みモード (wb) )

    interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_file))  #　保存されたTFLiteモデルをメモリに読み込み、推論を行う準備をするためのインタープリターを作成します。
    interpreter_fp16.allocate_tensors() #メモリを確保。モデルが使用するテンソル（データ構造）をメモリに割り当てます。TFLiteモデルをインタープリターにロードするだけでは、テンソルのメモリは割り当てられていません。この行を実行することで、モデルが推論に必要なメモリを確保します。
    
    input_details = interpreter_fp16.get_input_details() #入力層の情報を取得。モデルの入力に関する詳細情報をリスト形式で返します。各要素は、入力テンソルの形状、データ型、名前などの情報を含む辞書です。
    output_details = interpreter_fp16.get_output_details() #出力層の情報を取得。モデルの出力に関する詳細情報をリスト形式で返します。各要素は、出力テンソルの形状、データ型、名前などの情報を含む辞書です。
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for k in range(n_data):
        input_shape = input_details[0]['shape']

        input_data = np.reshape(x_train[k], input_shape)

        interpreter_fp16.set_tensor(input_details[0]['index'], np.array(input_data, dtype=np.float32))

        interpreter_fp16.invoke()
        output_data = interpreter_fp16.get_tensor(output_details[0]['index'])

        threshold = 0.5
        output_data2 = (output_data > threshold).astype(np.int)

        #print(output_data2)

        tp += np.sum((np.array(y_train[k])==1)&(np.array(output_data2)==1))
        tn += np.sum((np.array(y_train[k])==0)&(np.array(output_data2)==0))
        fp += np.sum((np.array(y_train[k])==0)&(np.array(output_data2)==1))
        fn += np.sum((np.array(y_train[k])==1)&(np.array(output_data2)==0))          
    confusion_matrix1 = np.array([[tp, fp],
                                 [fn, tn]])
    print(confusion_matrix1)
    print('正解率' + str((tp+tn)/(tp+tn+fp+fn)))
    print('適合率' + str(tp/(tp+fp)))
    print('再現率' + str(tp/(tp+fn)))
    print('  F値 ' + str(2*tp/(2*tp+fp+fn)))
    
    return n1[i], (tp+tn)/(tp+tn+fp+fn), tp/(tp+fn)
    
if __name__ == '__main__':

    with multiprocessing.Pool(processes=5) as pool:     #//5つのプロセスを使って並列処理を行うプールを作成
        r = pool.map(learn, range(0, 100, 1))          #learn関数を引数の範囲（0から99までの数値）でマップし、それぞれの値に対して関数を実行します。この場合、learn 関数は学習を行い、結果を返します
        print(r)
