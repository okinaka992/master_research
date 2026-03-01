# ann_learn2.pyを改良したプログラム
# 並列化によって、学習が上手くいっていないかを確認するために、並列をやめたプログラム
# cd workspace/research2/experiment
#　実行コマンド：　python3 ann_learn7.py

import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.layers import LeakyReLU
from tensorflow.keras.utils import get_custom_objects
from keras import optimizers
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay  # 学習率の調整

import matplotlib
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time
import pathlib
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

import sys
import argparse
import os
import multiprocessing
from multiprocessing import Process



#グローバル変数
cir = 's344'  # 対象回路
input_data_file = cir + 'input'  # 入力データファイル
correct_data_file = cir + '分割正解データ' + '/' + cir + 'integrated_output'  # 統合後の正解データファイル
suplit_num_file = cir + '分割正解データ' + '/' + cir + 'suplit_num'  # 分割数が保存されたファイル
input_data_num = None  # 1個のモデルにおける学習データ数
input_node_num = None  #入力層におけるノード数 初期値は8　学習結果によって変更
mid_node_num = 256  #生成器の中間層のノード数　初期値は128　学習結果によって変更
# mid_node_num2 = 256  #生成器の中間層のノード数　初期値は128　学習結果によって変更
output_node_num = None #　出力層のノード数＝分割数による
num_models = None  # 学習させるモデルの数
# learning_rate = 0.0005
# dropout_rate = 0.2
epochs = 1000
batch_size = 4
model_folder = cir + 'sepmodel'  # 学習済みモデルを保存するフォルダ

processes = 8  # 並列処理のプロセス数


def mk_input_data():
    # 入力データファイルを開いてデータを読み込む
    with open(input_data_file) as f:
        lines = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]

    # print(lines)
    
    global input_data_num
    input_data_num = int(len(lines))      #学習データ数を設定。学習データ数は入力データの行数
    global input_node_num               #グローバル変数を書き換え
    input_node_num = int(len(lines[0]))      #入力ノード数を設定。入力ノード数は入力データの各行の要素数

    int_lines = [list(map(int, _)) for _ in lines]  #list要素の型をint型に変換

    return np.array(int_lines)


def mk_output_data(fname):
    # 正解データファイルを開いてデータを読み込む
    with open(fname) as f:
        lines = [_.replace("\n", "") for _ in f.readlines()]
    
    lines = [value.split(",") for value in lines] # カンマ区切りで各信号線をリストに格納
    
    # print("dsadsa")
    # print(lines[0])

    for i in range(len(lines)):
        for j in range(len(lines[i])):
            lines[i][j] = float(lines[i][j])
    
    # print("gaga")
    # print(lines[0])
    
    global output_node_num               #グローバル変数を書き換え
    output_node_num = int(len(lines[0]))      #出力ノード数を設定。出力ノード数は正解データの各行の要素数

    int_lines = [list(map(float, _)) for _ in lines]  #list要素の型をfloat型に変換

    return np.array(int_lines)


# モデルの学習を行う関数
def build_and_train_model(input_data, correct_data, model_id):
    print('モデル' + str(model_id) + 'の学習を開始します')
    print(correct_data)

    # モデルの構築
    model = Sequential()
    model.add(Dense(mid_node_num, input_dim=input_node_num, activation='relu'))
    # model.add(Dropout(dropout_rate))  # dropout_rateの割合でノードを無効化
    # model.add(Dense(mid_node_num2, activation='relu'))
    model.add(Dense(output_node_num, activation='sigmoid'))

    model.summary()

     # 学習率スケジューリングの設定
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,  # 初期学習率
        decay_steps=10000,           # 学習率が減衰する間隔
        decay_rate=0.96,             # 減衰率
        staircase=True               # 階段的に減衰
    )

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy'])
    
    import time
    t1 = time.time()

    # モデルの学習
    result = model.fit(input_data, correct_data, epochs=epochs, validation_data=(input_data, correct_data), batch_size=batch_size)

    t2 = time.time()
    training_time = t2-t1
    print('モデル' + str(model_id) + ' 学習時間：' + str(training_time) + '秒')

    # 学習結果の損失を可視化
    plt.plot(result.history['loss'], label='Training Loss')
    plt.plot(result.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(cir + 'model' + str(model_id) + '_loss.png')
    plt.show()

    # 学習結果の表示
    model_output = model.predict(input_data)
    with open(cir + 'model' + str(model_id) + '_before_save_output.txt', 'w') as f:
        f.write(np.array2string(model_output, threshold=np.inf))  # 省略を防ぐために threshold を無限大に設定
        f.write('\n')

    plt.plot(range(1, epochs+1), result.history['accuracy'], label="training")
    plt.plot(range(1, epochs+1), result.history['val_accuracy'], label="validation")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # tfliteでモデルの保存
     #学習したモデルを保存
    #to tensorflow lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)   #KerasモデルをTFLite変換用のコンバータを作成。from_keras_model メソッドは、Kerasモデルオブジェクトを入力として受け取り、そのモデルをTFLite形式に変換するための設定や操作を行えるコンバータを返します
    converter.optimizations = [tf.lite.Optimize.DEFAULT]     # モデルのサイズの削減や性能を向上させるための最適化を有効化。⇒TFLiteモデルのサイズを縮小したり、推論速度を向上させたりします。
    # converter.target_spec.supported_types = [tf.float16]     # TFLiteモデルで16ビット浮動小数点（FP16）を使用する設定を適用。⇒モデルサイズ、推論時のメモリ使用量の削減

    tflite_model = converter.convert()     #これまで設定したコンバータオブジェクト (converter) を使用して、TFLite形式のモデルを生成。converter.convert() の戻り値は、TFLite形式に変換されたモデルをバイナリデータとして返します。このデータはメモリ上に存在し、後でファイルに保存することができます。

    # モデルの保存
    with open(model_folder + '/' + cir + 'model_' + str(model_id) + '.tflite', 'wb') as f:  # モデルを保存するファイルを開く. wbはバイナリ書き込みモード
        f.write(tflite_model)


def ann_evaluation(input_data, correct_data, model_id):
    # モデルの読み込み
    with open(model_folder + '/' + cir + 'model_' + str(model_id) + '.tflite', 'rb') as f:  # モデルを読み込むファイルを開く
        tflite_model = f.read()

    # モデルの評価
    interpreter = tf.lite.Interpreter(model_content=tflite_model)  # TFLite形式のモデルを読み込む。保存されたTFLiteモデルをメモリに読み込み、推論を行う準備をするためのインタープリターを作成します。
    interpreter.allocate_tensors()  # #メモリを確保。モデルが使用するテンソル（データ構造）をメモリに割り当てます。TFLiteモデルをインタープリターにロードするだけでは、テンソルのメモリは割り当てられていません。この行を実行することで、モデルが推論に必要なメモリを確保します。

    input_details = interpreter.get_input_details()  # モデルの入力テンソルの詳細を取得。モデルの入力に関する詳細情報をリスト形式で返します。各要素は、入力テンソルの形状、データ型、名前などの情報を含む辞書です。
    output_details = interpreter.get_output_details()  # モデルの出力テンソルの詳細を取得。モデルの出力に関する詳細情報をリスト形式で返します。各要素は、出力テンソルの形状、データ型、名前などの情報を含む辞書です。
    
    correct_num = 0  # 正解数
    # output_data2 = []
    for i in range(input_data_num):
        if model_id == 0:
            print(i)
        
        input_shape = input_details[0]['shape']  # 入力データの形状を取得.先ほど取得したinput_detailsのリストの中から、形状情報を取得。input_details[0]['shape']は、入力テンソルの形状を取得するためのコードです。['shape']は、辞書内の shape キーを指定しています。

        reshape_input_data = np.reshape(input_data[i], input_shape)  # NumPy配列の形状を指定した形 (input_shape) に変更します。これにより、入力データの形状がモデルの入力テンソルの形状と一致します。


        # モデルの入力データを設定
        interpreter.set_tensor(input_details[0]['index'], np.array(reshape_input_data, dtype=np.float32))   # モデルの入力テンソルにデータを設定します。入力テンソルのインデックス、データを指定します。[0]['index']は、入力テンソルのインデックスを取得するためのコードです。

        # モデルの推論を実行
        interpreter.invoke()

        # モデルの出力データを取得
        output_data = interpreter.get_tensor(output_details[0]['index'])
        #output_dataをoutput_data2に追加
        # output_data2 = np.append(output_data2, output_data)
        # print("aaaa")
        # print(type(output_data))
        # print(type(correct_data))

        # with open(cir + 'model' + str(model_id) + '_output.txt', 'a') as f:
        #     f.write(str(output_data) + '\n')
        #     if i == input_data_num - 1:
        #         f.write('\n')
        
        # print("11111111")
        # print(output_data)
        # print("22222222")
        # print(output_data[0][0], output_data[0][1], output_data[0][2], output_data[0][3])

        count = 0
        for j in range(output_node_num):
            if output_data[0][j] < 0.25:
                output_data[0][j] = 0
            elif output_data[0][j] < 0.5:
                output_data[0][j] = 0.375
            elif output_data[0][j] < 0.75:
                output_data[0][j] = 0.625
            else:
                output_data[0][j] = 1
            
            # print("bbbb")
            # print(type(output_data))
            # print("fdaffdsa")
            # print(float(output_data[0][j]))
            # print(float(correct_data[i][j]))
            
            if correct_data[i][j] == output_data[0][j]:
                count += 1
        
        with open(cir + 'model' + str(model_id) + '_savecorrect_data.txt', 'a') as f:
            f.write(str(correct_data[i]) + '\n')
            if i == input_data_num - 1:
                f.write('\n')
        
        # 正解率
        if count == output_node_num:
            correct_num += 1
        
        output_data = np.array(output_data)

    # print(output_data2[1][4])
    # output_data3 = [0] * input_data_num
    # for i in range(input_data_num):
    #     for value in output_data2[i]:
    #         print(output_data2[i])
    #         for j in range(output_node_num):
    #             if value < 0.25:
    #                 output_data3[i][j] = 0
    #             elif value < 0.5:
    #                 output_data3[i][j] = 0.375
    #             elif value < 0.75:
    #                 output_data3[i][j] = 0.625
    #             else:
    #                 output_data3[i][j] = 1
    
    # with open(cir + 'model' + str(model_id) + '_extraoutput.txt', 'w') as f:
    #     f.write(str(output_data3) + '\n')
    #     f.write('\n')

    # モデルの評価結果
    print('正解数：' + str(correct_num))
    print('入力データ数：' + str(input_data_num))
    accuracy = correct_num / input_data_num
    # precision = precision_score(correct_data, output_data, average='macro')
    # recall = recall_score(correct_data, output_data, average='macro')
    # f1 = f1_score(correct_data, output_data, average='macro')

    print('モデル' + str(model_id) + 'の評価結果')
    print('Accuracy: ' + str(accuracy))
    with open(cir + 'model_accuracy.txt', 'a') as f:
        f.write('モデル' + str(model_id) + 'の正解率：' + str(accuracy) + '\n')

    # print('Precision: ' + precision)
    # print('Recall: ' + recall)
    # print('F1: ' + f1)

    # 混同行列
    # cm = confusion_matrix(correct_data, output_data)
    # print('Confusion Matrix: ')
    # print(cm)

    return accuracy
    

def worker(model_id):
    correct_file = correct_data_file + str(model_id)  # 分割された正解データファイル名に変更　＝　s344分割正解データ/s344integrated_output + 番号

    input_data = mk_input_data()
    correct_data = mk_output_data(correct_file)
    print(model_id, correct_data)

    build_and_train_model(input_data, correct_data, model_id)

    accuracy = ann_evaluation(input_data, correct_data, model_id)

    return accuracy


if __name__ == '__main__':


    # 学習させるモデルの数=分割されたデータの数を取得＝学習させるモデルの数
    with open(suplit_num_file, 'r') as f:
        num_models = int(f.readline())
        # print(suplit_num)

    p = multiprocessing.Pool(processes=processes)     # 5つのプロセスを使って並列処理を行うプールを作成。processesは、cpuのコア数
    result = p.map(worker, range(0, num_models))         #worker関数を引数の範囲でマップし、それぞれの値に対して関数を実行します。
    p.close()   #処理を終了
    p.join()    #プロセスが終了するまで待つ
    with open(cir + 'model_accuracy.txt', 'a') as f:
        f.write('\n')
    print(result)

