# ann_learn5.pyを改良したプログラム
# 並列化によって、学習が上手くいっていないかを確認するために、並列をやめたプログラム
#　学習データの離散化を行った
# cd workspace/research2/experiment
#　実行コマンド：　python3 ann_learn6.py

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
mid_node_num = 128  #生成器の中間層のノード数　初期値は128　学習結果によって変更
# mid_node_num2 = 256  #生成器の中間層のノード数　初期値は128　学習結果によって変更
output_node_num = None #　出力層のノード数＝分割数による
num_models = None  # 学習させるモデルの数
learning_rate = 0.0005
dropout_rate = 0
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

    # データの離散化 (例: 値を10区分に分ける)
    discrete_data = np.array(int_lines)
    max_val = discrete_data.max()  # データの最大値
    min_val = discrete_data.min()  # データの最小値
    bins = np.linspace(min_val, max_val, num=10)  # 区分の境界を作成
    discrete_data = np.digitize(discrete_data, bins)  # 値を区分に割り当て

    print("discrete_data")
    np.set_printoptions(threshold=np.inf)
    print(discrete_data)


    return discrete_data


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

    float_lines = [list(map(float, _)) for _ in lines]  # list要素の型をfloat型に変換

    # データの離散化 (例: 値を4区分に分ける)
    discrete_data = np.array(float_lines)
    bins = [0, 0.25, 0.5, 0.75, 1.0]  # 区分の境界を作成
    discrete_data = np.digitize(discrete_data, bins, right=True)  # 値を区分に割り当て

    # print("discrete_data")
    # np.set_printoptions(threshold=np.inf)
    # print(discrete_data)


    return discrete_data


# モデルの学習を行う関数
def build_and_train_model(input_data, correct_data, model_id):
    print('モデル' + str(model_id) + 'の学習を開始します')
    print(correct_data)

    # モデルの構築
    model = Sequential()
    model.add(Dense(mid_node_num, input_dim=input_node_num, activation='relu'))
    # model.add(Dense(mid_node_num2, activation='relu'))
    model.add(Dense(output_node_num, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    
    import time
    t1 = time.time()

    # モデルの学習
    result = model.fit(input_data, correct_data, epochs=epochs, validation_data=(input_data, correct_data), batch_size=batch_size)

    t2 = time.time()
    training_time = t2-t1
    print('モデル' + str(model_id) + ' 学習時間：' + str(training_time) + '秒')

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
    with open(model_folder + '/' + cir + 'model_' + str(model_id) + '.tflite', 'rb') as f:
        tflite_model = f.read()

    # モデルの評価
    interpreter = tf.lite.Interpreter(model_content=tflite_model)  # TFLite形式のモデルを読み込む
    interpreter.allocate_tensors()  # メモリを確保

    input_details = interpreter.get_input_details()  # 入力テンソルの詳細を取得
    output_details = interpreter.get_output_details()  # 出力テンソルの詳細を取得
    
    correct_num = 0  # 正解数
    total_samples = len(input_data)  # 入力データ数

    for i in range(total_samples):
        # 入力データの形状を取得
        input_shape = input_details[0]['shape']
        reshape_input_data = np.reshape(input_data[i], input_shape)  # 入力データをモデルの形状にリシェイプ

        # モデルの入力データを設定
        interpreter.set_tensor(input_details[0]['index'], np.array(reshape_input_data, dtype=np.float32))

        # モデルの推論を実行
        interpreter.invoke()

        # モデルの出力データを取得
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 出力データの離散化
        bins = [0, 0.25, 0.5, 0.75, 1.0]  # 区分の境界
        discrete_output = np.digitize(output_data, bins, right=True)

        # 正解データの離散化
        discrete_correct = np.digitize(correct_data[i], bins, right=True)

        # 一致しているかを確認
        if np.array_equal(discrete_output, discrete_correct):
            correct_num += 1

    # 正解率の計算
    accuracy = correct_num / total_samples

    # 結果の表示
    print(f"モデル{model_id}の評価結果")
    print(f"正解数：{correct_num}")
    print(f"入力データ数：{total_samples}")
    print(f"Accuracy: {accuracy:.4f}")

    # 結果をファイルに保存
    with open(cir + 'model_accuracy.txt', 'a') as f:
        f.write(f'モデル{model_id}の正解率：{accuracy:.4f}\n')

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

    # p = multiprocessing.Pool(processes=processes)     # 5つのプロセスを使って並列処理を行うプールを作成。processesは、cpuのコア数
    result = worker(1)         #worker関数を引数の範囲でマップし、それぞれの値に対して関数を実行します。
    # p.close()   #処理を終了
    # p.join()    #プロセスが終了するまで待つ
    # with open(cir + 'model_accuracy.txt', 'a') as f:
    #     f.write('\n')
    print(result)

