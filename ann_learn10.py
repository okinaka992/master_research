# ipythonを用いて、学習を行うプログラム
# ipythonは並列処理に対応しておらず、学習途中で停止してしまうのでモデルをひとつづつ学習させる
# また、GPUを使用しているため、その効果を最大限引き出すために、バッチサイズを大きく設定している
# cd workspace/research2/experiment
#　実行コマンド：　python3 ann_learn2.py

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
cir = 's1488'  # 対象回路
input_data_file = cir + 'input'  # 入力データファイル
correct_data_file = cir + '分割正解データ' + '/' + cir + 'integrated_output'  # 統合後の正解データファイル
suplit_num_file = cir + '分割正解データ' + '/' + cir + 'suplit_num'  # 分割数が保存されたファイル
single_line_file = cir + '分割正解データ' + '/' + cir + 'single_line'  # 統合されていない信号線があるかが保存されたファイル
learning_data_file = 'learning_data/' + cir + '_learning_data/' # 学習データが保存されたファイル
error_output_file = 'error_output/' + cir + 'error_output/'  # エラー出力ファイル
input_data_num = None  # 1個のモデルにおける学習データ数
input_node_num = None  #入力層におけるノード数 初期値は8　学習結果によって変更
mid_node_num = 100  #生成器の中間層のノード数　初期値は128　学習結果によって変更
mid_node_num2 = 200  #生成器の中間層のノード数　初期値は128　学習結果によって変更
output_node_num = None #　出力層のノード数＝分割数による
num_models = None  # 学習させるモデルの数
single_flag = None # 統合されていない信号線があるかどうかを示す # 0:存在しない, 1:存在する
learning_rate = 0.001
dropout_rate = 0
epochs = 1200
batch_size = 128
model_folder = cir + 'sepmodel'  # 学習済みモデルを保存するフォルダ
# correct_value = [0.00, 0.25, 0.75, 1.00]  # 正解データの値・種類
# correct_value = [-1.00, -0.40, 0.40, 1.00]  # 出力層の活性化関数がtanhの場合の正解データの値（0~1ではなく、-1~1に変換する必要があるため） # 0->-1, 1->-0.5, 2->0.5, 3->1
correct_value = [0, 1, 2, 3]
# threshold = [0.02, 0.5, 0.98]  # ANNの出力値を変換するための閾値
# threshold = [-0.75, 0, 0.75]
threshold = [0.5, 1.5, 2.5]

processes = 9  # 並列処理のプロセス数


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
    model.add(Dense(mid_node_num, input_dim=input_node_num, activation='relu'))  # 入力層と中間層1
    model.add(Dense(mid_node_num2, activation='relu'))  # 中間層2
    model.add(Dense(output_node_num, activation='linear'))  # sigmoid or linear

    model.summary()

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    # loss = 'binary_crossentropy'  # 二値分類の場合の損失関数
    # loss = 'mean_squared_error'  # 回帰問題の場合の損失関数
    
    import time
    t1 = time.time()

    # モデルの学習
    result = model.fit(input_data, correct_data, epochs=epochs, validation_data=(input_data, correct_data), batch_size=batch_size, verbose=1)

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
    plt.savefig(learning_data_file + cir + 'model' + str(model_id) + '_loss.png')
    plt.show()

    # 学習結果の正解率を可視化
    plt.plot(result.history['accuracy'], label='Training accuracy')
    plt.plot(result.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(learning_data_file + cir + 'model' + str(model_id) + '_accuracy.png')
    plt.show()

    # 学習結果の表示
    model_output = model.predict(input_data)
    with open(learning_data_file + cir + 'model' + str(model_id) + '_before_save_output.txt', 'w') as f:
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
    model_output_data = [0 for _ in range(input_data_num)]  # モデルの出力データを格納するリスト
    # output_data2 = []
    f = open("s344_model_outputa.txt", 'w')  # 正解データファイルを開く
    g = open(error_output_file + cir + 'model' + str(model_id) + '_error_output.txt', 'w')  # エラー出力ファイルを開く
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

        convert_output_data = output_data.copy()
        if (model_id == (num_models - 1)) and single_flag == 1:  # 最後のモデルで、統合されていない信号線がある場合
            for j in range(output_node_num):
                if j < (output_node_num - 1):  # 最後の出力ノード以外の場合
                    if convert_output_data[0][j] < threshold[0]:  # 閾値を使って出力データを変換
                        convert_output_data[0][j] = correct_value[0]  # 0に変換
                    elif convert_output_data[0][j] < threshold[1]:
                        convert_output_data[0][j] = correct_value[1]
                    elif convert_output_data[0][j] < threshold[2]:
                        convert_output_data[0][j] = correct_value[2]
                    else:
                        convert_output_data[0][j] = correct_value[3]
                else:  # 統合されていない信号線がある場合、最後の出力ノードは統合されていない信号線の値を持つため、それは₀または1の値を持つため、0.5を閾値として0または1に変換する
                    if convert_output_data[0][j] <= 0.5:
                        convert_output_data[0][j] = correct_value[0]  # 0に変換
                    else:
                        convert_output_data[0][j] = correct_value[3]
                f.write(str(convert_output_data[0][j]) + " ")  # 出力データをファイルに書き込む
        else:  # 最後のモデルで、統合されていない信号線がない場合、または最後のモデル以外の場合
            for j in range(output_node_num):
                if convert_output_data[0][j] < threshold[0]:  # 閾値を使って出力データを変換
                    convert_output_data[0][j] = correct_value[0]  # 0に変換
                elif convert_output_data[0][j] < threshold[1]:
                    convert_output_data[0][j] = correct_value[1]
                elif convert_output_data[0][j] < threshold[2]:
                    convert_output_data[0][j] = correct_value[2]
                else:
                    convert_output_data[0][j] = correct_value[3]
                f.write(str(convert_output_data[0][j]) + " ")  # 出力データをファイルに書き込む
        
        f.write("\n")  # 出力データをファイルに書き込む
            
        # print("bbbb")
        # print(convert_output_data)
        # print(type(convert_output_data))
        # print("fdaffdsa")
        # print(float(convert_output_data[0][j]))
        # print(float(correct_data[i][j]))

        
        if i == 0:
            print("convert_output_data：", convert_output_data[0])
        
        model_output_data[i] = convert_output_data[0]  #モデルの出力データを格納

        # with open(learning_data_file + cir + 'model' + str(model_id) + '_savecorrect_data.txt', 'a') as f:
        #     f.write(str(correct_data[i]) + '\n')
        #     if i == input_data_num - 1:
        #         f.write('\n')
        
        # if model_id == 0:
        #     print("正解データ：", correct_data[i])
        #     print("出力データ：", convert_output_data[0])
        #     if np.allclose(correct_data[i], convert_output_data[0], atol=1e-6): #numpy配列同士の比較
        #         print("正解")
        #     else:
        #         print("不正解")

         # 正解率
        if np.allclose(correct_data[i], convert_output_data[0], atol=1e-6): #numpy配列同士の比較
            correct_num += 1
        else:
            if epochs > 500:
                g.write('正解データ：' + str(correct_data[i]) + '\n')
                g.write('変換後出力データ：' + str(convert_output_data[0]) + '\n')
                g.write('変換前出力データ：' + str(output_data[0]) + '\n')
                g.write('\n')
        # else:
        #     if model_id == 0:
        #         print("正解データと一致しませんでした：", i, convert_output_data[0])
        
        convert_output_data = np.array(convert_output_data)
    
    f.close()  # 出力データファイルを閉じる
    g.close()  # エラー出力ファイルを閉じる

    # print(output_data2[1][4])
    # output_data3 = [0] * input_data_num
    # for i in range(input_data_num):
    #     for value in output_data2[i]:
    #         print(output_data2[i])
    #         for j in range(output_node_num):
    #             if value < threshold[0]:  # 閾値を使って出力データを変換
    #                 output_data3[i][j] = correct_value[0]  # 0に変換
    #             elif value < threshold[1]:
    #                 output_data3[i][j] = correct_value[1]
    #             elif value < threshold[2]:
    #                 output_data3[i][j] = correct_value[2]
    #             else:
    #                 output_data3[i][j] = correct_value[3]
    
    # with open(cir + 'model' + str(model_id) + '_extraoutput.txt', 'w') as f:
    #     f.write(str(output_data3) + '\n')
    #     f.write('\n')

    # モデルの評価結果
    print('正解数：' + str(correct_num))
    print('入力データ数：' + str(input_data_num))
    accuracy = correct_num / input_data_num
    # precision = precision_score(correct_data, convert_output_data, average='macro')
    # recall = recall_score(correct_data, convert_output_data, average='macro')
    # f1 = f1_score(correct_data, convert_output_data, average='macro')

    print('モデル' + str(model_id) + 'の評価結果')
    print('Accuracy: ' + str(accuracy))
    with open(learning_data_file + cir + 'model_accuracy.txt', 'a') as f:
        f.write('モデル' + str(model_id) + 'の正解率：' + str(accuracy) + '\n')
    
    with open(learning_data_file + cir + 'model' + str(model_id) + '_output_data.txt', 'w') as f:
        for i in range(input_data_num):
            f.write(', '.join(map(str, model_output_data[i])) + '\n')
    
    # print(model_output_data)


    # print('Precision: ' + precision)
    # print('Recall: ' + recall)
    # print('F1: ' + f1)

    # 混同行列
    # cm = confusion_matrix(correct_data, convert_output_data)
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

    with open(single_line_file, 'r') as f:
        single_flag = int(f.readline().replace("\n", ""))  # 統合されていない信号線があるかどうかを取得
    
    # 部分モデルの正解率記録
    import datetime as dt
    datetime = dt.datetime.now()
    import time
    start = time.time()
    with open('model_correct_rate.txt', 'a') as g:
        print("\n", file=g) 
        print(datetime, file=g)
        print("実行プログラム：" + os.path.basename(__file__), file=g)
        print("モデル分割数：" + str(num_models), file=g)
        print("対象回路：" + cir, file=g)
        print("中間層1ノード数：" + str(mid_node_num) + "   中間層2ノード数：" + str(mid_node_num2) + "   出力ノード数：" + str(output_node_num) + "   エポック数：" + str(epochs) + "   バッチサイズ：" + str(batch_size), file=g)

    result = [0 for _ in range(num_models)]  # 各モデルの正解率を格納するリスト
    for i in range(num_models):
        result[i] = worker(i)

    end = time.time() #プログラム終了時間

    print(f"学習時間：{end - start:.4f}秒")

    with open(cir + 'model_accuracy.txt', 'a') as f:
        f.write('\n')

    with open('model_correct_rate.txt', 'a') as g:
        print(f"学習時間：{end - start:.4f}秒", file=g)
        print(result, file=g) 
    print(result)

