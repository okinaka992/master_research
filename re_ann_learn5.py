# re_ann_learn3.pyを編集したプログラム。
# learn_data_delite5.ipynbで、作成した正解データを学習できる。
# 再学習時に，中間層ノード数を増やす場合と，エポック数を増やす場合どちらも自動的に対応できるようにした。
# 中間ノード数を増やさないまま，エポック数だけ任意の回数（interval_value）繰り返して増やして学習した後に，中間層ノード数を増やす，というように設定している
# 例えば，エポック数を3回繰り返して増やした後，中間層ノード数を増やす，というように設定できる。
# test_flag=Trueにすると，テストモードになり，エポック数が1になる。

# モデルごとにエポック数を設定したい場合に対応。
# re_ann_learn.pyでは，一度再学習したら終わりだが，このプログラムでは，正解率が100%になるまで再学習を繰り返すプログラム。
# ann_learn16.pyを編集したプログラム。
# ann_learn16.pyで学習したモデルのうち、正解率が100%でなかったモデルを再学習させるプログラム。
# learn_data_delite3.ipynbで、設定した正解データを学習できる。
# all_models_learnによって全てのモデルを学習させるか、一部のモデルを学習指せるかを指定。True: 全てのモデルを学習させる, False: 1回にprocess個のモデルだけ学習させる
# cd workspace/research2/experiment
#　実行コマンド：　python3 re_ann_learn2.py

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

import smtplib  # メール送信に必要なモジュール
from email.message import EmailMessage # メールの内容を作成するためのモジュール


#グローバル変数
cir = 's38584'  # 対象回路
test_flag = True  # テストモードかどうかのフラグ. True: テストモード, False: 通常モード ⇒ テストモードの場合は，エポックを1にする
protect_mode = True  # 保護モードかどうかのフラグ. True: 保護モード, False: 通常モード ⇒ 保護モードの場合は，tmpフォルダにデータを保存する

suplited_correct_data_folder = cir + '分割正解データ/'  # 削除後の正解データフォルダ
suplited_delited_correct_data_folder = cir + '分割正解データ削除後_test/'  # 削除後の正解データフォルダ
model_folder = cir + 'sepmodel2/'  # 全てのモデルを学習させる場合に、学習済みモデルを保存するフォルダ
model_test_folder = cir + 'sepmodel_test/'  # 一部のモデルを学習させる場合に、学習済みモデルを保存するフォルダ

input_data_file = cir + '分割入力データ' + '/' + cir + 'input'  # 入力データファイル
correct_data_file = suplited_delited_correct_data_folder + cir + 'integrated_output'  # 統合後の正解データファイル
middle_layer_node_num_file = suplited_delited_correct_data_folder + cir + 'middle_layer_node_num'  # 各正解データごとの中間層ノード数が保存されたファイル

epoch_num_file = suplited_delited_correct_data_folder + cir + 'epoch_num'  # 各分割正解データごとのエポック数が保存されたファイル
epoch_num_after_added_file = suplited_delited_correct_data_folder + cir + 'epoch_num_after_added'  # 学習回数を追加した後のエポック数を保存するファイル

delited_data_num_file = suplited_delited_correct_data_folder + cir + 'delited_data_num'  # 各分割正解データごとに削除された正解データ数が保存されたファイル
correct_data_value_file = suplited_correct_data_folder + cir + 'correct_value'  # 正解データの種類（値）を保存するファイル
threshold_file = suplited_correct_data_folder + cir + 'threshold'  # 閾値を保存するファイル
suplit_num_file = suplited_correct_data_folder + cir + 'suplit_num'  # 分割数が保存されたファイル
single_line_file = suplited_correct_data_folder + cir + 'single_line'  # 統合されていない信号線があるかが保存されたファイル
learning_data_file = 'learning_data/' + cir + '_learning_data/' # 学習データが保存されたファイル
error_output_file = 'error_output/' + cir + 'error_output/'  # エラー出力ファイル
data_correct_rate_all_file = cir + 'data_and_correct_rate_all' # ノード数，データ数，正解率 →これまでのすべての記録
data_correct_rate_file = cir + 'data_and_correct_rate' # ノード数，データ数，正解率　⇒　直近の記録
# re_data_correct_rate_file = cir + 're_data_and_correct_rate' # 再学習後のノード数，データ数，正解率 ⇒ data_correct_rate_fileと一緒だと万一学習が失敗したときに困るので、再学習後は別ファイルに保存
model_correct_rate_file = 'model_correct_rate.txt'  # 今までの学習記録を保存するファイル
result_file = cir + 'model_correct_rate'  # モデルの正解率を保存するファイル
re_model_correct_rate_file = 're_model_correct_rate.txt' # 再学習後のモデルの正解率記録ファイル
re_model_correct_rate_middle_node_file = cir + 're_model_correct_rate_middle_node.txt' # 再学習後のモデルの中間層ノード数記録ファイル
single_line_inf_file = suplited_delited_correct_data_folder + cir + 'single_line_inf'  # 統合されていない信号線の情報を保存するファイル
re_learning_status_file = 're_learning_status.txt'  # 再学習の進捗状況を保存するファイル
finished_model_file = 'finished_model.txt'   # 学習が終わったモデルのIDを保存するファイル

not_100_percent_models_file = model_folder + cir + 'not_100_percent_models'  # 正解率が100%でなかったモデルの記録ファイル

single_line_model = None  # 統合されていない信号線があるモデル番号を格納するリスト
single_line_suplit_idx = None  # 統合されていない信号線の値が、該当正解データ中のどのインデックス番号に対応しているかを格納する変数
input_node_num = None  #入力層におけるノード数 初期値は8　学習結果によって変更

mid_node_num = None  #生成器の中間層のノード数
mid_node_num2 = None  #生成器の中間層のノード数
add_mid_node_num = 0  # 中間層ノード1に追加するノードの数
add_mid_node_num2 = 0  # 中間層ノード2に追加するノードの数

how_many_adding_node = 25  # 追加する中間層ノード1数の基本値(増やす際には25個づつ増やす)
how_many_adding_node2 = 25  # 追加する中間層ノード2数の基本値(増やす際には25個づつ増やす)

the_number_of_adding_node = 4  # 中間層ノード数を増やす最大回数
interval_value = 3  # 中間層ノード数を増やすまでの再学習回数の間隔　＝　検証するエポック数の数（例：エポック数の探索範囲が[100, 200, 300]なら3=interval_value， [100, 200, 300, 400, 500, 600]なら6=interval_value）
max_re_learning_count =  the_number_of_adding_node * interval_value  # 最大再学習回数．max_re_learning_countをinterval_valueで必ず割れるようにするために， how_many_node_addを掛けている

add_epoch_num = 200  # 追加するエポック数

output_node_num = None #　出力層のノード数＝分割数による
output_node_num_record = None  # 出力層のノード数を記録する変数 = ファイルに記録する用
last_model_output_node_num_record = None  # 最後のモデルの出力層のノード数を記録する変数 = ファイルに記録する用　⇒ 最後のモデルは信号線の数やモデル分割数によって、出力層のノード数が他のモデルと異なることがあるため
num_models = None  # 学習させるモデルの数
single_flag = None # 統合されていない信号線があるかどうかを示す # 0:存在しない, 1:存在する
learning_rate = 0.0005
dropout_rate = 0
epochs = None
batch_size = 4
correct_value = None  # 正解データの値・種類
threshold = None  # ANNの出力値を変換するための閾値

not_100_percent_models = []  # 正解率が100%でなかったモデル番号のリスト

re_learning_count = 0  # 再学習回数カウント用変数

all_models_learn = True  # 全てのモデルを学習させるかどうかのフラグ. True: 全てのモデルを学習させる, False: 1回にprocess個のモデルだけ学習させる
re_learning_flag = True  # 再学習を行うかどうかのフラグ. True: 再学習を行う, False: 再学習を行わない
    
processes = 9  # 並列処理のプロセス数


def safe(path):
    if protect_mode:
        return 'tmp/' + os.path.basename(path)
    return path


def mk_input_data(input_file):
    # 入力データファイルを開いてデータを読み込む
    with open(input_file) as f:
        lines = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]

    # print(lines)
    
    input_data_num = int(len(lines))      #学習データ数を設定。学習データ数は入力データの行数
    global input_node_num               #グローバル変数を書き換え
    input_node_num = int(len(lines[0]))      #入力ノード数を設定。入力ノード数は入力データの各行の要素数

    int_lines = [list(map(int, _)) for _ in lines]  #list要素の型をint型に変換

    return np.array(int_lines), input_data_num


def mk_output_data(fname, model_id):
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

    if model_id == 0:  # 最初のモデルの場合
        with open(model_correct_rate_file, 'a') as g:
            print("最後のモデル以外の出力ノード数：" + str(output_node_num), file=g)
    elif model_id == (num_models - 1): # 最後のモデルの場合
        with open(model_correct_rate_file, 'a') as g:
            print("最後のモデルの出力ノード数：" + str(output_node_num), file=g)
    
    int_lines = [list(map(float, _)) for _ in lines]  #list要素の型をfloat型に変換

    return np.array(int_lines)

def set_middle_node_num(model_id):

    with open(middle_layer_node_num_file, 'r') as f:
        lines = f.readlines()[2:]  # 先頭2行は説明なのでスキップ
        middle_node = [[int(x) for x in line.strip().split(",")] for line in lines if line.strip()]

    global mid_node_num, mid_node_num2
    mid_node_num = middle_node[model_id][0] + add_mid_node_num
    mid_node_num2 = middle_node[model_id][1] + add_mid_node_num2

    print("モデル" + str(model_id) + "の中間層ノード数：" + str(mid_node_num) + ", " + str(mid_node_num2))

    return mid_node_num, mid_node_num2

def set_epochs(model_id):
    global epochs
    
    if re_learning_count == 0:  # 最初の学習の場合
        with open(epoch_num_file) as f:
            epoch_nums = [int(line.strip()) for line in f.readlines() if line.strip()]
        times_value = 0
    elif re_learning_count % interval_value == 0:  # 中間ノードを増やして最初の学習を行う場合（※エポック数も増やす）⇒初期のエポック数を参照したいため，「epoch_nums」を読み込む
        with open(epoch_num_file) as f:
            epoch_nums = [int(line.strip()) for line in f.readlines() if line.strip()]
            times_value = re_learning_count // interval_value  # 中間ノード数が増えるたびに初期値も増やす
    else:                                          # 中間ノードを据え置きでエポック数だけを増やして正解率の改善を図る場合 ⇒　学習回数を追加した後のエポック数を参照したいため，「epoch_num_after_added_file」を読み込む
        with open(epoch_num_after_added_file) as f:
            epoch_nums = [int(line.strip()) for line in f.readlines() if line.strip()]
        
        times_value = re_learning_count % interval_value  # 中間ノード数が増えるたびに初期値も増やす

    epochs = epoch_nums[model_id] + (add_epoch_num * times_value)

    if test_flag == True:
        epochs = 1  # テストモードの場合は，エポックを1にする

    print("モデル" + str(model_id) + "のエポック数：" + str(epochs))

    return epochs


# モデルの学習を行う関数
def build_and_train_model(input_data, correct_data, model_id):
    print('モデル' + str(model_id) + 'の学習を開始します')
    print(correct_data)

    with open("model_middle_node.txt", 'a') as f:
        f.write('モデル' + str(model_id) + 'の中間層ノード数：' + str(mid_node_num) + ', ' + str(mid_node_num2) + '\n')

    # モデルの構築
    model = Sequential()
    model.add(Dense(mid_node_num, input_dim=input_node_num, activation='tanh'))  # 入力層と中間層1
    model.add(Dense(mid_node_num2, activation='tanh'))  # 中間層2
    model.add(Dense(output_node_num, activation='sigmoid'))  # sigmoid or linear
    # model.add(Dense(mid_node_num, input_dim=input_node_num))  # 入力層と中間層1
    # model.add(LeakyReLU(alpha=0.01))  # LeakyReLU活性化関数を追加
    # model.add(Dense(mid_node_num2, activation='tanh'))  # 中間層2
    # model.add(LeakyReLU(alpha=0.01))  # LeakyReLU活性化関数を追加
    # model.add(Dense(output_node_num, activation='linear'))  # sigmoid or linear

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    # loss = 'binary_crossentropy'  # 二値分類の場合の損失関数
    # loss = 'mean_squared_error'  # 回帰問題の場合の損失関数
    
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
    with open(model_folder + cir + 'model_' + str(model_id) + '.tflite', 'wb') as f:  # モデルを保存するファイルを開く. wbはバイナリ書き込みモード
        f.write(tflite_model)


def ann_evaluation(input_data, correct_data, model_id, input_data_num):
    # モデルの読み込み
    with open(model_folder + cir + 'model_' + str(model_id) + '.tflite', 'rb') as f:  # モデルを読み込むファイルを開く
        tflite_model = f.read()

    # モデルの評価
    interpreter = tf.lite.Interpreter(model_content=tflite_model)  # TFLite形式のモデルを読み込む。保存されたTFLiteモデルをメモリに読み込み、推論を行う準備をするためのインタープリターを作成します。
    interpreter.allocate_tensors()  # #メモリを確保。モデルが使用するテンソル（データ構造）をメモリに割り当てます。TFLiteモデルをインタープリターにロードするだけでは、テンソルのメモリは割り当てられていません。この行を実行することで、モデルが推論に必要なメモリを確保します。

    input_details = interpreter.get_input_details()  # モデルの入力テンソルの詳細を取得。モデルの入力に関する詳細情報をリスト形式で返します。各要素は、入力テンソルの形状、データ型、名前などの情報を含む辞書です。
    output_details = interpreter.get_output_details()  # モデルの出力テンソルの詳細を取得。モデルの出力に関する詳細情報をリスト形式で返します。各要素は、出力テンソルの形状、データ型、名前などの情報を含む辞書です。
    
    correct_num = 0  # 正解数
    model_output_data = [0 for _ in range(input_data_num)]  # モデルの出力データを格納するリスト
    # output_data2 = []
    # f = open("s344_model_outputa.txt", 'w')  # 正解データファイルを開く
    # g = open(error_output_file + cir + 'model' + str(model_id) + '_error_output.txt', 'w')  # エラー出力ファイルを開く
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
        if (model_id == single_line_model) and single_flag == 1:  # 最後のモデルで、統合されていない信号線がある場合
            # print("single_line_suplit" + str(single_line_suplit_idx))
            for j in range(output_node_num):
                if j != single_line_suplit_idx:  # 統合されていない信号線の出力ノード以外の場合
                    if convert_output_data[0][j] < threshold[0]:  # 閾値を使って出力データを変換
                        convert_output_data[0][j] = correct_value[0]  # 0に変換
                    elif convert_output_data[0][j] < threshold[1]:
                        convert_output_data[0][j] = correct_value[1]
                    elif convert_output_data[0][j] < threshold[2]:
                        convert_output_data[0][j] = correct_value[2]
                    else:
                        convert_output_data[0][j] = correct_value[3]
                else:  # 統合されていない信号線がある場合、その信号線を学習した出力ノードは統合されていない信号線の値を持つため、0または1に変換する
                    if convert_output_data[0][j] <= threshold[1]:
                        convert_output_data[0][j] = correct_value[0]
                    else:
                        convert_output_data[0][j] = correct_value[3]
                # f.write(str(convert_output_data[0][j]) + " ")  # 出力データをファイルに書き込む
        else:  # 統合されていない信号線がない場合、または統合されていない信号線はあるが統合された信号線を学習したモデル以外の場合
            for j in range(output_node_num):
                if convert_output_data[0][j] < threshold[0]:  # 閾値を使って出力データを変換
                    convert_output_data[0][j] = correct_value[0]  # 0に変換
                elif convert_output_data[0][j] < threshold[1]:
                    convert_output_data[0][j] = correct_value[1]
                elif convert_output_data[0][j] < threshold[2]:
                    convert_output_data[0][j] = correct_value[2]
                else:
                    convert_output_data[0][j] = correct_value[3]
                # f.write(str(convert_output_data[0][j]) + " ")  # 出力データをファイルに書き込む
        
        # f.write("\n")  # 出力データをファイルに書き込む
            
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
        #     print("正解データと一致しました：", i, convert_output_data[0])
        # else:
        #     print("正解データと一致しませんでした：", i, convert_output_data[0])

        # else:
        #     if epochs > 500:
        #         g.write('正解データ：' + str(correct_data[i]) + '\n')
        #         g.write('変換後出力データ：' + str(convert_output_data[0]) + '\n')
        #         g.write('変換前出力データ：' + str(output_data[0]) + '\n')
        #         g.write('\n')


        # else:
        #     if model_id == 0:
        #         print("正解データと一致しませんでした：", i, convert_output_data[0])
        
        convert_output_data = np.array(convert_output_data)
    
    # f.close()  # 出力データファイルを閉じる
    # g.close()  # エラー出力ファイルを閉じる

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
    

def worker(id):

    model_id = not_100_percent_models[id]  # 学習させるモデル番号を取得

    input_file = input_data_file + str(model_id)  # 分割された入力データファイル名
    correct_file = correct_data_file + str(model_id)  # 分割された正解データファイル名に変更　＝　s344分割正解データ/s344integrated_output + 番号

    input_data, input_data_num = mk_input_data(input_file)
    correct_data = mk_output_data(correct_file, model_id)
    print(model_id, correct_data)

    mid_node_num, mid_node_num2 = set_middle_node_num(model_id)

    epochs = set_epochs(model_id)

    build_and_train_model(input_data, correct_data, model_id)

    accuracy = ann_evaluation(input_data, correct_data, model_id, input_data_num)

    with open(finished_model_file, 'a') as f:
        f.write("モデルID：" + str(model_id) + "   正解率：" + str(accuracy) + '\n')

    return accuracy, mid_node_num, mid_node_num2, epochs


if __name__ == '__main__':

    if protect_mode:
        
        

    if test_flag == True:
        model_folder = model_test_folder

    import datetime as dt
    import time

    datetime = dt.datetime.now()
    with open(re_learning_status_file, 'w') as f:
        f.write('\n')
        f.write(str(datetime) + '\n')

    while re_learning_flag == True and re_learning_count < max_re_learning_count:  # 再学習を繰り返す場合

        # ループごとにfinish_model_fileを初期化
        with open(finished_model_file, 'w') as f:
            f.write('\n')
    
        # エポックを3回繰り返して増やした後，中間層ノード数を増やす　⇒　エポックと中間ノードを交互に増やしていく
        if re_learning_count != 0 and re_learning_count % interval_value == 0:  # 1回目（re_learning_count=0）は中間層ノード数を増やさない，3回目（re_learning_count=3）以降は中間層ノード数を増やす
            add_mid_node_num = how_many_adding_node  # 中間層ノード数を増やす
            add_mid_node_num2 = how_many_adding_node2  # 中間層ノード数を増やす
        else:
            add_mid_node_num = 0  # 中間層ノード数を増やさない
            add_mid_node_num2 = 0  # 中間層ノード数を増やさない

        # 再学習が必要なモデル番号を読み込む
        with open(not_100_percent_models_file, 'r') as f:
            line = f.readline()
            lines = f.readlines()
            not_100_percent_models = [int(line.replace("\n", "")) for line in lines]

        num_models = int(len(not_100_percent_models))

        if num_models == 0:  # 再学習が必要なモデルがない場合、再学習を終了
            print("再学習が必要なモデルがありません。再学習を終了します。")
            break

        # 正解データの種類（値）を取得
        with open(correct_data_value_file, 'r') as f:
            correct_value = [float(value) for value in f.readline().split()]
        
        # 閾値を取得
        with open(threshold_file, 'r') as f:
            threshold = [float(value) for value in f.readline().split()]
        
        print("正解データの種類（値）：", correct_value)
        print("閾値：", threshold)

        with open(single_line_file, 'r') as f:
            single_flag = int(f.readline().replace("\n", ""))  # 統合されていない信号線があるかどうかを取得

        if single_flag == 1:  # 統合されていない信号線がある場合
            with open(single_line_inf_file, 'r') as f:
                line = f.readline().replace("\n", "")
                single_line_model = int(f.readline().replace("\n", ""))         # 統合されていない信号線が含まれるモデル番号
                single_line_suplit_idx = int(f.readline().replace("\n", ""))    # 統合されていない信号線が含まれる分割データのインデックス番号
            
        
        # 部分モデルの正解率記録
        datetime = dt.datetime.now()
        start = time.time()
        with open(re_model_correct_rate_file, 'a') as g:
            print("\n", file=g) 
            print(datetime, file=g)
            print("実行プログラム：" + os.path.basename(__file__), file=g)
            print("モデルの保存場所：" + model_folder, file=g)
            print("再学習対象モデル数：" + str(num_models), file=g)
            print("再学習対象モデル番号：" + str(not_100_percent_models), file=g)
            print("再学習回数（re_learning_count）：" + str(re_learning_count), file=g)
            print("対象回路：" + cir, file=g)
            print("正解データ；" + str(correct_value), file=g)
            print("閾値：" + str(threshold), file=g)
            print("エポック数：" + str(epochs) + "   バッチサイズ：" + str(batch_size), file=g)
        
        # 再学習後、再学習していないモデル（正解率100％のモデル）を含めたすべてのモデルの正解率を記録
        with open(model_correct_rate_file, 'a') as g:
            print("\n", file=g) 
            print(datetime, file=g)
            print("実行プログラム：" + os.path.basename(__file__), file=g)
            print("モデルの保存場所：" + model_folder, file=g)
            print("再学習対象モデル数：" + str(num_models), file=g)
            print("再学習対象モデル番号：" + str(not_100_percent_models), file=g)
            print("学習回数（re_learning_count）：" + str(re_learning_count), file=g)
            print("対象回路：" + cir, file=g)
            print("正解データ；" + str(correct_value), file=g)
            print("閾値：" + str(threshold), file=g)
            print("エポック数：" + str(epochs) + "   バッチサイズ：" + str(batch_size), file=g)
            print("中間層1ノード数：" + str(mid_node_num) + "   中間層2ノード数：" + str(mid_node_num2), file=g)

        p = multiprocessing.Pool(processes=processes)     # 5つのプロセスを使って並列処理を行うプールを作成。processesは、cpuのコア数
        worker_result = p.map(worker, range(0, num_models))         #worker関数を引数の範囲でマップし、それぞれの値に対して関数を実行します。
        p.close()   #処理を終了
        p.join()    #プロセスが終了するまで待つ
        end = time.time() #プログラム終了時間

        print(f"学習時間：{end - start:.4f}秒")

        # worker_resultから正解率と中間層ノード数を取得
        result = [0 for _ in range(num_models)]  # 正解率を格納するリスト
        mid_node_num_list = [0 for _ in range(num_models)]
        mid_node_num2_list = [0 for _ in range(num_models)]
        epochs_num_list = [0 for _ in range(num_models)]
        for i in range(num_models):
            result[i] = worker_result[i][0]  # 正解率
            mid_node_num_list[i] = worker_result[i][1]  # 中間層1ノード数
            mid_node_num2_list[i] = worker_result[i][2]  # 中間層2ノード数
            epochs_num_list[i] = worker_result[i][3]  # エポック数

        with open(cir + 'model_accuracy.txt', 'a') as f:
            f.write('\n')

        with open(re_model_correct_rate_file, 'a') as g:
            print("中間層1ノード数：" + str(mid_node_num), file=g)
            print("中間層2ノード数：" + str(mid_node_num2), file=g)
            print(f"学習時間：{end - start:.4f}秒", file=g)
            print(result, file=g) 
            print("平均正解率：", sum(result) / len(result), file=g)
            print("正解率100%のモデル数：", result.count(1.0), file=g)
        print(result)
        print("平均正解率：", sum(result) / len(result))
        print("正解率100%のモデル数：", result.count(1.0))
        print("モデル分割数：" + str(num_models))


        if re_learning_count != 0: # 2回目以降の再学習の場合　⇒　1回目は，再学習前のデータがない（再学習ではない）ので読み込まない
            # 再学習前の学習データを読み込む（※正解率100%のモデルも含む）
            with open(data_correct_rate_file, 'r') as f:
                line = f.readlines()  # 先頭2行を読み込む
                top_2_lines = line[:2]
                learning_data = line[2:]  # 3行目以降を読み込む

            learning_data_temp = [line.strip().replace("\n", "") for line in learning_data]
            all_correct_rate_data = [float(line.split(",")[-1]) for line in learning_data_temp if line.strip() != ""]  # 正解率のデータを抽出

            for i in range(len(not_100_percent_models)):
                all_correct_rate_data[not_100_percent_models[i]] = result[i]  # 再学習後の正解率で更新
        
            # 正解率をファイルに保存
            with open(result_file, 'w') as f:
                for model_id, accuracy in enumerate(all_correct_rate_data):
                    f.write(f"モデル{model_id}\n")
                    f.write(f"{accuracy}\n")
        else:  # 最初の再学習の場合
            # 正解率をファイルに保存
            with open(result_file, 'w') as f:
                for model_id, accuracy in enumerate(result):
                    f.write(f"モデル{model_id}\n")
                    f.write(f"{accuracy}\n")


        # 再学習後、再学習していないモデル（正解率100％のモデル）を含めたすべてのモデルの正解率を記録
        with open(model_correct_rate_file, 'a') as g:
            print(f"学習時間：{end - start:.4f}秒", file=g)
            print(result, file=g) 
            print("平均正解率：", sum(result) / len(result), file=g)
            print("正解率100%のモデル数：", result.count(1.0), file=g)


        with open(middle_layer_node_num_file) as f:
            lines = f.readlines()[2:]  # 先頭2行は説明なのでスキップ
            middle_node = [[int(x) for x in line.strip().split(",")] for line in lines if line.strip()]
        
        with open(delited_data_num_file) as f:
            lines = f.readlines()[4:]  # 先頭2行は説明なのでスキップ

        sep = ":" # 区切り文字
        delited_data_num = [[int(x) for x in (line.partition(sep)[2] if sep in line else line).strip().split(",") if x.strip() != ""] for line in lines if line.strip()]

        for i in range(len(delited_data_num)):
            delited_data_num[i] = delited_data_num[i][0]  # 二重リストを一次元リストに変換

        with open(data_correct_rate_all_file, 'a') as f:
            f.write("\n\n")
            f.write(datetime.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("実行プログラム：" + os.path.basename(__file__) + "\n")
            f.write("ノード数,  削除後正解データ数,  正解率\n")
            for i in range(len(result)):
                f.write(str(middle_node[i][0]) + "-" + str(middle_node[i][1]) + ",  " + str(delited_data_num[i]) + ",  " + str(result[i]) + "\n")


        if re_learning_count != 0:  # 2回目以降の再学習の場合
            # 再学習後のモデル情報をlearning_dataの該当箇所に上書き
            for idx, i in enumerate(not_100_percent_models):
                learning_data[i] = str(i) + ",  " + str(middle_node[i][0]) + "-" + str(middle_node[i][1]) + ",  " + str(delited_data_num[i]) + ",  " + str(result[idx]) + "\n"

            # 上記で更新した情報と再学習していないモデル（正解率100％のモデル）を含めたすべてのモデルのモデル情報を更新
            with open(data_correct_rate_file, 'w') as f:
                f.write(datetime.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("モデル番号,  ノード数,  削除後正解データ数,  正解率\n")
                for i in range(len(learning_data)):
                    f.write(learning_data[i])

        else:  # 最初の再学習の場合
            with open(data_correct_rate_file, 'w') as f:
                f.write(datetime.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("モデル番号,  ノード数,  削除後正解データ数,  正解率\n")
                for i in range(len(result)):
                    f.write(str(i) + ",  " + str(middle_node[i][0]) + "-" + str(middle_node[i][1]) + ",  " + str(delited_data_num[i]) + ",  " + str(result[i]) + "\n")
        
        # 正解率100%でなかったモデル番号を記録
        with open(not_100_percent_models_file, 'w') as f:
                f.write("モデル番号,  ノード数,  削除後正解データ数,  正解率\n")
                for i in range(len(result)):
                    if result[i] < 1.0:
                        f.write(str(not_100_percent_models[i]) + "\n")

        
      # 中間層ノード数の更新
        with open(middle_layer_node_num_file, 'r') as f:
            lines = f.readlines()[2:]  # 先頭2行は説明なのでスキップ
            middle_node = [[int(x) for x in line.strip().split(",")] for line in lines if line.strip()]


        for i in range(num_models):
            middle_node[not_100_percent_models[i]][0] = mid_node_num_list[i]
            middle_node[not_100_percent_models[i]][1] = mid_node_num2_list[i]
        
        with open(middle_layer_node_num_file, 'w') as f:
            f.write(lines[0])  # 先頭2行の説明を書き込む
            f.write(lines[1])
            for i in range(len(middle_node)):
                f.write(str(middle_node[i][0]) + "," + str(middle_node[i][1]) + "\n")


      # エポック数の更新
        # 更新前のエポック数を読み込む
        with open(epoch_num_after_added_file, 'r') as f:
            epoch_num = [int(line.strip()) for line in f.readlines() if line.strip()]
        
        # エポック数を追加したモデルのエポック数を更新
        for i in range(num_models):
            epoch_num[not_100_percent_models[i]] = epochs_num_list[i]
        
        # 更新後のエポック数を保存
        with open(epoch_num_after_added_file, 'w') as f:
            for i in range(len(epoch_num)):
                f.write(str(epoch_num[i]) + "\n")

        type(result)


        if len(not_100_percent_models) == 0:  # 正解率100%のモデルがすべてになった場合、再学習を終了
            print("すべてのモデルの正解率が100%になりました。再学習を終了します。")
            break
        elif re_learning_count == max_re_learning_count - 1:  # 最大再学習回数に達した場合、再学習を終了
            print("最大再学習回数に達しました。再学習を終了します。")
            break
        elif len(not_100_percent_models) > 0:  # 正解率100%のモデルがまだある場合、再学習を続行
            with open(re_learning_status_file, 'a') as f:
                f.write("再学習回数：" + str(re_learning_count + 1) + "\n")
                f.write("正解率100%でなかったモデル番号：" + str(not_100_percent_models) + "\n")

        re_learning_count += 1  # 再学習回数をカウント
    


