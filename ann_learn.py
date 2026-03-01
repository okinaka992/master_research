import multiprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



#グローバル変数
cir = 's344'  # 対象回路
input_data_file = cir + 'input'  # 入力データファイル
correct_data_file = cir + '分割正解データ' + 'integrated_output'  # 統合後の正解データファイル
input_num = None  #入力層におけるノード数 初期値は8　学習結果によって変更
mid_nm = 64  #生成器の中間層のノード数　初期値は128　学習結果によって変更
output_num = None #　出力層のノード数＝分割数による
test_ptn_file = "gen_create.vec" #GANが生成したテストパターンを保存するファイル

def load_data(input_file, correct_file):
    # データの読み込み
    input_data = tf.data.experimental.load(input_file)
    correct_data = tf.data.experimental.load(correct_file)
    return input_data, correct_data

# モデルの構築を行う関数
def build_and_train_model(input_data, correct_data, model_id):
    # モデルの構築
    model = Sequential()
    model.add(Dense(mid_nm, input_dim=input_num, activation='relu'))
    model.add(Dense(output_num, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    # モデルの学習
    model.fit(input_data, correct_data, epochs=10, batch_size=4)
    
    # モデルの保存
    model.save(f'model_{model_id}.h5')  # モデルを保存 model_idは引数で受け取る

def worker(model_id):
    correct_file = f"{correct_data_file}{model_id}"
    input_data, correct_data = load_data(input_data_file, correct_file)
    build_and_train_model(input_data, correct_data, model_id)

if __name__ == '__main__':
    num_models = 4  # 並列に実行するモデルの数
    input_num = 8  # 入力層のノード数
    output_num = 10  # 出力層のノード数

    processes = []
    for i in range(num_models):
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()