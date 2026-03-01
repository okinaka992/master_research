# model_size_cal_test.ipynbを編集したもの
# model_size_cal_test.ipynbでは再学習後のメモリ量を計算するが、このプログラムはモデルを学習しなくてもlenar_data_suplit_test.ipynb,learn_data_delite_test.ipynbで競ってた中間ノード数からメモリ量を計算する
# データを何個ずつ分割するかを決めるときにこれで計算して一番小さい分割数を見つける

# モデルの総メモリ量を計算するプログラム


def main(cir):
    #変数
    input_data_file = cir + '分割入力データ/' + cir + 'input'  # 入力データフォルダ
    suplited_correct_data_folder = cir + '分割正解データ_test/'  # 削除後の正解データフォルダ
    suplit_num_file = suplited_correct_data_folder + cir + 'suplit_num'  # 分割数が保存されたファイル
    output_node_num_file = suplited_correct_data_folder + '/' + cir + 'suplit_data_num'  # 各モデルの出力ノード数が保存されたファイル
    middle_layer_node_num_file = cir + "分割正解データ削除後_test/" + cir + 'middle_layer_node_num'  # 中間層のノード数を保存するファイル
    model_size_file = cir + 'モデルメモリ量/' + cir + '_model_memory_size'
    model_size_test_file = cir + 'モデルメモリ量/' + cir + '_model_memory_size_test'
        
    # モデルの分割数を取得
    with open(suplit_num_file, 'r') as f:
        num_models = int(f.readline())
    
    print("モデル数:", num_models)
    
    # 入力ノード数を取得　⇒　全モデルで同じ
    with open(input_data_file + '0', 'r') as f:
        line = [f.readline().strip().replace("\n", "")]  # 先頭行を読み込む
        input_node_num = len(line[0].strip())
    
    print("入力ノード数:", input_node_num)
    
    # 出力ノード数を取得　⇒　モデルによって異なる可能性がある
    with open(output_node_num_file, 'r') as f:
        output_node_num = [int(line.strip().replace("\n", "")) for line in f.readlines()]
    
    print("各モデルの出力ノード数:", output_node_num)
    print("len(output_node_num)：", len(output_node_num))
    
    
    # 全モデルの中間ノード数を取得
    with open(middle_layer_node_num_file, 'r') as f:
        line = f.readlines()  # 先頭2行を読み込む
        top_2_lines = line[:2]
        learning_data = line[2:]  # 3行目以降を読み込む
    
    # learning_data_temp = [line.strip().replace("\n", "") for line in learning_data]
    middle_node_data_temp = [line.strip().replace("\n", "") for line in learning_data]
    # [line.split(",")[1].strip() for line in learning_data_temp if line.strip() != ""]  # 正解率のデータを抽出
    middle_node_data = [[0 for _ in range(2)] for _ in range(len(middle_node_data_temp))]  # 初期化
    for i in range(len(middle_node_data_temp)):
        middle_node_data[i][0] = int(middle_node_data_temp[i].split(",")[0])
        middle_node_data[i][1] = int(middle_node_data_temp[i].split(",")[1])
    
    print("len(middle_node_data):", len(middle_node_data))
    
    print("num_models:", num_models)
    print(len(middle_node_data), "個のモデルの中間ノード数データを取得しました。")
    print("len(middle_node_data)：", len(middle_node_data))
    print("len(output_node_num)：", len(output_node_num))
    
    # 各モデルの中間ノード数を計算
    model_size = 0
    for i in range(num_models):
        model_size += 4*((input_node_num+1)*middle_node_data[i][0] + (middle_node_data[i][0]+1)*middle_node_data[i][1] + (middle_node_data[i][1]+1)*output_node_num[i])
    
    print("モデルの総メモリ量(バイト):", model_size/2, "バイト")
    print("モデルの総メモリ量(メガバイト):", model_size/2/1024/1024, "MB")  # 1MB = 1024*1024バイト 2で割っているのはTFlite形式でモデルを保存する際に32ビットから16ビットに削減されメモリ量が半分になるため
    
    import os
    import datetime as dt
    import time
    
    datetime = dt.datetime.now()
    os.makedirs(cir + 'モデルメモリ量', exist_ok=True)
    with open(model_size_test_file, 'a') as f:
        f.write('\n')
        f.write(str(datetime) + '\n')
        f.write("実行プログラム：model_size_cal_test.ipynb\n")
        f.write("モデル分割数：" + str(num_models) + "\n")
        f.write("各モデルの出力ノード数:" + str(output_node_num) + "\n")
        f.write("モデルの総メモリ量(バイト):" + str(model_size/2) + "バイト\n")
        f.write("モデルの総メモリ量(メガバイト):" + str(model_size/2/1024/1024) + "MB\n")