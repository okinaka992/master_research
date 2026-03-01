# 正解データを分割するプログラム
# learn_data_suplit.pyを改良したもの
# 特定のモデルの正解データをさらに分割するプログラム
# 正解データを0~1に正規化. 0~3の値を0~1に変換。0->0, 1->0.375, 2->0.625, 3->1 (※値がなるべく閾値から離れるように0は0のまま、3は1とした)
# cd workspace/research2/experiment
#　実行コマンド：　python3 learn_data_suplit.py

#グローバル変数
cir = 's344'  # 対象回路
correct_data_file = cir + 'integrated_output'  # 正解データファイル
suplit_num = 21 # 何個ずつ分割するか
correct_data_folder = cir + '分割正解データ' # 正解データを分割して保存するフォルダ
suplit_correct_data_file = cir + 'integrated_output'  # 分割された正解データファイル
suplit_num_file = cir + 'suplit_num'  # 分割数を保存するファイル
more_suplit_num = [] # さらに分割するモデル番号を格納するリスト

# 正解データファイルを開いてデータを読み込む
with open(correct_data_file) as f:
  # 1行目 統合後の信号線の数を読み込む
  signal_sum = int(f.readline())

  # 2行目（最適なペア信号線）を読み込む。カンマ区切りで各信号線をリストに格納
  best_pair = f.readline().split(',')

  # 3行目以降を読み込む。_にいれた各文字列の空白文字を削除
  correct_data = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]  # 各行の空白文字と改行文字を削除

correct_data = [list(_) for _ in correct_data]  # 1次元リストを2次元リストに変換

print(len(correct_data))
# print(correct_data[0])
# print(correct_data[0][0])
# print(type(correct_data[0]))

# 分割された正解データを格納するリスト
suplit_correct_data = [["0" for _ in range(len(correct_data[0]))] for _ in range(suplit_num)] # 2次元配列を0で初期化。列数は故障候補値の数（correct_dataの列数）、行数はsuplit_num=分割数

for i in range(signal_sum//suplit_num):  # 統合後の信号線数をsuplit_numで割る=分割数
    with open(correct_data_folder + '/' + suplit_correct_data_file + str(i), 'w') as f:  # 分割された正解データを書き込むファイルを開く
        for j in range(len(correct_data)):
            for k in range(i*suplit_num, i*suplit_num + suplit_num):    # 分割数ごとに左から右にsuplit_num個ずつ正解データを書き込む.正解データファイルごとに、分割される数（suplit_num）だけ、左にずれるので、kの範囲はi*suplit_numからi*suplit_num + suplit_num となる。iは0から始まるので、単純に、i*suplit_numではなく、i*suplit_num + suplit_numにしなければならない。
                # 正解データを0~1に正規化. 0~3の値を0~1に変換。0->0, 1->0.375, 2->0.625, 3->1
                if correct_data[j][k] == '0':
                    correct_data[j][k] = '0.000'
                elif correct_data[j][k] == '1':
                    correct_data[j][k] = '0.375'
                elif correct_data[j][k] == '2':
                    correct_data[j][k] = '0.625'
                elif correct_data[j][k] == '3':
                    correct_data[j][k] = '1.000'

                if k < i*suplit_num + suplit_num - 1:  # 最後の要素以外はカンマを付ける
                    f.write(correct_data[j][k] + ',')
                else:                                  # 最後の要素はカンマを付けない
                    f.write(correct_data[j][k] + '\n')
    
suplit_model_num = str(int(i) + 1)  # 分割モデル数をsuplit_model_numに格納
# print(suplit_model_num)

with open(correct_data_folder + '/' + suplit_num_file, 'w') as f:
    f.write(suplit_model_num + '\n')
        
  
