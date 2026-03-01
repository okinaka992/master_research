# 正解データを分割するプログラム
# cd workspace/research2/experiment
#　実行コマンド：　python3 learn_data_suplit.py

#グローバル変数
cir = 's382'  # 対象回路
correct_data_file = cir + 'integrated_output'  # 正解データファイル
suplit_num = 21 # 何個ずつ分割するか
correct_data_folder = cir + '分割正解データ' # 正解データを分割して保存するフォルダ
suplit_correct_data_file = cir + 'integrated_output'  # 分割された正解データファイル
suplit_num_file = cir + 'suplit_num'  # 分割数を保存するファイル


# 正解データファイルを開いてデータを読み込む
with open(correct_data_file) as f:
  # 1行目 統合後の信号線の数を読み込む
  signal_sum = int(f.readline())

  # 2行目（最適なペア信号線）を読み込む。カンマ区切りで各信号線をリストに格納
  best_pair = f.readline().split(',')

  # 3行目以降を読み込む。_にいれた各文字列の空白文字を削除
  correct_data = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]  # 各行の空白文字と改行文字を削除

print(len(correct_data))

# 分割された正解データを格納するリスト
suplit_correct_data = [["0" for _ in range(len(correct_data[0]))] for _ in range(suplit_num)] # 2次元配列を0で初期化。列数は故障候補値の数（correct_dataの列数）、行数はsuplit_num=分割数

for i in range(signal_sum//suplit_num):  # 統合後の信号線数をsuplit_numで割る=分割数
    with open(correct_data_folder + '/' + suplit_correct_data_file + str(i), 'w') as f:  # 分割された正解データを書き込むファイルを開く
        for j in range(len(correct_data)):
            for k in range(i*suplit_num, i*suplit_num + suplit_num - 1):    # 分割数ごとに左から右にsuplit_num個ずつ正解データを書き込む
                f.write(correct_data[j][k] + ',')
            f.write(correct_data[j][k+1] + '\n')
    
suplit_num = str(int(i) + 1)  # 分割数をsuplit_numに格納
# print(suplit_num)

with open(correct_data_folder + '/' + suplit_num_file, 'w') as f:
    f.write(suplit_num + '\n')
        
  
