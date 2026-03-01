import numpy as np

cir = 's344'  # 対象回路
simular_output = cir + 'simular_output'  # 最適なペア順に並べ替えられた正解データファイル
net_list = 'c' + cir   # ネットリストファイル
integrated_output = cir + 'integrated_output'  # 統合後の正解データファイル

integration = 2 # 故障候補値の統合数

# 正解データファイルを開いてデータを読み込む
with open(simular_output) as f:
  # 1行目（最適なペア信号線）を読み込む。カンマ区切りで各信号線をリストに格納
  best_pair = f.readline().split(',')

  #2行目意向を読み込む。_にいれた各文字列の空白文字を削除
  correct_data = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]  # 各行の空白文字と改行文字を削除


# 信号線数を取得
single_flag = 0  # 最適なペアが存在しない信号線があるかどうかを示すフラグ, 0:存在しない, 1:存在する
if len(correct_data) % integration == 0:  # 故障候補値の数がintegrationで割り切れるとき=全ての信号線に最適なペアが存在するとき
  signal_sum = len(correct_data)
else:                                     # 故障候補値の数がintegrationで割り切れないとき=信号線に最適なペアが存在しないとき
  signal_sum = len(correct_data)  - 1
  single_flag = 1

# 信号線統合後の正解データを格納するリスト
integrated_data = [["0" for _ in range(len(correct_data[0]))] for _ in range(signal_sum // integration + single_flag)] # 2次元配列を0で初期化。列数は故障候補値の数（correct_dataの列数）、行数は信号線数をintegrationで割った商にflagを足したもの＝統合後の信号線数。flagは、最適なペアが存在しない信号線があるとき1、ないとき0
# print(len(integrated_data))
# print(len(integrated_data[0]))

# 信号線を統合したときの正解データを作成
integrated_index = 0  # 統合した信号線のインデックス
for i in range(0, signal_sum, integration):  # 故障候補値をintegration個ずつ統合
  for j in range(len(correct_data[i])):
    if correct_data[i][j] == '0' and correct_data[i+1][j] == '0':  # どちらの信号線も0のとき
      integrated_data[integrated_index][j] = '0'
    elif correct_data[i][j] == '1' and correct_data[i+1][j] == '0':  # 信号線番号が小さい方の信号線が1のとき
      integrated_data[integrated_index][j] = '1'
    elif correct_data[i][j] == '0' and correct_data[i+1][j] == '1':  # 信号線番号が大きい方の信号線が1のとき
      integrated_data[integrated_index][j] = '2'
    elif correct_data[i][j] == '1' and correct_data[i+1][j] == '1':  # どちらの信号線も1のとき
      integrated_data[integrated_index][j] = '3'
  
  integrated_index += 1

# 最適なペアが存在しない信号線があるとき
if single_flag == 1:
  integrated_data[integrated_index] = correct_data[integrated_index]

# 統合した正解データを転置　⇒　intewgrated_dataの行と列を入れ替える ⇒　各信号線の故障候補値を列ごとに書き込むため
integrated_data2 = [list(col) for col in zip(*integrated_data)]

# 統合した正解データをファイルに書き込む
with open(integrated_output, 'w') as f:
  # 統合後の信号線数を書き込む
  f.write(str(len(integrated_data2[0])) + '\n')

  # 最適なペア信号線を書き込む
  for i in range(len(best_pair) - 1):
    f.write(best_pair[i] + ',')  # 2行目に最適なペア信号線を書き込む
  f.write(best_pair[i+1])

  # 3行目以降に統合した正解データを書き込む
  for data in integrated_data2:
    f.write(','.join(data) + '\n')

