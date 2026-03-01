#各信号線の故障候補値を比較、最も値が似通っている信号線ペア（ハミング距離最小のペア）を見つけるプログラム
#※完成版

import numpy as np
from scipy.spatial import distance  # ハミング距離を求めるためのモジュール

cir = 's38584'  # 対象回路
correct_data = cir + 'output'  # 正解データファイル
net_list = 'c' + cir   # ネットリストファイル
simular_output = cir + 'simular_output'  # 並べ替え後の正解データを記述するファイル

integration = 2 # 故障候補値の統合数

# 正解データファイルを開いてデータを読み込む
with open(correct_data) as f:
  #_にいれた各文字列の空白文字を削除
  lines = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]  # 各行の空白文字と改行文字を削除

# ネットリストを開いて、入力信号線数、出力信号線数、その他の信号線数を取得
with open('c' + cir, 'r') as f:
  line_inf = f.readline()  # 1行目はネットリストの情報
  output_line_num = int(line_inf.split()[1])
  input_line_num = int(line_inf.split()[2])

print(output_line_num)
print(input_line_num)  # 入力信号線の数を出力

print(lines)  # 各行のデータを出力
# 各行の縦の列をリストに格納
signal_val = ['' for _ in range(len(lines[0]))]   # 信号線の数分のリストを作成
for i in range(len(lines[0])):
  for j in range(len(lines)):
    signal_val[i] = signal_val[i] + lines[j][i]
# print(signal_val[0])
# print(signal_val[1])
# print(signal_val[2])
# print(signal_val[3])
# print(signal_val[334])
signal_num = len(signal_val)  # 信号線の総数を取得

# 探索する信号線の信号線番号のリストを作成 = 0から信号線の数までのリスト = [0, 1, 2, ..., signal_num-1] ⇒　最適なペアを探すときに、同じ信号線を比較することがないようにするのに使う
line_num_list = [0 for _ in range(signal_num)]  # 信号線の数分のリストを作成
# 診断対象となる信号線は、出力信号線以外の信号線であり、信号線番号は入力信号線、出力信号線、その他の信号線の順に1, 2, ..., となっている。そのため、入力信号線番号の後は、出力信号線の数を足している
for i in range(signal_num):
  if i < input_line_num:
    line_num_list[i] = i + 1  # 入力信号線のインデックスをそのまま格納
  else:
    line_num_list[i] = i + 1 + output_line_num
# 探索する信号線のインデックスのリスト
print(line_num_list)
print(len(line_num_list))  # 信号線の数を出力

signal_val_index_list = [i for i in range(signal_num)]  # 信号線の数分のリストを作成
print(signal_val_index_list)  # 信号線のインデックスのリストを出力
print(len(signal_val_index_list))  # 信号線の数を出力

# ハミング距離を求め,ハミング距離が最小の信号線ペアのインデックスをファイルに書き込む
f = open('hamming_distance', 'w')
f.write(f"インデックス➀ インデックス➁ ハミング距離\n")

#探索時間を計測
import time
start = time.time()

# ハミング距離を求め,ハミング距離が最小の信号線ペアを見つける
best_pair = []  # 最適なペアを格納するリスト
best_pair_index = []  # 最適なペアのインデックスを格納するリスト
single_line = -1  # 最適なペアがない(=信号線数が奇数)信号線のインデックスを格納する変数
while len(signal_val_index_list) > 0:  # wile文にすることで、インデックスを削除しても問題なく処理できる
    idx = signal_val_index_list[0] # 探索リストの先頭の信号線のインデックス
    min_hamming = -1 # ハミング距離の最小値を格納する変数
    min_hamming_index = 0 # ハミング距離が最小の信号線のインデックスを格納する変数

    # 探索リストに残っている信号線が1本の場合 = 信号線数が奇数本の場合
    if len(signal_val_index_list) == 1:
        print(signal_val_index_list)
        single_line_index = signal_val_index_list[0]  # 探索リストに残っている信号線のインデックス
        single_line = line_num_list[idx]  # 探索リストに残っている信号線のインデックス
        print(f"信号線{single_line}には最適なペアがありません。")
        f.write(f"{single_line}\n")
        break

    for j in signal_val_index_list[1:]:  # 探索リストの先頭の信号線と他の信号線を比較
        #　文字列をリストに変換
        signal_val[idx] = list(signal_val[idx])
        signal_val[j] = list(signal_val[j])
        hamming = distance.hamming(signal_val[idx], signal_val[j]) * len(signal_val[idx])  # ハミング距離を求める
        if min_hamming == -1 or hamming < min_hamming:  # ハミング距離が最小のペアを更新
            min_hamming = hamming
            # print(j)
            min_hamming_index = j
    # ハミング距離が最小の信号線ペアを出力
    # print(f"インデックス{line_num_list[idx]}とインデックス{min_hamming_index}が最適なペアです。")
    # print(f"ハミング距離は{min_hamming}です。")
    # ハミング距離が最小のペアとハミング距離をファイルに書き込む
    f.write(f"{line_num_list[idx]} {line_num_list[min_hamming_index]} {min_hamming}\n")
    best_pair.append([line_num_list[idx], line_num_list[min_hamming_index]])  # 最適なペアをリストに追加
    best_pair_index.append([idx, min_hamming_index])  # 最適なペアのインデックスをリストに追加
    signal_val_index_list.remove(idx)  # 既に最適なペアになった信号線は探索リストから削除
    # if idx == 1:
    #    print(f"インデックス{line_num_list[idx]}とインデックス{min_hamming_index}が最適なペアです。")
    signal_val_index_list.remove(min_hamming_index)  # 既に最適なペアになった信号線は探索リストから削除

f.close()

# 探索リストに残っている信号線が1本の場合 = 信号線数が奇数本の場合
if len(signal_val_index_list) == 1:
    single_line = line_num_list[idx]  # 探索リストに残っている信号線のインデックス
    print(f"信号線{single_line}には最適なペアがありません。")

print(len(best_pair))
print("最適なペアのリスト：" + str(best_pair))  # 最適なペアを出力

# ハミング距離が最小の信号線ペアをファイルに書き込む。故障診断（diagnosis.ipynb)で使用する
with open(cir + 'pair_list', 'w') as f:
  f.write(f"統合数 {integration}\n")
  f.write(f"信号線数 {signal_num}\n")
  for i in range(len(best_pair)):  # 最適なペアの数だけループ
    f.write(str(best_pair[i][0]) + " " + str(best_pair[i][1]) + "\n")
  if len(signal_val_index_list) == 1:  # 信号線数が奇数本の場合.インデックスリストに残っている信号線のインデックスを出力
    f.write(str(single_line) + "\n")
  
print(len(best_pair))  # 最適なペアの数を出力
 
# ハミング距離が最小の信号線ペア順に故障候補値をファイルに書き込む
# 1行目は、並べ替え後の信号線番号を列挙。2行目以降は、各信号線の故障候補値を列挙
# 並べ替え前の正解データファイル（s~output）では、各行がテストパターン、各列が信号線を表していたが、並べ替え後のファイル（s~simular_output）では、各行が信号線、各列がテストパターンを表すようにする
integration_list = list(range(0, integration))  # for文の要素として使うため、リストに変換
with open(simular_output, "w") as f:
  # 最適なペアが存在しない信号線があるかどうかを示す, 0:存在しない, 1:存在する
  if len(signal_val_index_list) == 0: # ペアが存在しない信号線がない場合＝統合に余りがない場合
    f.write(f"0 #ペアが存在しない信号線はありません(最後の行の値も統合されています）\n")  # 信号線数を出力
  else:  # ペアが存在しない信号線があります＝統合に余りがある場合
    f.write(f"1 #ペアが存在しない信号線はあります（最後の行の値は統合されていません）\n") 

  # 2行目に並べ替え後の信号線番号を列挙
  for i in range(len(best_pair)):  # 最適なペアの数だけループ
    for j in integration_list:
      if (i != len(best_pair) - 1) or (j != integration - 1):
        f.write(f"{best_pair[i][j]},")
  if len(signal_val_index_list) == 0:  # ペアが存在しない信号線がない場合
    f.write(f"{best_pair[i][j]}\n")    # 最後の信号線の番号の後に改行を入れる
  else:
    f.write(f"{best_pair[i][j]},")
    f.write(f"{single_line}\n")

  # 3行目以降に各信号線の故障候補値を列挙
  for i in range(len(best_pair)):  
    for j in integration_list:
      for k in range(len(signal_val[j]) - 1):
        f.write(f"{signal_val[best_pair_index[i][j]][k]},")
      f.write(f"{signal_val[best_pair_index[i][j]][k + 1]}\n")  # 最後の信号線の値の後に改行を入れる
  
  # 信号線数が奇数本の場合、最後の行にペアがいない信号線の故障候補値を列挙
  if len(signal_val_index_list) == 1:
    for i in range(len(signal_val[single_line_index]) - 1):
        f.write(f"{signal_val[single_line_index][i]},")
    f.write(f"{signal_val[single_line_index][i]}\n")
    print("信号線の本数は奇数でした。")

end = time.time() #プログラム終了時間
print(f"学習時間：{end - start:.4f}秒")