#各信号線の故障候補値を比較、最も値が似通っている信号線ペア（ハミング距離最小のペア）を見つけるプログラム

import numpy as np
from scipy.spatial import distance
import time

cir = 's38584'
correct_data = cir + 'output'
net_list = 'c' + cir
simular_output = cir + 'simular_output'
integration = 2

# 正解データを読み込む（カンマと改行を削除）
with open(correct_data) as f:
    lines = [line.replace(",", "").replace("\n", "") for line in f.readlines()]

# c ファイルから出力線数・入力線数を取得
with open('c' + cir, 'r') as f:
    line_inf = f.readline()
    output_line_num = int(line_inf.split()[1])
    input_line_num = int(line_inf.split()[2])  # ★修正：[2] を使う

print(f"出力線数: {output_line_num}, 入力線数: {input_line_num}")
print(f"総行数: {len(lines)}")

# 各信号線のデータを列方向で抽出（NumPy 配列で高速化）
signal_val = np.array([[lines[j][i] for j in range(len(lines))] for i in range(len(lines[0]))], dtype='U1')
signal_num = signal_val.shape[0]

print(f"信号線総数: {signal_num}")

# line_num_list の構築（元と同じロジック）
line_num_list = [0] * signal_num
for i in range(signal_num):
    if i < input_line_num:
        line_num_list[i] = i + 1
    else:
        line_num_list[i] = i + 1 + output_line_num

print(line_num_list[:10])

signal_val_index_list = list(range(signal_num))

# ハミング距離ファイルを開く
f = open('hamming_distance', 'w')
f.write(f"インデックス➀ インデックス➁ ハミング距離\n")

start = time.time()

# greedy pairing（元と同じ方式）
best_pair = []
best_pair_index = []
single_line = -1
single_line_index = -1

while len(signal_val_index_list) > 0:
    idx = signal_val_index_list[0]
    min_hamming = -1
    min_hamming_index = 0

    # 信号線が1本だけ残った場合
    if len(signal_val_index_list) == 1:
        single_line_index = signal_val_index_list[0]
        single_line = line_num_list[idx]
        print(f"信号線{single_line}には最適なペアがありません。")
        f.write(f"{single_line}\n")
        break

    # 他のすべての信号線とハミング距離を計算
    for j in signal_val_index_list[1:]:
        hamming = distance.hamming(signal_val[idx], signal_val[j]) * len(signal_val[idx])
        
        if min_hamming == -1 or hamming < min_hamming:
            min_hamming = hamming
            min_hamming_index = j

    # 最適なペアを記録
    f.write(f"{line_num_list[idx]} {line_num_list[min_hamming_index]} {int(min_hamming)}\n")
    best_pair.append([line_num_list[idx], line_num_list[min_hamming_index]])
    best_pair_index.append([idx, min_hamming_index])
    
    # 処理済みの信号線を削除
    signal_val_index_list.remove(idx)
    signal_val_index_list.remove(min_hamming_index)

f.close()

print(len(best_pair))
print("最適なペアのリスト：" + str(best_pair[:5]))  # 最初の5ペアを表示

# pair_list ファイルに出力
with open(cir + 'pair_list', 'w') as f:
    f.write(f"統合数 {integration}\n")
    f.write(f"信号線数 {signal_num}\n")
    for i in range(len(best_pair)):
        f.write(str(best_pair[i][0]) + " " + str(best_pair[i][1]) + "\n")
    if len(signal_val_index_list) == 1:
        f.write(str(single_line) + "\n")

# simular_output ファイルに出力
integration_list = list(range(0, integration))
with open(simular_output, "w") as f:
    if len(signal_val_index_list) == 0:
        f.write(f"0 #ペアが存在しない信号線はありません(最後の行の値も統合されています）\n")
    else:
        f.write(f"1 #ペアが存在しない信号線はあります（最後の行の値は統合されていません）\n")

    # 2行目に並べ替え後の信号線番号を列挙
    for i in range(len(best_pair)):
        for j in integration_list:
            if (i != len(best_pair) - 1) or (j != integration - 1):
                f.write(f"{best_pair[i][j]},")
    if len(signal_val_index_list) == 0:
        f.write(f"{best_pair[len(best_pair)-1][integration-1]}\n")
    else:
        f.write(f"{best_pair[len(best_pair)-1][integration-1]},")
        f.write(f"{single_line}\n")

    # 3行目以降に各信号線の故障候補値を列挙
    for i in range(len(best_pair)):
        for j in integration_list:
            idx = best_pair_index[i][j]
            for k in range(len(signal_val[idx]) - 1):
                f.write(f"{signal_val[idx][k]},")
            f.write(f"{signal_val[idx][-1]}\n")

    # 奇数の場合、最後の信号線を出力
    if len(signal_val_index_list) == 1:
        for i in range(len(signal_val[single_line_index]) - 1):
            f.write(f"{signal_val[single_line_index][i]},")
        f.write(f"{signal_val[single_line_index][-1]}\n")
        print("信号線の本数は奇数でした。")

end = time.time()
print(f"探索時間：{end - start:.4f}秒")
print("プログラムは終了しました")