#各信号線の故障候補値を比較、最も値が似通っている信号線ペア（ハミング距離最小のペア）を見つけるプログラム

import numpy as np
from scipy.spatial import distance

cir = 's38584'
correct_data = cir + 'output'
net_list = 'c' + cir
simular_output = cir + 'simular_output'
integration = 2

# 正解データを NumPy 配列として読み込む（高速化）
with open(correct_data) as f:
    lines = [line.strip() for line in f.readlines()]

# ネットリストから出力線数・入力線数を取得
with open('c' + cir, 'r') as f:
    output_line_num = int(f.readline().split()[1])
    input_line_num = int(f.readline().split()[1])

print(f"出力線数: {output_line_num}, 入力線数: {input_line_num}")
print(f"総行数: {len(lines)}")

# 各信号線のデータを列方向で抽出（転置）
signal_val = np.array([[lines[i][j] for i in range(len(lines))] for j in range(len(lines[0]))], dtype='U1')
signal_num = signal_val.shape[0]

print(f"信号線総数: {signal_num}")

# 診断対象信号線のインデックス（出力線以外）
line_num_list = list(range(input_line_num, signal_num))  # 出力線以外

# ハミング距離行列を一括計算（NumPy + SciPy）
print("ハミング距離を計算中...")
num_target = len(line_num_list)

# メモリが足りなければ、チャンク分割で計算
chunk_size = 1000  # 一度に処理するペア数
min_hamming = float('inf')
best_pair = None

for i in range(num_target):
    # i番目の信号線
    sig_i = signal_val[line_num_list[i]]
    
    # i+1番目以降の信号線とのハミング距離を一括計算
    if i % 100 == 0:
        print(f"処理中: {i}/{num_target}")
    
    for j in range(i + 1, num_target):
        sig_j = signal_val[line_num_list[j]]
        hamming_dist = distance.hamming(sig_i, sig_j)
        
        if hamming_dist < min_hamming:
            min_hamming = hamming_dist
            best_pair = (line_num_list[i], line_num_list[j], hamming_dist)

print(f"\n最小ハミング距離: {min_hamming}")
print(f"ペア: {best_pair[0]}, {best_pair[1]}")

# 結果を出力ファイルに書き込む
with open(simular_output, 'w') as f:
    f.write(f"{best_pair[0]} {best_pair[1]} {best_pair[2]}\n")

print("プログラムは終了しました")