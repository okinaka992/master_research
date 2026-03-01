# 正解データを分割するプログラム
# 正解データを0~1に正規化. 0~3の値を0~1に変換。0->0, 1->0.375, 2->0.625, 3->1 (※値がなるべく閾値から離れるように0は0のまま、3は1とした)
# cd workspace/research2/experiment
#　実行コマンド：　python3 learn_data_suplit.py

#グローバル変数
cir = 's5378'  # 対象回路
correct_data_file = cir + 'integrated_output'  # 正解データファイル
suplit_num = 54 # 何個ずつ分割するか
correct_data_folder = cir + '分割正解データ' # 正解データを分割して保存するフォルダ
suplit_correct_data_file = cir + 'integrated_output'  # 分割された正解データファイル
correct_data_value_file = cir + 'correct_value'  # 正解データの種類（値）を保存するファイル
threshold_file = cir + 'threshold'  # 閾値を保存するファイル
suplit_num_file = cir + 'suplit_num'  # 分割数を保存するファイル
single_line_file = cir + 'single_line' 
correct_value = [0.00, 0.25, 0.75, 1.00]  # 正解データの値
# correct_value = [-1, -0.33, 0.33, 1]  # 出力層の活性化関数がtanhの場合の正解データの値（0~1ではなく、-1~1に変換する必要があるため） # 0->-1, 1->-0.5, 2->0.5, 3->1
# correct_value = [0, 1, 2, 3]
threshold = [0.02, 0.5, 0.98]


# 正解データファイルを開いてデータを読み込む
with open(correct_data_file) as f:
  single_flag = int(f.readline().split()[0])  # 最適なペアが存在しない信号線があるかどうかを示す # 0:存在しない, 1:存在する

  # 2行目 統合後の信号線の数を読み込む(ペアが存在しない信号線の数も含む)
  signal_sum = int(f.readline())

  # 3行目（最適なペア信号線）を読み込む。カンマ区切りで各信号線をリストに格納
  best_pair = f.readline().split(',')

  # 4行目以降を読み込む。_にいれた各文字列の空白文字を削除
  correct_data = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]  # 各行の空白文字と改行文字を削除

correct_data = [list(_) for _ in correct_data]  # 1次元リストを2次元リストに変換

print(len(correct_data))
# print(correct_data[0])
# print(correct_data[0][0])
# print(type(correct_data[0]))

# 分割された正解データを格納するリスト
suplit_correct_data = [["0" for _ in range(len(correct_data[0]))] for _ in range(suplit_num)] # 2次元配列を0で初期化。列数は故障候補値の数（correct_dataの列数）、行数はsuplit_num=分割数

remainder_flag = 0  # signal_sumをsuplit_numで割り切れる亜銅貨を示すフラグ
remainder_num = 0  # 割り切れない数を格納する変数
if signal_sum % suplit_num != 0:  # signal_sumがsuplit_numで割り切れない場合
    remainder_flag = 1 # 割り切れないフラグを立てる
    remainder_num = signal_sum % suplit_num  # 割り切れない数を求める
output_node_num = suplit_num  # 部分モデルの出力ノード数（※最後の部分モデルは割り切れない場合があるので、考慮する必要あり）

correct_value_count = [0, 0, 0, 0]  # 正解データの値のカウント用リスト
for i in range((signal_sum//suplit_num)):  # 統合後の信号線数をsuplit_numで割る=モデルの分割数 「//」では計算結果が切り捨てられるので、+remainderをすることで、割り切れない場合も考慮する
    if (i == ((signal_sum//suplit_num)) - 1) and (remainder_flag == 1):  # 最後の部分モデルかつ、余りがある場合
        output_node_num = suplit_num + remainder_num  # 最後の部分モデルの出力ノード数は、suplit_numに余りとなるremainder_numを加えたものになる = 最後のモデルは出力ノードが増える
    with open(correct_data_folder + '/' + suplit_correct_data_file + str(i), 'w') as f:  # 分割された正解データを書き込むファイルを開く
        for j in range(len(correct_data)):
            # print("output_node_num:", output_node_num)
            # print("i:", i)
            for k in range(i*suplit_num, i*suplit_num + output_node_num):    # 分割数ごとに左から右にsuplit_num個ずつ正解データを書き込む.正解データファイルごとに、部分モデルの出力ノード数（output_node_num）だけ、左にずれるので、kの範囲はi*suplit_numからi*suplit_num + output_node_num となる。iは0から始まるので、単純に、i*suplit_numではなく、i*suplit_num + output_node_numにしなければならない。
                if (single_flag == 1) and (i == ((signal_sum//suplit_num) + remainder_flag) - 1) and (k == (i*suplit_num + output_node_num - 1)):  # 最適なペアが存在しない信号線が存在し、ペアが存在しない信号線だった場合 = 最後の部分モデルの最後の要素の場合（つまり、ペアが存在しない信号線の値は書き替えない）
                    if correct_data[j][k] == '0':
                        correct_data[j][k] = correct_value[0]
                        correct_value_count[0] += 1
                    elif correct_data[j][k] == '1':
                        correct_data[j][k] = correct_value[3]
                        correct_value_count[3] += 1
                else:
                    # 正解データを0~1に正規化. 0~3の値を0~1に変換。
                    if correct_data[j][k] == '0':
                        correct_data[j][k] = correct_value[0]
                        correct_value_count[0] += 1
                    elif correct_data[j][k] == '1':
                        correct_data[j][k] = correct_value[1]
                        correct_value_count[1] += 1
                    elif correct_data[j][k] == '2':
                        correct_data[j][k] = correct_value[2]
                        correct_value_count[2] += 1
                    elif correct_data[j][k] == '3':
                        correct_data[j][k] = correct_value[3]
                        correct_value_count[3] += 1

                if k < i*suplit_num + output_node_num - 1:  # 最後の要素以外はカンマを付ける
                    f.write(str(correct_data[j][k]) + ',')
                else:                                  # 最後の要素はカンマを付けない
                    f.write(str(correct_data[j][k]) + '\n')
    
model_suplit_num = str(int(i) + 1)  # 分割数をsuplit_numに格納
# print(suplit_num)

for i in range(len(correct_value_count)):
    print(f'正解データの値 {correct_value[i]} の数: {correct_value_count[i]}')  # 正解データの値の数を表示

# 正解データの値を保存するファイルを開く
with open(correct_data_folder + '/' + correct_data_value_file, 'w') as f:  # 正解データの値を保存するファイルを開く
    for i in range(len(correct_value)):
        f.write(str(correct_value[i]) + ' ')
    f.write('\n')  # 正解データの値を保存する

# 閾値を保存するファイルを開く
with open(correct_data_folder + '/' + threshold_file, 'w') as f:  #
    for i in range(len(threshold)):
        f.write(str(threshold[i]) + ' ')
    f.write('\n')  # 閾値を保存する

with open(correct_data_folder + '/' + cir + 'suplit_data_num', 'w') as f:  # データの分割数を保存するファイルを開く
    f.write(str(suplit_num) + '\n')  # データの分割数を保存する

with open(correct_data_folder + '/' + suplit_num_file, 'w') as f: # モデルの分割数を保存するファイルを開く
    f.write(model_suplit_num + '\n')

# 統合されていない信号線があるかどうかを保存 0:存在しない、1:存在する
with open(correct_data_folder + '/' + single_line_file, 'w') as f:
    f.write(str(single_flag) + '\n')
    if single_flag == 0:  # 最適なペアが存在しない信号線が存在する場合
        f.write('統合されていない信号線はありません。\n')
    else:  # 最適なペアが存在しない信号線が存在しない場合
        f.write('統合されていない信号線があります。\n')
        