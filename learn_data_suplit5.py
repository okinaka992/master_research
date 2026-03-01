# learn_data_suplit4.pyを変更したプログラム
# 正解データを分割する際に、貪欲法によって各部分モデルに均等に信号線を分配するように変更した
# learn_data_suplit.pyでは，最後の部分モデルに信号線の余りを全て入れていたが，learn_data_suplit4.pyでは，余りを各部分モデルに1つずつ分配するように変更した，
# この時，ノード数を増やす部分モデルは，最後の部分モデルから数えて余りの個数分となる．（例　モデルが5つ，余りが3の場合，最後のモデルとその一つ前のモデル,もうひとつ前部分モデルの出力ノード数を1つずつ増やす）
# 正解データを分割するプログラム
# 正解データを0~1に正規化. 0~3の値を0~1に変換。0->0, 1->0.375, 2->0.625, 3->1 (※値がなるべく閾値から離れるように0は0のまま、3は1とした)
# cd workspace/research2/experiment
#　実行コマンド：　python3 learn_data_suplit.py

#グローバル変数
cir = 's5378'  # 対象回路
correct_data_file = cir + 'integrated_output'  # 正解データファイル
suplit_num = 10 # 何個ずつ分割するか
correct_data_folder = cir + '分割正解データ' # 正解データを分割して保存するフォルダ
suplit_correct_data_file = cir + 'integrated_output'  # 分割された正解データファイル
correct_data_value_file = cir + 'correct_value'  # 正解データの種類（値）を保存するファイル
threshold_file = cir + 'threshold'  # 閾値を保存するファイル
suplit_num_file = cir + 'suplit_num'  # 分割数を保存するファイル
model_output_node_num_file = cir + 'model_output_node_num'  # 部分モデルの出力ノード数を保存するファイル
single_line_file = cir + 'single_line' 
pair_list_file = cir + 'pair_list'  # simular.ipynbで作成した信号線の統合情報を取得するファイル
pair_list2_file = cir + 'pair_list2'  # simular.ipynbで作成した信号線の統合情報を貪欲法で並び替えたものに対応させる
correct_value = [0.00, 0.25, 0.75, 1.00]  # 正解データの値
# correct_value = [-1, -0.33, 0.33, 1]  # 出力層の活性化関数がtanhの場合の正解データの値（0~1ではなく、-1~1に変換する必要があるため） # 0->-1, 1->-0.5, 2->0.5, 3->1
# correct_value = [0, 1, 2, 3]
threshold = [0.02, 0.5, 0.98]


# --- ここから挿入: 列分配の貪欲アルゴリズム ---
import random
from collections import Counter

def distribute_columns_greedy(correct_data, target_sizes, tries=10, seed=None):
    """
    correct_data: list of rows (each row is list/iterable of column values)
    target_sizes: list of desired column counts per split (len = num_splits)
    returns: splits (list of lists of column indices)
    """
    random.seed(seed)
    R = len(correct_data)
    C = len(correct_data[0])
    cols_all = list(range(C))
    best_splits = None
    best_score = None

    for t in range(tries):
        cols = cols_all[:]
        random.shuffle(cols)
        num_splits = len(target_sizes)
        splits = [[] for _ in range(num_splits)]
        # current signature per split: tuple per row
        cur_sig = [tuple(() for _ in range(R)) for _ in range(num_splits)]  # 初期状態では各スプリットのシグネチャは空
        cur_del = [0]*num_splits 

        for c in cols:
            best_s = None
            best_delta = None
            best_new_sig = None
            best_new_del = None

            for s in range(num_splits):
                if len(splits[s]) >= target_sizes[s]:
                    continue
                temp_sig = tuple(cur_sig[s][r] + (correct_data[r][c],) for r in range(R))
                cnt = Counter(temp_sig)
                new_del = sum(v-1 for v in cnt.values())
                delta = new_del - cur_del[s]
                if (best_delta is None) or (delta < best_delta):
                    best_delta = delta
                    best_s = s
                    best_new_sig = temp_sig
                    best_new_del = new_del

            if best_s is None:
                # 空きのある最短スプリットに入れる
                lengths = [len(x) for x in splits]
                best_s = min(range(num_splits), key=lambda i: lengths[i])
                best_new_sig = tuple(cur_sig[best_s][r] + (correct_data[r][c],) for r in range(R))
                cnt = Counter(best_new_sig); best_new_del = sum(v-1 for v in cnt.values())

            # commit
            splits[best_s].append(c)
            cur_sig[best_s] = best_new_sig
            cur_del[best_s] = best_new_del

        # 評価: ここでは平均削除数（小さいほど良い）
        score = sum(cur_del)/len(cur_del)
        if (best_score is None) or (score < best_score):
            best_score = score
            best_splits = splits

    return best_splits


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

print("signal_sum:", signal_sum)  # 統合された信号線の数

# モデルの分割数を計算
model_suplit_num = signal_sum // suplit_num  #　統合された信号線数を分割数で割る
print("model_suplit_num：", model_suplit_num)

# 分割された正解データを格納するリスト
suplit_correct_data = [["0" for _ in range(len(correct_data[0]))] for _ in range(suplit_num)] # 2次元配列を0で初期化。列数は故障候補値の数（correct_dataの列数）、行数はsuplit_num=分割数

remainder_flag = 0  # signal_sumをsuplit_numで割り切れる亜銅貨を示すフラグ
remainder_num = 0  # 割り切れない数を格納する変数
if signal_sum % suplit_num != 0:  # signal_sumがsuplit_numで割り切れない場合
    remainder_flag = 1 # 割り切れないフラグを立てる
    remainder_num = signal_sum % suplit_num  # 割り切れない数を求める

print("remainder_num:", remainder_num)
print("remainder_num//model_suplit_num:", remainder_num//model_suplit_num)

# 部分モデルの出力ノード数を計算
# 余りがある場合，余りの数remainder_numと同数の部分モデルの出力ノード数を1つずつ増やす．この時，ノード数を増やす部分モデルは，最後の部分モデルから数えてremainder_num個分となる．
output_node_num = [suplit_num] * model_suplit_num # 部分モデルの出力ノード数を格納するリスト．初期値はsuplit_numで初期化．余りがない場合は，このままの値を使用する．
if (remainder_num > model_suplit_num) and (remainder_flag == 1):  # 余りがあり，かつ，余りが部分モデルの数より大きい場合.全てのモデルの出力ノード数を1つずつ増やす
        for j in range(remainder_num//model_suplit_num):  # 余りの数が部分モデルの何倍かを計算し，倍数分だけ各部分モデルの出力ノード数を1つずつ増やす
            output_node_num = [x + 1 for x in output_node_num]
            remainder_num -= model_suplit_num

print("remainder_num:", remainder_num)
if (remainder_num > 0) and (remainder_flag == 1):  # 残りの部分モデル数が余り以下の場合、かつ、余りがある場合
    for i in range(model_suplit_num):
        output_node_num[model_suplit_num - 1 - i] += 1
        remainder_num -= 1
        if remainder_num == 0:
            break

print("output_node_num:", output_node_num)


# 呼び出し（output_node_num を target_sizes として使用）
# correct_data は既に list(list(...)) の形式で定義済み
splits = distribute_columns_greedy(correct_data, target_sizes=output_node_num, tries=10, seed=42)

# 確認表示（任意）
print("分配結果（各スプリットの列インデックス数）:", [len(s) for s in splits])


correct_value_count = [0, 0, 0, 0]  # 正解データの値のカウント用リスト
suplit_sum_count = 0  # 分割された正解データの信号線数のカウント用変数
for i in range(model_suplit_num):
    with open(correct_data_folder + '/' + suplit_correct_data_file + str(i), 'w') as f:
        cols = splits[i]  # このスプリットに割り当てられた列インデックスのリスト
        for j in range(len(correct_data)):
            for idx_c, c in enumerate(cols):
                # single_flag の特別処理は元のコードロジックを踏襲
                if (single_flag == 1) and (i == (model_suplit_num - 1)) and (idx_c == (len(cols) - 1)):
                    if correct_data[j][c] == '0':
                        correct_data[j][c] = correct_value[0]
                        correct_value_count[0] += 1
                    elif correct_data[j][c] == '1':
                        correct_data[j][c] = correct_value[3]
                        correct_value_count[3] += 1
                else:
                    if correct_data[j][c] == '0':
                        correct_data[j][c] = correct_value[0]
                        correct_value_count[0] += 1
                    elif correct_data[j][c] == '1':
                        correct_data[j][c] = correct_value[1]
                        correct_value_count[1] += 1
                    elif correct_data[j][c] == '2':
                        correct_data[j][c] = correct_value[2]
                        correct_value_count[2] += 1
                    elif correct_data[j][c] == '3':
                        correct_data[j][c] = correct_value[3]
                        correct_value_count[3] += 1

                # カンマと改行の制御
                if idx_c < len(cols) - 1:
                    f.write(str(correct_data[j][c]) + ',')
                else:
                    f.write(str(correct_data[j][c]) + '\n')
            
    suplit_sum_count += output_node_num[i]  # 分割された正解データの信号線数をカウント
    

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
    for i in range(len(output_node_num)):
        f.write(str(output_node_num[i]) + '\n')  # データの分割数を保存する

with open(correct_data_folder + '/' + suplit_num_file, 'w') as f: # モデルの分割数を保存するファイルを開く
    f.write(str(model_suplit_num) + '\n')

# 統合されていない信号線があるかどうかを保存 0:存在しない、1:存在する
with open(correct_data_folder + '/' + single_line_file, 'w') as f:
    f.write(str(single_flag) + '\n')
    if single_flag == 0:  # 最適なペアが存在しない信号線が存在する場合
        f.write('統合されていない信号線はありません。\n')
    else:  # 最適なペアが存在しない信号線が存在しない場合
        f.write('統合されていない信号線があります。\n')
        