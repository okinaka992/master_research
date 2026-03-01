# learn_data_suplit10.pyを変更したプログラム
# モデルのメモリサイズを計算するためだけに使用

# 正解データを分割する際に、貪欲法によって削除数がなるべく多くなるような組み合わせになるように変更した．その際，分割正解データ間の削除数の差は考慮しないようにした．
# learn_data_suplit.pyでは，最後の部分モデルに信号線の余りを全て入れていたが，learn_data_suplit4.pyでは，余りを各部分モデルに1つずつ分配するように変更した，
# この時，ノード数を増やす部分モデルは，最後の部分モデルから数えて余りの個数分となる．（例　モデルが5つ，余りが3の場合，最後のモデルとその一つ前のモデル,もうひとつ前部分モデルの出力ノード数を1つずつ増やす）
# 正解データを分割するプログラム
# 正解データを0~1に正規化. 0~3の値を0~1に変換。0->0, 1->0.375, 2->0.625, 3->1 (※値がなるべく閾値から離れるように0は0のまま、3は1とした)
# cd workspace/research2/experiment
#　実行コマンド：　python3 learn_data_suplit_test2.py


import learn_data_delite_test as ld   # 呼び出しプログラムを設定
import model_size_cal_test as msc

import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import random
import time

# 64bit 基数（ハッシュの基底）。uint64 オーバーフローを利用してハッシュ更新を行う
BASE = np.uint64(1315423911)

#グローバル変数
cir = 's1494'  # 対象回路
suplit_num = None # 何個ずつ分割するか
suplit_num_candidate = [10, 15, 20]  # 何個ずつ分割するかの候補
correct_data_file = cir + 'integrated_output'  # 正解データファイル
correct_data_folder = cir + '分割正解データ_test' # 正解データを分割して保存するフォルダ
suplit_correct_data_file = cir + 'integrated_output'  # 分割された正解データファイル
correct_data_value_file = cir + 'correct_value'  # 正解データの種類（値）を保存するファイル
threshold_file = cir + 'threshold'  # 閾値を保存するファイル
suplit_num_file = cir + 'suplit_num'  # 分割数を保存するファイル
model_output_node_num_file = cir + 'model_output_node_num'  # 部分モデルの出力ノード数を保存するファイル
single_line_file = cir + 'single_line' 
pair_list_file = cir + 'pair_list'  # simular.ipynbで作成した信号線の統合情報を取得するファイル
pair_list2_file = cir + 'pair_list2_test'  # simular.ipynbで作成した信号線の統合情報を貪欲法で並び替えたものに対応させる
single_line_inf_file = correct_data_folder + "/" + cir + 'single_line_inf'  # 統合されていない信号線の情報を保存するファイル
correct_value_count_file = correct_data_folder + "/" + cir + 'correct_value_count'  # 正解データの値の数を保存するファイル
model_and_line_file = correct_data_folder + "/" + cir + 'model_and_line'  # 各部分モデルが担当する信号線の情報を保存するファイル
correct_value = [0.00, 0.25, 0.75, 1.00]  # 正解データの値
# correct_value = [-1, -0.33, 0.33, 1]  # 出力層の活性化関数がtanhの場合の正解データの値（0~1ではなく、-1~1に変換する必要があるため） # 0->-1, 1->-0.5, 2->0.5, 3->1
# correct_value = [0, 1, 2, 3]
threshold = [0.02, 0.5, 0.98]


# # 各回路における分割数を設定
# if cir == 's1488':
#     suplit_num = 24
# elif cir == 's1494':
#     suplit_num = 15
# elif cir == 's5378':
#     suplit_num = 10
# elif cir == 's9234':
#     suplit_num = 20


# --- ここから挿入: 列分配の貪欲アルゴリズム ---
import random
from collections import Counter
import statistics   # 追加: 分散/標準偏差計算に使用

def prepare_value_array(correct_data):
    """
    correct_data: list of strings (rows). 変換して val_arr (R x C) uint8 を返す。
    各セルは 0..3 の整数と仮定。
    """
    # convert to 2D array of chars
    R = len(correct_data)
    C = len(correct_data[0]) if R>0 else 0
    val_arr = np.empty((R, C), dtype=np.uint8)
    for i, row in enumerate(correct_data):
        # row is iterable of chars or numbers
        for j, ch in enumerate(row):
            # 文字 '0','1','2','3' -> int
            if isinstance(ch, str):
                if ch.isdigit():
                    v = ord(ch) - ord('0')
                else:
                    v = 0
            else:
                v = int(ch)
            val_arr[i, j] = v
    return val_arr  # shape (R, C), dtype uint8


def run_single_try_hash(seed_val, val_arr, target_sizes, balance_weight, use_range):
    """
    1試行: ハッシュ方式で貪欲割当を実行する。
    val_arr: (R, C) uint8, rows=patterns, cols=columns
    returns (score, splits, cur_del_list)
    """

    # 何回目なのかを表示⇒trizeの回数呼び出される
    print(f"試行 {seed_val}回目 開始")

    np.random.seed(int(seed_val) & 0xffffffff)
    random.seed(int(seed_val) & 0xffffffff)

    R, C = val_arr.shape
    cols = np.arange(C)
    np.random.shuffle(cols)

    num_splits = len(target_sizes)
    splits = [[] for _ in range(num_splits)]

    # cur_hash: 各スプリットの per-row hash (R,) uint64
    cur_hash = [np.zeros(R, dtype=np.uint64) for _ in range(num_splits)]
    # cur_del: 各スプリットの重複数（R - unique_count）
    cur_del = np.array([R - 1] * num_splits, dtype=np.int64)  # initially all rows identical (empty signature)

    # precompute per-column contribution: map value -> uint64 small values
    # We use simply small integers for values (1..4), but to reduce collisions one could use random per-value constants per column.
    # Simple approach: use value+1 (so 0->1, 1->2, ...)
    # We'll do: contrib = (val + 1).astype(uint64)
    # If you want lower collision risk, use a random multiplier per column:
    rng = np.random.RandomState(0)
    col_mult = rng.randint(1, 2**61, size=C, dtype=np.uint64)  # random multiplier per column

    for c in cols:
        best_s = None
        best_delta = None
        best_new_hash = None
        best_new_del = None

        # column contribution as uint64 array per row: (val+1) * col_mult[c]
        # compute vectorized
        valcol = val_arr[:, c].astype(np.uint64) + np.uint64(1)
        contrib = (valcol * np.uint64(col_mult[c])).astype(np.uint64)

        for s in range(num_splits):
            if len(splits[s]) >= target_sizes[s]:
                continue
            # new_hash = cur_hash[s] * BASE + contrib  (uint64 overflow OK)
            # use numpy vectorized arithmetic
            new_hash = (cur_hash[s] * np.uint64(BASE) + contrib).astype(np.uint64)

            # count duplicates via np.unique on new_hash
            uniq, counts = np.unique(new_hash, return_counts=True)
            new_del = int(np.sum(counts - 1))
            delta = new_del - int(cur_del[s])

            # choose split that minimizes delta (i.e., reduces duplicates most)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_s = s
                best_new_hash = new_hash
                best_new_del = new_del
            elif delta == best_delta:
                # tie-breaker: prefer smaller spread (std) after assignment
                cand = cur_del.copy()
                cand[s] = new_del
                bestcand = cur_del.copy()
                bestcand[best_s] = best_new_del
                if use_range:
                    if (cand.max() - cand.min()) < (bestcand.max() - bestcand.min()):
                        best_s = s
                        best_new_hash = new_hash
                        best_new_del = new_del
                else:
                    if (cand.std() < bestcand.std()):
                        best_s = s
                        best_new_hash = new_hash
                        best_new_del = new_del

        if best_s is None:
            # fallback: put into shortest split
            lens = np.array([len(x) for x in splits])
            best_s = int(np.argmin(lens))
            valcol = val_arr[:, c].astype(np.uint64) + np.uint64(1)
            contrib = (valcol * np.uint64(col_mult[c])).astype(np.uint64)
            best_new_hash = (cur_hash[best_s] * np.uint64(BASE) + contrib).astype(np.uint64)
            uniq, counts = np.unique(best_new_hash, return_counts=True)
            best_new_del = int(np.sum(counts - 1))

        # commit
        splits[best_s].append(int(c))
        cur_hash[best_s] = best_new_hash
        cur_del[best_s] = best_new_del

    mean_del = float(np.mean(cur_del))
    spread = float(np.std(cur_del)) if (not use_range and len(cur_del) > 1) else float(np.max(cur_del)-np.min(cur_del))
    score = mean_del + balance_weight * spread

    # 何試行目が終了したのかを表示
    print(f"試行 {seed_val}回目 終了: スコア {score:.2f}")

    return (score, splits, cur_del.tolist())


def distribute_columns_greedy_hash_parallel(correct_data, target_sizes, tries, seed, num_workers, balance_weight, use_range):
    """
    全体ラッパ。correct_data は list(list) のまま受け取り val_arr を作成して共有的に使う。
    並列化は tries ごとに run_single_try_hash を実行する方法。
    """
    val_arr = prepare_value_array(correct_data)  # R x C uint8

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    base_seeds = [random.randint(0, 2**31-1) for _ in range(tries)]
    func = partial(run_single_try_hash, val_arr=val_arr, target_sizes=target_sizes, balance_weight=balance_weight, use_range=use_range)

    # Use Pool.map (fork on Linux yields COW for val_arr but ensure Pool is started after val_arr created)
    with Pool(processes=num_workers) as pool:
        results = pool.map(func, base_seeds)

    best = min(results, key=lambda x: x[0])
    return best[1], best[2]  # splits, cur_del



def main():
    # 探索範囲回（suplit_num_candidate回）繰り返し
    for candidate in range(len(suplit_num_candidate)):

        suplit_num = int(suplit_num_candidate[candidate])
    
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
        # balance_weight: 平均削除数と分散(差)のトレードオフ係数。大きいほど「分割正解データ間の削除数の差を小さくする」重みが大きくなる。0は、平均削除数の最大化のみを考慮。範囲に上限はない（実数）。実務上は 0〜数値（例 0〜2）がよく使われる。
        # use_range: Trueの場合、各スプリットの削除数の範囲（最大値 - 最小値）を最小化することを目的関数に含める。「最悪ケース差（最大差）」を評価。極端な偏り（あるスプリットだけ極端に大きい／小さい）を厳しく抑えたいときに有効。
        # use_range: Falseの場合、削除数の分散を最小化(正解データ間の削除データ数の差を最小化)することを目的関数に含める。
        print("\n=== 高速版の列分配を開始 ===")
        start_time = time.time()

        splits, cur_del = distribute_columns_greedy_hash_parallel(
            correct_data,
            output_node_num,
            tries=15, 
            seed=0, 
            num_workers=5,
            balance_weight=0.0, 
            use_range=False)

        elapsed_time = time.time() - start_time
        print(f"列分配完了：{elapsed_time:.2f}秒\n")
        
        # pair_list_fileの書き換えのために、リストの中身を昇順にソート
        for s in splits:
            s.sort()
        
        # 確認表示（任意）
        print("分配結果（各スプリットの列インデックス数）:", [len(s) for s in splits])
        with open(model_and_line_file, 'w') as f:
            f.write(str(suplit_num) + '\n')
            for i, s in enumerate(splits):
                f.write(f"スプリット {i}: 列インデックス {s}\n")
        
        # 貪欲法により、ペアの順序が変わっているため、pair_list2_fileを作成し、pair_list_fileの内容を貪欲法で並び替えたものに対応させる
        with open(pair_list_file, "r") as f:
            #１行目と２行目を読み込む
            lines = f.readlines()
            num_data = [_ for _ in lines[:2]]  # 1行目と2行目を取得
            pair_data = lines[2:]  # 3行目以降を取得
        
        with open(pair_list2_file, "w") as f:
            f.write(num_data[0])  # 1行目を書き込む
            f.write(num_data[1])  # 2行目を書き込む
            for i in range(len(splits)):
                for j in range(len(splits[i])):
                    f.write(pair_data[splits[i][j]])  # 3行目以降を書き込む
                    if pair_data[splits[i][j]] == pair_data[-1]:
                        single_idx = splits[i][j]  # 統合されていない信号線のインデックスを取得
                        single_line_model = i  # 統合されていない信号線が属するモデル番号を格納
        
        print("single_idx:", single_idx)
        print("single_line_model:", single_line_model)
        
        
        correct_value_count = [[0, 0, 0, 0] for _ in range(model_suplit_num)]  # 正解データの値のカウント用リスト
        suplit_sum_count = 0  # 分割された正解データの信号線数のカウント用変数
        for i in range(model_suplit_num):
            with open(correct_data_folder + '/' + suplit_correct_data_file + str(i), 'w') as f:
                cols = splits[i]  # このスプリットに割り当てられた列インデックスのリスト
                for j in range(len(correct_data)):
                    for idx_c, c in enumerate(cols):
                        # single_flag の特別処理は元のコードロジックを踏襲
                        if (single_flag == 1) and (i == single_line_model) and (c == single_idx):
                            if correct_data[j][c] == '0':
                                correct_data[j][c] = correct_value[0]
                                correct_value_count[i][0] += 1
                            elif correct_data[j][c] == '1':
                                correct_data[j][c] = correct_value[3]
                                correct_value_count[i][3] += 1
        
                            single_line_suplit_idx = idx_c # 統合されていない信号線の分割正解データ内でのインデックスを取得
                        else:
                            if correct_data[j][c] == '0':
                                correct_data[j][c] = correct_value[0]
                                correct_value_count[i][0] += 1
                            elif correct_data[j][c] == '1':
                                correct_data[j][c] = correct_value[1]
                                correct_value_count[i][1] += 1
                            elif correct_data[j][c] == '2':
                                correct_data[j][c] = correct_value[2]
                                correct_value_count[i][2] += 1
                            elif correct_data[j][c] == '3':
                                correct_data[j][c] = correct_value[3]
                                correct_value_count[i][3] += 1
        
                        # カンマと改行の制御
                        if idx_c < len(cols) - 1:
                            f.write(str(correct_data[j][c]) + ',')
                        else:
                            f.write(str(correct_data[j][c]) + '\n')
                    
            suplit_sum_count += output_node_num[i]  # 分割された正解データの信号線数をカウント
        
        
        with open(correct_value_count_file, 'w') as f:
            for i in range(model_suplit_num):
                for j in range(len(correct_value)):
                    f.write("モデル" + str(i+1) + ": " + str(correct_value_count[i][j]) + '\n')
                f.write('\n')
            
        
        sum_correct_value_count = [sum(col) for col in zip(*correct_value_count)]  # 各正解データの値の合計を計算
        for i in range(len(correct_value)):
            print(f'正解データの値 {correct_value[i]} の数: {sum_correct_value_count[i]}')  # 正解データの値の数を表示
        
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
        
        with open(single_line_inf_file, 'w') as f:  # 統合されていない信号線の情報を保存するファイルを開く
            if single_flag == 1:  # 統合されていない信号線が存在する場合
                f.write('1\n')  # 統合されていない信号線が存在することを示す
                f.write(str(single_line_model) + '\n')  # 統合されていない信号線が属するモデル番号(0始まり）を保存
                f.write(str(single_line_suplit_idx) + '\n')  # 統合されていない信号線の分割正解データ内でのインデックスを保存
            else:
                f.write('0\n')  # 統合されていない信号線が存在しないことを示す
        
        # 統合されていない信号線があるかどうかを保存 0:存在しない、1:存在する
        with open(correct_data_folder + '/' + single_line_file, 'w') as f:
            f.write(str(single_flag) + '\n')
            if single_flag == 0:  # 最適なペアが存在しない信号線が存在する場合
                f.write('統合されていない信号線はありません。\n')
            else:  # 最適なペアが存在しない信号線が存在しない場合
                f.write('統合されていない信号線があります。\n')

        ld.main(cir)  # 分割正解データ削除プログラムを呼び出す
        msc.main(cir) # モデルサイズ計算プログラムを呼び出す
        
        print(str(candidate) + "回目のプログラムは終了しました")


if __name__ == "__main__":
    main()