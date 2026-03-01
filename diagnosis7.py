# diagnosis6.pyを編集．learn_data_delite3.ipynbで分割したデータを用いて学習させたANNを用いて故障診断を行うプログラム
# pair_list2_fileを使えるように変更．
# if output_data[0][j] <= 0.5: を if output_data[0][j] <= threshold[1]: に変更
# 正解データにおける信号線ペアの分割方法が単純な先頭からの振り分けではなく、削除数が均等になるように分割したモデルを診断する
# 425行目を編集

# learn_data_suplit4.pyで分割したデータを用いて学習させたANNを用いて故障診断を行うプログラム
# データの分割の際，余ったデータを最後のモデルに全て追加するわけではなく，余ったデータを均等に各モデルに追加するように変更したモデルを診断する
# 学習済みのANNを用いて故障診断を行うプログラム
# cd workspace/research2/experiment
#　実行コマンド：　python3 diagnosis7.py

import numpy as np
import tensorflow as tf
import os

import random

#グローバル変数
cir = 's15850'  # 対象回路
tp_file = cir + '.vec'  # テストパターンファイル名
part_stdic_file = cir + "stdic_bi/aout_" #縮退故障辞書ファイル名の一部
part_brdic_file = cir + "brdic_bi/aout_" #ブリッジ故障辞書ファイルの一部
st_diagnosis_dir = cir + "diagnosis_st_data/" #縮退故障診断を行うためのデータを保存するフォルダ
br_diagnosis_dir = cir + "diagnosis_br_data/" #ブリッジ故障診断を行うためのデータを保存するフォルダ
target_fault_num = 30  # 故障診断対象の故障の数⇒全ての故障の中からtarget_fault_num個をランダムに選択する
correct_output_file = 'correct_output/' + cir + '_correct_output'  # 正常な回路の出力ファイル
input_data_original_file = cir + 'input'  # 分割元の入力データファイル
input_data_supulit_file = cir + '分割入力データ/' + cir + 'input'  # 分割された入力データファイル
correct_data_file = cir + '分割正解データ削除後/' + cir + 'integrated_output'  # 統合後の正解データファイル
correct_data_integrated_inf_file = cir + '分割正解データ削除後/' + cir + 'integrated_inf'  # 正解データの統合情報を保存するファイル
correct_data_value_file = cir + '分割正解データ' + '/' + cir + 'correct_value'  # 正解データの種類（値）を保存するファイル
threshold_file = cir + '分割正解データ' + '/' + cir + 'threshold'  # 閾値を保存するファイル
suplit_num_file = cir + '分割正解データ' + '/' + cir + 'suplit_num'  # モデルの分割数が保存されたファイル
suplit_data_num_file = cir + '分割正解データ' + '/' + cir + 'suplit_data_num'  # データの分割数が保存されたファイル
single_line_file = cir + '分割正解データ' + '/' + cir + 'single_line'  # 統合されていない信号線があるかが保存されたファイル
single_line_inf_file = cir + '分割正解データ' + '/' + cir + 'single_line_inf'  # 統合されていない信号線の情報を保存するファイル
single_line_model = None  # 統合されていない信号線があるモデル番号を格納するリスト
original_input_data_num = None  # 1個のモデルにおける分割前の学習データ数
suplited_input_data_num = None  # 1個のモデルにおける統合された学習データ数
output_node_num = None #　出力層のノード数＝分割数による
num_models = None  # 学習させるモデルの数
model_folder = None  # 学習済みモデルを保存するフォルダ
fault_line_num = None  # 故障診断対象の回路における故障信号線の総数
fault_type_sum = 12  # 故障の種類の総数
correct_value = None  # 正解データの値・種類
# correct_value = [0, 1, 2, 3]
threshold = None  # ANNの出力値を変換するための閾値
# threshold = [0.5, 1.5, 2.5]

all_models_learn = True  # 全てのモデルを学習させるかどうかのフラグ. True: 全てのモデルを学習させる, False: 1回にprocess個のモデルだけ学習させる

processes = 8  # 並列処理のプロセス数

def setting_original_input_data_num():
    # 入力データファイルを開いてデータを読み込む
    with open(input_data_original_file) as f:
        lines = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]

    # print(lines)
    
    global original_input_data_num
    original_input_data_num = int(len(lines))      #学習データ数を設定。学習データ数は入力データの行数


def mk_input_data(input_data_file):
    # 入力データファイルを開いてデータを読み込む
    with open(input_data_file) as f:
        lines = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]

    # print(lines)
    
    global suplited_input_data_num
    suplited_input_data_num = int(len(lines))      #学習データ数を設定。学習データ数は入力データの行数

    int_lines = [list(map(int, _)) for _ in lines]  #list要素の型をint型に変換

    return np.array(int_lines)


def mk_output_data(fname):
    # 正解データファイルを開いてデータを読み込む
    with open(fname) as f:
        lines = [_.replace("\n", "") for _ in f.readlines()]
    
    lines = [value.split(",") for value in lines] # カンマ区切りで各信号線をリストに格納
    
    # print("dsadsa")
    # print(lines[0])

    for i in range(len(lines)):
        for j in range(len(lines[i])):
            lines[i][j] = float(lines[i][j])
    
    # print("gaga")
    # print(lines[0])
    
    global output_node_num               #グローバル変数を書き換え
    output_node_num = int(len(lines[0]))      #出力ノード数を設定。出力ノード数は正解データの各行の要素数
    print("output_node_num:", output_node_num)


# 故障が発生する信号線をtarget_fault_num個ランダムに選択
def get_fault_target(fault_all_line, fault_flag):  # fault_flagは0なら縮退故障、1ならブリッジ故障を表す
    fault_target_line = random.sample(fault_all_line, target_fault_num) # 想定する故障信号線をランダムに選ぶ
    fault_target_line = sorted(fault_target_line) # 小さい番号順に並び替える

    if fault_flag == 0:  # 縮退故障の診断を行う場合
        fault_target_type = random.choices([0, 1], k=target_fault_num)  # 0縮退故障か1縮退故障かをランダムに決定する
        with open(st_diagnosis_dir + 'st_fault_line', 'w') as f: # 診断対象故障信号線番号をファイルに保存
            f.write('診断対象縮退故障数' + str(target_fault_num) + '\n')  # 故障の数をファイルに保存
            for i in range(target_fault_num):
                f.write(str(fault_target_line[i]) + ' ' + "sa " + str(fault_target_type[i]) + "\n")
    else:  # ブリッジ故障の診断を行う場合
        fault_target_type = random.choices(list(range(10)), k=target_fault_num)  # ブリッジ故障の種類(10個の中から)をランダムに決定する。0~9の整数値をランダムに選ぶ
        with open(br_diagnosis_dir + 'br_fault_line', 'w') as f:  # 診断対象故障信号線番号をファイルに保存
            f.write('診断対象ブリッジ故障数' + str(target_fault_num) + '\n')  # 故障の数をファイルに保存
            for i in range(target_fault_num):
                f.write(str(fault_target_line[i]) + ' ' + "br_type " + str(fault_target_type[i]) + "\n")

    print(fault_target_line)
    print(fault_target_type)

    return fault_target_line, fault_target_type


# 故障出力値を取得してファイルに保存する関数
def get_fault_output(tp_num, fault_target_line, fault_target_type, fault_flag): # fault_flagは0なら縮退故障、1ならブリッジ故障を表す
    # 縮退故障、ブリッジ故障の辞書ファイルのパスを指定
    if fault_flag == 0:  # 縮退故障の診断を行う場合
        part_dic_file = part_stdic_file  # 縮退故障辞書ファイルのパスを指定
        diagnosis_file = st_diagnosis_dir + 'fault_output'  # 縮退故障診断用の故障出力保存ファイル名
    else:  # ブリッジ故障の診断を行う場合
        part_dic_file = part_brdic_file  # ブリッジ故障辞書ファイルのパスを指定
        diagnosis_file = br_diagnosis_dir + 'fault_output'  # ブリッジ故障診断用の故障出力保存ファイル名
        print(fault_target_line)
        print(fault_target_type)

    # 診断対象故障の出力値をファイルに保存
    for i in range(tp_num):
        # print(f"故障出力値を取得中：テストパターン{i}")
        with open(part_dic_file + str(i), 'r') as f:    # 故障辞書ファイルを指定
            fault_inf = f.readline()  # 故障情報
            lines = f.readlines()  # 故障辞書ファイルの全行を読み込む

        # 各辞書におけるIDと出力値を取得
        fault_output = [0 for _ in range(target_fault_num)]  # 故障出力値を格納するリスト
        idx = 0
        br_type_count = 0  # ブリッジ故障の種類をカウントする変数
        for j in range(0, len(lines), 2):
            if fault_flag == 0:  # 縮退故障の診断を行う場合
                if fault_target_line[idx] == int(lines[j].split()[3]) and fault_target_type[idx] == int(lines[j].split()[5]): # ファイルの「id 0 Fault 1 sa 0」の行から、信号線番号と0、1縮退故障情報を取得して、比較
                    fault_output[idx] = lines[j+1].replace('\n', '')  # 故障出力値を取得
                    idx += 1
            else:  # ブリッジ故障の診断を行う場合
                if fault_target_line[idx] == int(lines[j].split()[3]) and fault_target_type[idx] == br_type_count:  # ファイルの「id 0 Br_flt 1 1 2」の行から、信号線番号を取得して、br_type_countの値から、10種類あるブリッジ故障の種類を計算して、比較
                    fault_output[idx] = lines[j+1].replace('\n', '')  # 故障出力値を取得
                    idx += 1
                br_type_count += 1  # ブリッジ故障の種類をカウントする
                if br_type_count == 10:
                    br_type_count = 0

            if idx == target_fault_num:  # 診断対象の故障数に達したら、ループを抜ける。これは、診断対象の故障信号線が小さい順に並び替えられているため、target_fault_num個の故障を取得したら、ループを抜ける
                break

        with open(diagnosis_file + str(i), 'w') as f:
            for j in range(len(fault_output)):
                f.write(str(fault_output[j]) + '\n')
    
    # print(f"fault_output：{fault_output}")


# パスフェイル情報を取得してファイルに保存する関数
def get_pass_fail(tp_num, fault_flag):  # fault_flagは0なら縮退故障、1ならブリッジ故障を表す
    if fault_flag == 0:  # 縮退故障の診断を行う場合
        fault_output_file = st_diagnosis_dir + 'fault_output'  # 縮退故障の故障出力を保存しているファイルのパスを指定
        pass_fail_file = st_diagnosis_dir + 'pass_fail'  # 縮退故障のパスフェイル保存ファイル名
    else:  # ブリッジ故障の診断を行う場合
        fault_output_file = br_diagnosis_dir + 'fault_output'  # ブリッジ故障の故障出力を保存しているファイルのパスを指定
        pass_fail_file = br_diagnosis_dir + 'pass_fail'  # ブリッジ故障のパスフェイル保存ファイル名


    #正常な回路出力を取得する
    correct_output_value = []  # 正常な回路出力を格納するリスト
    with open(correct_output_file, 'r') as f:
        lines = f.readlines()  # 正常な回路出力の全行を読み込む
        for i in range(1, len(lines), 2):
            correct_output_value.append(lines[i].replace('\n', ''))

    print(f"correct_output_value: {correct_output_value}")  # 正常な回路出力を表示
    # print(correct_output_value)
    
    # 各故障の種類における回路出力と正常な回路出力を比較して、パスフェイル情報を取得
    fault_output_value = [[0 for _ in range(tp_num)] for _ in range(target_fault_num)]  # 故障出力値を格納する二次元リスト fault_output_value[30個ランダムに選んだ故障][テストパターン数]。fault_output_value[2][4]には、ランダムに選んだ故障のうち2番目に選んだ故障が発生した回路にテストパターン4を入力したときの出力値が格納される
    for i in range(tp_num):
        with open(fault_output_file + str(i), 'r') as f:
            lines = f.readlines()  # fault_output~ファイルから故障出力値の全行を読み込む
            for j in range(target_fault_num):
                fault_output_value[j][i] = lines[j].replace('\n', '')
    
    # print(f"fault_output_value{fault_output_value}")
    
    # 故障出力値と正常な回路出力を比較して、パスフェイル情報を取得
    pass_fail = [[0 for _ in range(tp_num)] for _ in range(target_fault_num)]  # パスフェイル情報を格納するリスト。診断を行う際に使用する
    for i in range(target_fault_num):
        with open(pass_fail_file + str(i), 'w') as f:
            for j in range(tp_num):
                if correct_output_value[j] == fault_output_value[i][j]:
                    pass_fail_value = '0' # pass 正常な回路出力と一致
                    pass_fail[i][j] = 0
                else:
                    pass_fail_value = '1' # fail 故障が発生した回路の出力値と正常な回路出力が一致しない
                    pass_fail[i][j] = 1
                
                f.write(pass_fail_value)

    # print(f"pass_fail{pass_fail}")

    return pass_fail  # パスフェイル情報を返す


#  故障候補を取得する関数
def get_fault_candidate(pass_fail, compare_model_pass_fail, fault_all_line, signal_num, br_missing_line, fault_flag): # fault_flagは0なら縮退故障、1ならブリッジ故障を表す
    if fault_flag == 0:  # 縮退故障の診断を行う場合
        fault_type_num = 2  # 縮退故障の種類は0縮退故障と1縮退故障の2種類
    else:  # ブリッジ故障の診断を行う場合
        fault_type_num = 10  # ブリッジ故障の種類は0~9の10種類
        signal_num = signal_num - len(br_missing_line)  # ブリッジ故障の診断を行う場合、ブリッジ故障が発生しない信号線を除外するため、signal_numを更新する
        # print(f"signal_num{signal_num}")
        # print(f"pass_fail{len(pass_fail)}")


    # 縮退故障のパスフェイル情報を比較
    fault_candidate = [[] for _ in range(target_fault_num)] # 故障候補を格納するリスト
    fault_type = [[] for _ in range(target_fault_num)] # 縮退故障の種類を格納するリスト 縮退故障の場合（0=0縮退故障、1=1縮退故障）。ブリッジ故障の場合（0~9=ブリッジ故障の種類）
    # print(f"fault_all_line{len(fault_all_line)}")
    for i in range(target_fault_num):
        if all( x == 0 for x in pass_fail[i]) == True:  #区別できない故障の場合（＝全てのパスフェイルが0の場合）は、故障候補が無数にある可能性があるため、評価には使用しない
            fault_candidate[i].append(-1)  # 故障候補に-1を設定
            fault_type[i].append(-1)
            print(f"fault_candidate：{fault_candidate[i]}")  # 故障候補が無数にあることを表示
            continue  # 次の故障に進む
        line_count = 0
        count = 0
        for j in range(signal_num*fault_type_num):  # signal_numは、診断対象の回路における信号線の総数。fault_type_numは、縮退故障の場合は2、ブリッジ故障の場合は10
            if compare_model_pass_fail[j] == pass_fail[i]:  # パスフェイル情報を比較
                # print(line_count, len(st_fault_candidate[i]), i, j, count)
                fault_candidate[i].append(fault_all_line[line_count])  # 故障候補を格納するリストに追加
                fault_type[i].append(count%fault_type_num)  # 故障の種類を格納するリストに追加
            
            count += 1
            if count == fault_type_num:  # st_compare_model_pass_failは、2行ごとが0または1縮退故障の信号線に対応しているため、2行ごとにカウントをリセットする
                line_count += 1
                count = 0
        
        if len(fault_candidate[i]) == 0:  # 故障候補が無い場合は、0を設定
            fault_candidate[i].append(0)  # 故障候補が無い場合は、0を設定
            fault_type[i].append(0)  # 故障の種類も0を設定
        print(f"fault_candidate[{i}]: {len(fault_candidate[i])}")  # 故障候補数を表示
    
    return fault_candidate, fault_type  # 故障候補と故障の種類を返す


# 故障候補の中に、診断対象の故障信号線が含まれているかを確認する関数
def check_fault_candidate(fault_candidate, fault_type, fault_target_line, fault_target_type, fault_flag): # fault_flagは0なら縮退故障、1ならブリッジ故障を表す
    if fault_flag == 0:  # 縮退故障の診断を行う場合
        correct_fault_candidate_file = st_diagnosis_dir + 'correct_fault_candidate'  # 正しい故障候補を保存するファイル名
        incorrect_fault_candidate_file = st_diagnosis_dir + 'incorrect_fault_candidate'  # 間違った故障候補を保存するファイル名
        diagnosis_rate_file = st_diagnosis_dir + 'diagnosis_rate'  # 診断率を保存するファイル名
        fault_name = '縮退故障'  # 故障の名前
        fault_symbol = ' sa '  # 縮退故障の記号
    else:  # ブリッジ故障の診断を行う場合
        correct_fault_candidate_file = br_diagnosis_dir + 'correct_fault_candidate'  # 正しい故障候補を保存するファイル名
        incorrect_fault_candidate_file = br_diagnosis_dir + 'incorrect_fault_candidate'  # 間違った故障候補を保存するファイル名
        diagnosis_rate_file = br_diagnosis_dir + 'diagnosis_rate'  # 診断率を保存するファイル名
        fault_name = 'ブリッジ故障'  # 故障の名前
        fault_symbol = ' br_type '  # ブリッジ故障の記号
    
    print(f"{fault_name}の診断を行います")

    print(f"fault_target_line:{fault_target_line}")

    find_count = 0  # 見つけられた故障の数
    not_find_count = 0  # 見つけられなかった故障の数
    indistinguishable_fault_count = 0  # 区別できない故障の数
    fault_line_candidate_sum = 0  # 故障候補の総数（故障信号線だけの総数）
    fault_type_candidate_sum = 0  # 故障候補の総数（故障信号線と故障の種類合わせた総数）
    sum_time = 0  # 診断時間の総和
    with open(correct_fault_candidate_file, 'w') as f:
        with open(incorrect_fault_candidate_file, 'w') as g:
            for i in range(target_fault_num):
                # 診断時間を計測開始
                import time
                start = time.time()
                fault_candidate[i] = sorted(fault_candidate[i]) # 小さい番号順に並び替える
                if fault_candidate[i] == [-1]:  # 故障候補が無数にある場合は、評価には使用しない
                    indistinguishable_fault_count += 1
                    print(f"区別できない故障のため、故障信号線{fault_target_line[i]} の診断は行いませんでした")
                    f.write("区別できない故障のため、故障信号線" + str(fault_target_line[i]) + " の診断は行いませんでした\n")
                    g.write("区別できない故障のため、故障信号線" + str(fault_target_line[i]) + " の診断は行いませんでした\n")
                    continue  # 次の故障に進む
                if fault_candidate[i] == [0]: # 故障候補が無い場合は、0を設定
                    print(f"実際に選んだ故障 {fault_target_line[i]} の故障候補は0でした")
                    g.write("故障候補が0だった故障：" + str(fault_target_line[i]) + fault_symbol + str(fault_target_type[i]) + '\n')
                    not_find_count += 1
                    continue
                
                fault_line_candidate_sum += len(set(fault_candidate[i]))  # 信号線のみの故障候補の総数を更新
                fault_type_candidate_sum += len(fault_candidate[i])  # 信号線と故障の種類合わせた故障候補の総数を更新
                for j in range(len(fault_candidate[i])):
                    if fault_candidate[i][j] == fault_target_line[i] and fault_type[i][j] == fault_target_type[i]:
                        find_count += 1
                        end = time.time()
                        elapsed_time = end - start
                        sum_time += elapsed_time
                        print(f"{fault_name}候補の中に実際に選んだ{fault_name}番号 {fault_candidate[i][j]} の {fault_type[i][j]} {fault_name}が含まれていました")
                        print(f"診断時間：{elapsed_time}秒")
                        f.write("故障候補（故障信号線のみ）：" + str(len(set(fault_candidate[i]))) + '\n')
                        f.write("故障候補数（故障の種類ごと）：" + str(len(fault_candidate[i])) + '\n')
                        f.write(str(fault_candidate[i][j]) + fault_symbol + str(fault_type[i][j]) + '\n')
                        break
                    if j == len(fault_candidate[i]) - 1:
                        print(f"故障候補の中に実際に選んだ故障 {fault_target_line[i]} の {fault_target_type[i]} は含まれていませんでした")
                        g.write("見つけられなかった故障：" + str(fault_target_line[i]) + fault_symbol + str(fault_target_type[i]) + '\n')
                        not_find_count += 1

    print("見つけられた故障の数：" + str(find_count))
    print("見つけられなかった故障の数：" + str(not_find_count))
    print("区別できない故障の数：" + str(indistinguishable_fault_count))
    print("平均故障候補数（故障信号線のみ）：" + str(fault_line_candidate_sum/(target_fault_num - indistinguishable_fault_count)))
    print("平均故障候補数（故障の種類ごと）：" + str(fault_type_candidate_sum/(target_fault_num - indistinguishable_fault_count)))
    print(fault_name + "の診断率：" + str((find_count/(target_fault_num - indistinguishable_fault_count))*100) + "%")
    print("平均診断時間：" + str(sum_time/(target_fault_num - indistinguishable_fault_count)) + "秒")

    import datetime as dt
    datetime = dt.datetime.now()
    with open(diagnosis_rate_file, 'a') as f:
        f.write("\n")
        f.write(str(datetime) + '\n')
        f.write("実行プログラム：" + os.path.basename(__file__) + '\n')
        f.write("見つけられた故障の数：" + str(find_count) + '\n')
        f.write("見つけられなかった故障の数：" + str(not_find_count) + '\n')
        f.write("区別できない故障の数：" + str(indistinguishable_fault_count) + '\n')
        f.write("平均故障候補数（故障信号線のみ）：" + str(fault_line_candidate_sum/(target_fault_num - indistinguishable_fault_count)) + '\n')
        f.write("平均故障候補数（故障の種類ごと）：" + str(fault_type_candidate_sum/(target_fault_num - indistinguishable_fault_count)) + '\n')
        f.write(fault_name + "の診断率：" + str((find_count/(target_fault_num - indistinguishable_fault_count))*100) + "%\n")
        f.write("平均診断時間：" + str(sum_time/(target_fault_num - indistinguishable_fault_count)) + "秒\n")



if __name__ == '__main__':

    # 全てのモデルを学習させるか、process個のモデルだけ学習させるかを設定
    if all_models_learn:
        model_folder = cir + 'sepmodel2'  # 全てのモデルを学習させる場合に、学習済みモデルを保存するフォルダ
        # 学習させるモデルの数=分割されたデータの数を取得＝学習させるモデルの数
    else:
        model_folder = cir + 'sepmodel_check2'  # process個のモデルだけ学習させる場合に、学習済みモデルを保存するフォルダ

    # 正解データの種類（値）を取得
    with open(correct_data_value_file, 'r') as f:
        correct_value = [float(value) for value in f.readline().split()]
    
    # 閾値を取得
    with open(threshold_file, 'r') as f:
        threshold = [float(value) for value in f.readline().split()]
    
    print("正解データの種類（値）：", correct_value)
    print("閾値：", threshold)
    
    # テストパターン数を取得
    with open(tp_file, 'r') as f:
        line = f.readline()
        tp_num = int(line.split()[0])  # テストパターン数
        print(tp_num)

    # 故障診断対象の回路における「縮退故障」の総数を取得
    with open(part_stdic_file + str(0), 'r') as f:  # 縮退故障辞書ファイルの1つを開く
        fault_inf = f.readline()  # 故障情報
        st_fault_num = int(fault_inf.split()[2])  # 対象回路で起こりうる故障数
        lines = f.readlines()[::2]  # 故障辞書ファイルのid情報を読み込む
        st_fault_all_line = [int(line.split()[3]) for line in lines[::2]]  # 縮退故障信号線番号を取得
        # print(st_fault_num)
        # print("st_fault_all_line：", st_fault_all_line)
        # print(suplit_num)

    # 故障診断対象の回路における「ブリッジ故障」の総数を取得
    with open(part_brdic_file + str(0), 'r') as f:  # ブリッジ故障辞書ファイルの1つを開く
        fault_inf = f.readline()  # 故障情報
        br_fault_num = int(fault_inf.split()[2]) # 対象回路で起こりうる故障数
        lines = f.readlines()[::2]  # 故障辞書ファイルのid情報を読み込む
        br_fault_all_line = [int(line.split()[3]) for line in lines[::10]]  # ブリッジ故障が発生する可能性がある信号線番号（支配される信号線の番号）をリストに格納　※縮退故障と異なり、出力信号線以外にもブリッジ故障が発生しない信号線があるため、故障辞書から読み込まないといけない
        # br_fault_type = [int(line.split()[4]) for line in lines[::2]] # ブリッジ故障における故障の種類（0、1どちらに支配されるのか）を取得
        # br_dominate_line = [int(line.split()[5]) for line in lines[::2]]  # 支配する信号線を格納
        # print(br_fault_num)
        # print("br_fault_all_line：", br_fault_all_line)
    

    # 診断対象となる縮退故障、ブリッジ故障それぞれに対する対象信号線番号を30個ランダムにそれぞれ取得
    st_fault_target_line, st_fault_target_type = get_fault_target(st_fault_all_line, 0) # 縮退故障が発生する信号線をtarget_fault_num個ランダムに選択。さらに、その信号線で発生するのは0縮退故障か1縮退故障かをランダムに決定する
    br_fault_target_line, br_fault_target_type = get_fault_target(br_fault_all_line, 1) # ブリッジ故障が発生する信号線をtarget_fault_num個ランダムに選択。さらに、その信号線で発生するのは0縮退故障か1縮退故障かをランダムに決定する

    # st_fault_target_line = [12, 62, 75, 90, 94, 111, 120, 174, 176, 262, 271, 288, 306, 323, 346, 347, 351, 417, 424, 457, 483, 601, 608, 617, 651, 657, 702, 706, 772, 782]
    # br_fault_target_line = [147, 154, 183, 348, 389, 398, 420, 431, 473, 574, 598, 739, 768, 784, 797, 814, 844, 939, 1016, 1089, 1128, 1218, 1310, 1315, 1332, 1365, 1386, 1405, 1487, 1519]

    print("診断対象の出力値をファイルに保存します")
    # 診断対象故障の出力値をファイルに保存
    get_fault_output(tp_num, st_fault_target_line, st_fault_target_type, 0) # 縮退故障時の故障出力値をファイルに保存
    get_fault_output(tp_num, br_fault_target_line, br_fault_target_type, 1) # ブリッジ故障時の故障出力値をファイルに保存
    print("診断対象の出力値の保存が完了しました")

    print("パスフェイル情報を取得します")
    # パスフェイル情報を取得してファイルに保存
    st_pass_fail = get_pass_fail(tp_num, 0)  # 縮退故障のパスフェイル情報を取得してファイルに保存
    br_pass_fail = get_pass_fail(tp_num, 1)  # ブリッジ故障のパスフェイル情報を取得してファイルに保存
    print("パスフェイル情報の取得が完了しました")

    # for i in range(len(st_pass_fail)):
    #     print(f"縮退故障のパスフェイル情報{st_pass_fail[i]}")
    
    
  # 学習済みの機械学習モデルに入力データを与えて、出力を取得する
    # 分割前の入力データ数を取得
    setting_original_input_data_num()

    # 分割モデルの数を取得
    with open(suplit_num_file, 'r') as f:
        num_models = int(f.readline())
    
    # 診断対象の回路における故障信号線の総数を取得
    with open(cir + 'output', 'r') as f:
        line = f.readline().replace(",", "").replace("\n", "")
        global faulet_line_num
        fault_line_num = int(len(line))  # 診断対象の故障信号線の総数
        # print("fault_line_num" + str(fault_line_num))

    # 故障の種類ごとに、パスフェイル情報を格納するリストを作成
    model_pass_fail = [[0 for _ in range(fault_line_num)] for _ in range(original_input_data_num)]  # 故障出力値を格納するリスト fault_output_value[30個ランダムに選んだ故障][テストパターン数]。fault_output_value[2][4]には、ランダムに選んだ故障のうち2番目に選んだ故障が発生した回路にテストパターン4を入力したときの出力値が格納される
    print("model_pass_fail:", len(model_pass_fail), len(model_pass_fail[0]))  # model_pass_failのサイズを表示

    # ネットリストを開いて、入力信号線数、出力信号線数、その他の信号線数を取得
    with open('c' + cir, 'r') as f:
        line_inf = f.readline()  # 1行目はネットリストの情報
        output_line_num = int(line_inf.split()[1])
        input_line_num = int(line_inf.split()[2])

    
    # 統合されていない信号線があるかどうかを取得 0: 統合されてない信号線がない、1: 統合されていない信号線がある
    # 統合されていない信号線がある場合、最後のモデルの最後の出力ノードの値は統合されていない
    with open(single_line_file, 'r') as f:
        single_flag = int(f.readline().replace("\n", ""))

    with open(cir + 'pair_list2', 'r') as f:     # learn_data_delite3.ipynbで再作成した信号線の統合情報を取得
        integration_num = int(f.readline().split()[1])  # 統合数
        # print(integration_num)
        signal_num = int(f.readline().split()[1])  # 信号線の数
        print("signal_num", signal_num)
        raw_lines = [l.strip() for l in f.readlines() if l.strip()]
        print(raw_lines)
    
    # 各行を数値リスト（グループ）として保持。途中に単独行が混ざっていても対応。
    groups = []
    for l in raw_lines:
        toks = l.split()
        nums = [int(x) for x in toks]
        if len(nums) == integration_num or len(nums) == 1: # 行に1トークンしかない場合、またはintegration_numトークンがある場合
            groups.append(nums)
        else:
            # 行に複数トークンがあり，integration_num単位で分割したい場合
            for i in range(0, len(nums), integration_num):
                groups.append(nums[i:i+integration_num])
    
    # 必要に応じて groups は ['2956','4964'], ['5487'], ... のような形になる
    # flat に信号線ペアのIDをリストに格納（元の順序に沿った一次元リスト）
    line_id = []
    for g in groups:
        line_id.extend(g)
    
    print("len(line_id):", len(line_id))
    # print("line_id (before adjustment):", line_id)
    
    # 検査
    if len(line_id) != signal_num:
        print(f"警告: 期待する信号線数 {signal_num} と実際のトークン数 {len(line_id)} が一致しません。")
    
    if single_flag == 1:  # 統合されていない信号線がある場合
        with open(single_line_inf_file, 'r') as f:
            line = f.readline().replace("\n", "")
            single_line_model = int(f.readline().replace("\n", ""))         # 統合されていない信号線が含まれるモデル番号
            single_line_suplit_idx = int(f.readline().replace("\n", ""))    # 統合されていない信号線が含まれる分割データのインデックス番号

    
    # print("line_id:", line_id)
    # if signal_num % integration_num != 0:  # 信号線の数が奇数の場合、ペアができていない信号線のIDをリストに追加
    #     for i in range(signal_num % integration_num):
    #         line_id[len(lines) - (signal_num % integration_num) + i] = int(lines[len(lines) - (signal_num % integration_num)][i])
    
    # line_idで格納されている値は、信号線番号であるため、入力信号線でも出力信号線でもない信号線番号から出力信号線数を引いておく
    for i in range(len(line_id)):
        if line_id[i] > input_line_num + output_line_num:
            line_id[i] = line_id[i] - output_line_num

    line_id = [i - 1 for i in line_id] # line_idは1から始まる信号線番号であるため、0から始まるようにする
    print(line_id)
    print(len(line_id))
    print(line_id[0])  # 最後の要素を表示
    print(line_id[1])  # 最後の要素を表示
    print(line_id[-1])  # 最後の要素を表示

    # データを何個づつ分割したのかを取得
    with open(suplit_data_num_file, 'r') as f:
        suplit_data_num = f.readlines() # データの分割数を取得
        for i in range(len(suplit_data_num)):
            suplit_data_num[i] = suplit_data_num[i].replace("\n","")
        print(suplit_data_num)


    # print(f"input_data_num:{input_data_num}")

    # temp = [[0 for _ in range(int(fault_line_num/2) + single_flag)] for _ in range(input_data_num)]  # 故障出力値を格納するリスト 

    print("今からモデルごとに入力データを与えて、出力データを取得します")

    # 診断時間を測定
    import time
    start = time.time()

  # モデルごとに入力データを与えて、出力データを取得
    suplit_data_sum_count = 0  # 変換された正解データの総数をカウントする変数
    for model_id in range(num_models):

        input_data_file = input_data_supulit_file + str(model_id)  # 分割された入力データファイル名に変更　＝　s344分割入力データ/s344integrated_input + 番号
        input_data = mk_input_data(input_data_file)  # 分割された入力データを取得 関数内でグローバル変数input_data_numを設定

        # 正解データの統合情報を取得
        with open(correct_data_integrated_inf_file + str(model_id), 'r') as f:
            correct_data_integrated_inf = [_.replace("\n", "").split(",") for _ in f.readlines()]  # 正解データの統合情報を取得

        # モデルの読み込み
        with open(model_folder + '/' + cir + 'model_' + str(model_id) + '.tflite', 'rb') as f:  # モデルを読み込むファイルを開く
            tflite_model = f.read()

        # モデルの評価
        interpreter = tf.lite.Interpreter(model_content=tflite_model)  # TFLite形式のモデルを読み込む。保存されたTFLiteモデルをメモリに読み込み、推論を行う準備をするためのインタープリターを作成します。
        interpreter.allocate_tensors()  # #メモリを確保。モデルが使用するテンソル（データ構造）をメモリに割り当てます。TFLiteモデルをインタープリターにロードするだけでは、テンソルのメモリは割り当てられていません。この行を実行することで、モデルが推論に必要なメモリを確保します。

        input_details = interpreter.get_input_details()  # モデルの入力テンソルの詳細を取得。モデルの入力に関する詳細情報をリスト形式で返します。各要素は、入力テンソルの形状、データ型、名前などの情報を含む辞書です。
        output_details = interpreter.get_output_details()  # モデルの出力テンソルの詳細を取得。モデルの出力に関する詳細情報をリスト形式で返します。各要素は、出力テンソルの形状、データ型、名前などの情報を含む辞書です。

        correct_file = correct_data_file + str(model_id)  # 分割された正解データファイル名に変更　＝　s344分割正解データ/s344integrated_output + 番号
        mk_output_data(correct_file)  # 関数内でグローバル変数output_node_numを設定
        
        n = 0
        correct_data_integrated_inf_count = 0  # correct_data_integrated_infのカウント変数
        for i in range(suplited_input_data_num):
            
            input_shape = input_details[0]['shape']  # 入力データの形状を取得.先ほど取得したinput_detailsのリストの中から、形状情報を取得。input_details[0]['shape']は、入力テンソルの形状を取得するためのコードです。['shape']は、辞書内の shape キーを指定しています。

            reshape_input_data = np.reshape(input_data[i], input_shape)  # NumPy配列の形状を指定した形 (input_shape) に変更します。これにより、入力データの形状がモデルの入力テンソルの形状と一致します。


            # モデルの入力データを設定
            interpreter.set_tensor(input_details[0]['index'], np.array(reshape_input_data, dtype=np.float32))   # モデルの入力テンソルにデータを設定します。入力テンソルのインデックス、データを指定します。[0]['index']は、入力テンソルのインデックスを取得するためのコードです。

            # モデルの推論を実行
            interpreter.invoke()

            # モデルの出力データを取得
            output_data = interpreter.get_tensor(output_details[0]['index'])
        

            # 出力データを閾値で分類して、正しい故障出力値に変換
            if (model_id == single_line_model) and single_flag == 1:  # 統合されていない信号線があり，統合されていない信号線を学習したモデルの場合
                for j in range(output_node_num):
                    if j != single_line_suplit_idx:  # 統合されていない信号線の出力ノード以外の場合
                        if output_data[0][j] < threshold[0]:
                            output_data[0][j] = correct_value[0]
                        elif output_data[0][j] < threshold[1]:
                            output_data[0][j] = correct_value[1]
                        elif output_data[0][j] < threshold[2]:
                            output_data[0][j] = correct_value[2]
                        else:
                            output_data[0][j] = correct_value[3]
                    else:  # 統合されていない信号線がある場合、その信号線を学習した出力ノードは統合されていない信号線の値を持つため、0または1に変換する
                        if output_data[0][j] <= threshold[1]:
                            output_data[0][j] = correct_value[0]
                        else:
                            output_data[0][j] = correct_value[3]
                        # print(f"output_data[0][{j}]: {output_data[0][j]}")  # 最後の出力ノードの値を表示
            else:  # 統合されていない信号線がない場合、または統合されていない信号線はあるが統合された信号線を学習したモデル以外の場合
                for j in range(output_node_num):
                    if output_data[0][j] < threshold[0]:
                        output_data[0][j] = correct_value[0]  # 0縮退故障の値
                    elif output_data[0][j] < threshold[1]:
                        output_data[0][j] = correct_value[1]  # 1縮退故障の値
                    elif output_data[0][j] < threshold[2]:
                        output_data[0][j] = correct_value[2]  # ブリッジ故障の値
                    else:
                        output_data[0][j] = correct_value[3]  # 正常な回路の値
            

            # 信号線の統合を解いて、故障出力値をmodel_pass_failに格納
            for integ_num in range(len(correct_data_integrated_inf[i])):
                # temp_count = 0  # 故障出力値を格納するためのカウント変数
                count = 0  # 統合したものの統合を解くために必要なカウント変数
                idx = int(correct_data_integrated_inf[i][integ_num])  # correct_data_integrated_infの要素は文字列として読み込まれているため、intに変換
                if model_id == single_line_model and single_flag == 1:  # 統合されていない信号線があり，統合されていない信号線を学習したモデルの場合
                    for j in range(output_node_num):
                        if j != single_line_suplit_idx:  # 統合されていない信号線の出力ノード以外の場合
                            if output_data[0][j] == correct_value[0]:  # 0縮退故障の値の場合
                                # idxには，統合された信号線の行番号が格納されており，model_pass_failの行はテストパターン番号に対応しているため，model_pass_fail[idx]で，統合された信号線の行を指定している
                                model_pass_fail[idx][line_id[j + count + suplit_data_sum_count]]= 0
                                model_pass_fail[idx][line_id[j + count + suplit_data_sum_count + 1]] = 0
                            elif abs(output_data[0][j] - correct_value[1]) < 1e-2:  # floatの比較は、誤差を考慮してabsを使用
                                model_pass_fail[idx][line_id[j + count + suplit_data_sum_count]]= 1
                                model_pass_fail[idx][line_id[j + count + suplit_data_sum_count + 1]] = 0
                            elif abs(output_data[0][j] - correct_value[2]) < 1e-2:
                                model_pass_fail[idx][line_id[j + count + suplit_data_sum_count]]= 0
                                model_pass_fail[idx][line_id[j + count + suplit_data_sum_count + 1]] = 1
                            else:
                                model_pass_fail[idx][line_id[j + count + suplit_data_sum_count]]= 1
                                model_pass_fail[idx][line_id[j + count + suplit_data_sum_count + 1]] = 1
                            # temp[idx][temp_count+suplit_data_num*model_id] = output_data[0][j]  # 故障出力値を格納
                            # temp_count += 1
                            count += 1
                        else:  # 統合されていない信号線の出力ノードの場合
                            # print("line_id", line_id[j + count + suplit_data_sum_count])
                            # print("j", j)

                            if output_data[0][j] == correct_value[0]:  # 最後の出力ノードの値が0の場合
                                model_pass_fail[idx][line_id[j + count + suplit_data_sum_count]] = 0  # 統合されていない信号線の値をそのまま格納
                            elif output_data[0][j] == correct_value[3]:  # 最後の出力ノードの値が1縮退故障の値の場合
                                model_pass_fail[idx][line_id[j + count + suplit_data_sum_count]] = 1
                else:  # 統合されていない信号線がない場合、または統合された信号線はあるが，統合されていない信号線を学習したモデル以外の場合
                    for j in range(output_node_num):
                        # print(j + count + suplit_data_sum_count)
                        # print(f"line_id: {line_id[j + count + suplit_data_sum_count]}")
                        if output_data[0][j] == correct_value[0]:  # 0縮退故障の値の場合
                            model_pass_fail[idx][line_id[j + count + suplit_data_sum_count]]= 0
                            model_pass_fail[idx][line_id[j + count + suplit_data_sum_count + 1]] = 0
                        elif abs(output_data[0][j] - correct_value[1]) < 1e-2:
                            model_pass_fail[idx][line_id[j + count + suplit_data_sum_count]]= 1
                            model_pass_fail[idx][line_id[j + count + suplit_data_sum_count + 1]] = 0
                        elif abs(output_data[0][j] - correct_value[2]) < 1e-2:
                            model_pass_fail[idx][line_id[j + count + suplit_data_sum_count]]= 0
                            model_pass_fail[idx][line_id[j + count + suplit_data_sum_count + 1]] = 1
                        else:
                            model_pass_fail[idx][line_id[j + count + suplit_data_sum_count]]= 1
                            model_pass_fail[idx][line_id[j + count + suplit_data_sum_count + 1]] = 1
                        count += 1
                        # temp[idx][temp_count+suplit_data_num*model_id] = output_data[0][j]  # 故障出力値を格納
                        # temp_count += 1
    
            # print(output_data)                # output_dataには、テストパターンnのときの、故障があるかどうかの情報が格納されている
        
        if model_id == single_line_model and single_flag == 1:
            suplit_data_sum_count += integration_num*output_node_num - 1  # 処理された正解データの総数を更新．統合されていない信号線がある場合，その分1少なくなるため-1する
        else:
            suplit_data_sum_count += integration_num*output_node_num  # 処理された正解データの総数を更新



    # print(model_pass_fail[0])
    # print(model_pass_fail[1])
    # print(model_pass_fail[2])
    # print(model_pass_fail[3])
    # print(a)

    # print("model_pass_fail:", model_pass_fail)  # モデルの出力を表示
    # print(fault_line_num)
    # print(len(model_pass_fail))
    # print("model_pass_fail:", len(model_pass_fail), len(model_pass_fail[0]))  # model_pass_failのサイズを表示
    # print("model_pass_fail[0]:", model_pass_fail[0])  # model_pass_failの最初の行を表示
    # with open('s344output2', 'w') as f:  # モデルの出力をファイルに保存
    #     for i in range(input_data_num):
    #         for j in range(fault_line_num - 1):
    #             # print(f"i{i}, j{j}")
    #             f.write(str(int(model_pass_fail[i][j])) + ',')  # 小数点を消すためにint()を使う
    #         f.write(str(int(model_pass_fail[i][j+1])) + '\n')
    
    # with open('s344output2_temp', 'w') as f:  # モデルの出力をファイルに保存
    #     f.write('\n\n')
    #     for i in range(input_data_num):
    #         for j in range(int(fault_line_num/2) + single_flag - 1):
    #             if temp[i][j] == 0:
    #                 temp[i][j] = 0
    #             elif abs(temp[i][j] - correct_value[1]) < 1e-2:  # floatの比較は、誤差を考慮してabsを使用
    #                 temp[i][j] = 1
    #             elif abs(temp[i][j] - correct_value[2]) < 1e-2:
    #                 temp[i][j] = 2
    #             else:
    #                 temp[i][j] = 3
    #             f.write(str(temp[i][j]) + ',')
    #         f.write(str(temp[i][j+1]) + '\n')

  # モデル出力を故障の種類（fault_type_sum個）に分ける
    # model_pass_failは、行はモデルの入力データ数、列は故障信号線の数を表す二次元リストであるため、パスフェイル情報（pass_failnファイル）と比較するために、行と列を入れ替える
    model_pass_fail = [[row[i] for row in model_pass_fail] for i in range(len(model_pass_fail[0]))]  #内包表記を使って行と列を入れ替える

    compare_model_pass_fail = [[0 for _ in range(tp_num)] for _ in range(signal_num*fault_type_sum)] # 各行が各故障が起きた場合の各テストパターンのパスフェイル情報を格納する二次元リスト

    for i in range(signal_num):
        for j in range(fault_type_sum):
            for k in range(tp_num):
                compare_model_pass_fail[i*fault_type_sum + j][k] = model_pass_fail[i][k*fault_type_sum + j] 

    # print("compare_model_pass_fail", compare_model_pass_fail)
    # with open('aaa', 'w') as f:
    #     for i in range(len(compare_model_pass_fail)):
    #         f.write(str(compare_model_pass_fail[i]) + '\n')

    
    st_compare_model_pass_fail = [[0 for _ in range(tp_num)] for _ in range(signal_num*2)] # 縮退故障用のパスフェイルリスト。各行が各故障が起きた場合の各テストパターンのパスフェイル情報を格納する二次元リスト
    br_compare_model_pass_fail = [[0 for _ in range(tp_num)] for _ in range(signal_num*(fault_type_sum - 2))] # ブリッジ故障用のパスフェイルリスト。各行が各故障が起きた場合の各テストパターンのパスフェイル情報を格納する二次元リスト

    st_idx = 0
    br_idx = 0
    for i in range(signal_num*fault_type_sum):
        if (i%fault_type_sum == 0) or (i%fault_type_sum == 1): # 縮退故障の場合
            for j in range(tp_num):
                st_compare_model_pass_fail[st_idx][j] = compare_model_pass_fail[i][j]
            st_idx += 1
        else: # ブリッジ故障の場合
            for j in range(tp_num):
                br_compare_model_pass_fail[br_idx][j] = compare_model_pass_fail[i][j]
            br_idx += 1
    
    # print("st_compare_model_pass_fail", st_compare_model_pass_fail)
    # print("br_compare_model_pass_fail", br_compare_model_pass_fail)

    # with open("aa", 'w') as f:
    #     for i in range(len(st_compare_model_pass_fail)):
    #         f.write(str(st_compare_model_pass_fail[i]) + '\n')

    # with open("a", 'w') as f:
    #     for i in range(len(br_compare_model_pass_fail)):
    #         f.write(str(br_compare_model_pass_fail[i]) + '\n')

    # ブリッジ故障においては、縮退故障と異なり、出力信号線以外にもブリッジ故障が生じない信号線があり、ファイルから読み取るしかない。
    # br_fault_all_lineから出力信号線以外で故障が発生しない信号線を特定する
    br_fault_all_line = sorted(br_fault_all_line)  # ブリッジ故障信号線番号を小さい順に並び替える

    with open("c" + cir, 'r') as f:
        line = f.readline()
        all_signal_num  = int(line.split()[0]) # 出力信号線も含めた全信号線の総数を取得
        cir_output_line_num = int(line.split()[1])  # 出力信号線の総数を取得
        cir_input_line_num = int(line.split()[2])  # 入力信号線の総数を取得
    
    # br_fault_all_lineの中に含まれていない信号線番号を取得
    br_missing_line = [line for line in range(1, (all_signal_num + 1)) if line not in br_fault_all_line]
    # print("br_missing_line：", br_missing_line)  # ブリッジ故障が発生しない信号線番号を表示

    # br_missing_lineには、出力信号線の番号も含まれているため、削除する
    for i in range(cir_output_line_num):
        br_missing_line.remove(i + cir_input_line_num + 1)  # 出力信号線の番号を削除

    br_missing_line = sorted(br_missing_line, reverse=True)  # ブリッジ故障が発生しない信号線番号を大きい順に並び替える.br_missing_lineを削除する際に、インデックスのずれをなくすために、逆順に並び替えて、インデックスの大きい物から削除する 
    
    # print("br_compare_model_pass_failの長さ：", len(br_compare_model_pass_fail))  # ブリッジ故障のパスフェイル情報の長さを表示
    # print("br_fault_all_line：", br_fault_all_line)  # ブリッジ故障が発生する信号線番号を表示
    print("br_missing_line：", br_missing_line)  # ブリッジ故障が発生しない信号線番号を表示
    print("br_missing_lineの長さ：", len(br_missing_line))  # ブリッジ故障が発生しない信号線番号の長さを表示

    # br_compare_model_pass_failから、ブリッジ故障が発生しない信号線に対応しているパスフェイル情報を削除する
    # （注意）br_compare_model_pass_failには、出力信号線は含まれていない。インデックスが小さいものから、入力信号線、入力信号線と出力信号線以外の信号線の順番になっている。
    for i in range(len(br_missing_line)):
        if br_missing_line[i] <= cir_input_line_num: # 上の注意から欠番の信号線が入力信号線であるときは、以下のようにする
            for j in range(10):  # ブリッジ故障の種類は0~9の10種類なので,10回繰り返す
                # with open("bb", 'a') as f:
                #     f.write(str(br_missing_line[i] - 1) + '\n')
                br_compare_model_pass_fail.pop((br_missing_line[i] - 1)*10)  # ブリッジ故障が発生しない信号線番号を削除
            # print(br_missing_line[i] - 1)
        else: # 上の注意から欠番の信号線が入力信号線以外の信号線であるときは、br_compare_model_pass_failには出力信号線は含まれていないので以下のようにする
            for j in range(10): 
                br_compare_model_pass_fail.pop((br_missing_line[i] - cir_output_line_num - 1)*10)
                # with open("bb", 'a') as f:
                #     f.write(str(br_missing_line[i] - cir_output_line_num - 1) + '\n')
            # print(br_missing_line[i] - cir_output_line_num - 1)
    
    # print("br_compare_model_pass_failの長さ：", len(br_compare_model_pass_fail))  # ブリッジ故障のパスフェイル情報の長さを表示


    # print("delite_count：",i)  # 削除した数を表示
    # with open("b", 'w') as f:
    #     for i in range(len(br_compare_model_pass_fail)):
    #         f.write(str(br_compare_model_pass_fail[i]) + '\n')

    # print(st_compare_model_pass_fail[1])
    # print(st_compare_model_pass_fail[2])
    # print(st_compare_model_pass_fail[3])
    # print(br_compare_model_pass_fail[1])
    # print(br_compare_model_pass_fail[2])
    # print(br_compare_model_pass_fail[3])

    # for i in range(len(st_pass_fail)):
    #     print(f"st_pass_fail[{i}]: {st_pass_fail[i]}")  # 縮退故障のパスフェイル情報を表示

    # with open('dd', 'w') as f:  # 縮退故障のパスフェイル情報をファイルに保存
    #     for i in range(len(st_compare_model_pass_fail)):
    #         for j in range(len(st_compare_model_pass_fail[i])):
    #             f.write(str(st_compare_model_pass_fail[i][j]))
    #         f.write('\n')

    

    # ランダムに選んだ故障のパスフェイル情報（pass_failリスト）とcompare_model_pass_failを比較して、縮退故障候補を取得する
    st_fault_candidate, st_fault_type = get_fault_candidate(st_pass_fail, st_compare_model_pass_fail, st_fault_all_line, signal_num, br_missing_line, 0)  # 縮退故障の候補を取得(縮退故障の時、br_missing_lineは使わない、引数に設定しているから一応おいている)
    br_fault_candidate, br_fault_type = get_fault_candidate(br_pass_fail, br_compare_model_pass_fail, br_fault_all_line, signal_num, br_missing_line, 1)  # ブリッジ故障の候補を取得

    
    # 故障候補の中に、実際に縮退故障として選んだ信号線と故障の種類が一致するものがあるかどうかを確認
    start_st = time.time()
    check_fault_candidate(st_fault_candidate, st_fault_type, st_fault_target_line, st_fault_target_type, 0)  # 縮退故障の候補を確認
    end1 = time.time()
    print(f"縮退故障診断時間：{end1 - start:.4f}秒")
    check_fault_candidate(br_fault_candidate, br_fault_type, br_fault_target_line, br_fault_target_type, 1) # ブリッジ故障の候補を確認
    end2 = time.time()
    print(f"ブリッジ故障診断時間：{end2 - (start_st - end1) - start:.4f}秒")
