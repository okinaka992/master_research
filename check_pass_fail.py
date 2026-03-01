# 任意の信号線におけるパスフェイル情報を確認するプログラム
# cd workspace/research2/experiment
#　実行コマンド：　python3 check_pass_fail.py

import numpy as np
import tensorflow as tf

import random

#グローバル変数
cir = 's1488'  # 対象回路
tp_file = cir + '.vec'  # テストパターンファイル名
part_stdic_file = cir + "stdic_bi/aout_" #縮退故障辞書ファイル名の一部
part_brdic_file = cir + "brdic_bi/aout_" #ブリッジ故障辞書ファイルの一部
st_diagnosis_dir = cir + "diagnosis_st_data/" #縮退故障診断を行うためのデータを保存するフォルダ
br_diagnosis_dir = cir + "diagnosis_br_data/" #ブリッジ故障診断を行うためのデータを保存するフォルダ
target_fault_num = 30  # 故障診断対象の故障の数⇒全ての故障の中からtarget_fault_num個をランダムに選択する
correct_output_file = 'correct_output/' + cir + '_correct_output'  # 正常な回路の出力ファイル
input_data_file = cir + 'input'  # 入力データファイル
correct_data_file = cir + '分割正解データ' + '/' + cir + 'integrated_output'  # 統合後の正解データファイル
suplit_num_file = cir + '分割正解データ' + '/' + cir + 'suplit_num'  # モデルの分割数が保存されたファイル
suplit_data_num_file = cir + '分割正解データ' + '/' + cir + 'suplit_data_num'  # データの分割数が保存されたファイル
single_line_file = cir + '分割正解データ' + '/' + cir + 'single_line'  # 統合されていない信号線があるかが保存されたファイル
input_data_num = None  # 1個のモデルにおける学習データ数
input_node_num = None  #入力層におけるノード数 初期値は8　学習結果によって変更
output_node_num = None #　出力層のノード数＝分割数による
num_models = None  # 学習させるモデルの数
model_folder = cir + 'sepmodel'  # 学習済みモデルを保存するフォルダ
fault_line_num = None  # 故障診断対象の回路における故障信号線の総数
fault_type_sum = 12  # 故障の種類の総数
# correct_value = [0.00, 0.15, 0.75, 1.00]  # 正解データの値・種類
correct_value = [0, 1, 2, 3]
# threshold = [0.02, 0.5, 0.98]  # ANNの出力値を変換するための閾値
threshold = [0.5, 1.5, 2.5]


# 故障出力値を取得してファイルに保存する関数
def get_fault_output(tp_num, fault_target_line, fault_target_type, fault_flag): # fault_flagは0なら縮退故障、1ならブリッジ故障を表す
    # 縮退故障、ブリッジ故障の辞書ファイルのパスを指定
    if fault_flag == 0:  # 縮退故障の診断を行う場合
        part_dic_file = part_stdic_file  # 縮退故障辞書ファイルのパスを指定
        diagnosis_file = st_diagnosis_dir + 'fault_output'  # 縮退故障診断用の故障出力保存ファイル名
    else:  # ブリッジ故障の診断を行う場合
        part_dic_file = part_brdic_file  # ブリッジ故障辞書ファイルのパスを指定
        diagnosis_file = br_diagnosis_dir + 'fault_output'  # ブリッジ故障診断用の故障出力保存ファイル名

    # 診断対象故障の出力値をファイルに保存
    for i in range(tp_num):
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
                if fault_target_line[idx] == int(lines[j].split()[3]) and fault_target_type[idx] == (br_type_count%10):  # ファイルの「id 0 Br_flt 1 1 2」の行から、信号線番号を取得して、br_type_countの値から、10種類あるブリッジ故障の種類を計算して、比較
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



if __name__ == '__main__':
    
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
    st_fault_target_line = [12, 62, 75, 90, 94, 111, 120, 174, 176, 262, 271, 288, 306, 323, 346, 347, 351, 417, 424, 457, 483, 601, 608, 617, 651, 657, 702, 706, 772, 782] # 縮退故障が発生する信号線をtarget_fault_num個ランダムに選択。さらに、その信号線で発生するのは0縮退故障か1縮退故障かをランダムに決定する
    st_fault_target_type = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]  # 縮退故障が発生する信号線で発生するのは0縮退故障か1縮退故障かをランダムに決定する
    br_fault_target_line = [12, 62, 75, 90, 94, 111, 120, 174, 176, 262, 271, 288, 306, 323, 346, 347, 351, 417, 424, 457, 483, 601, 608, 617, 651, 657, 702, 706, 772, 782] # ブリッジ故障が発生する信号線をtarget_fault_num個ランダムに選択。さらに、その信号線で発生するのは0縮退故障か1縮退故障かをランダムに決定する
    br_fault_target_type = [random.randint(0, 9) for _ in range(target_fault_num)]  # ブリッジ故障が発生する信号線で発生するのは0~9のブリッジ故障かをランダムに決定する

    # 診断対象故障の出力値をファイルに保存
    get_fault_output(tp_num, st_fault_target_line, st_fault_target_type, 0) # 縮退故障時の故障出力値をファイルに保存
    get_fault_output(tp_num, br_fault_target_line, br_fault_target_type, 1) # ブリッジ故障時の故障出力値をファイルに保存


    # パスフェイル情報を取得してファイルに保存
    st_pass_fail = get_pass_fail(tp_num, 0)  # 縮退故障のパスフェイル情報を取得してファイルに保存
    br_pass_fail = get_pass_fail(tp_num, 1)  # ブリッジ故障のパスフェイル情報を取得してファイルに保存
    
    for i in range(target_fault_num):
        print("縮退故障のパスフェイル情報：", st_pass_fail[i])