# 任意の信号線におけるパスフェイル情報を確認するプログラム
# 縮退故障辞書、ブリッジ故障辞書から、各故障の出力値を取得して、正常な回路の出力値と比較してパスフェイル情報を取得し、同じパスフェイル情報を持つ故障候補の数を取得するプログラム
# cd workspace/research2/experiment
#　実行コマンド：　python3 check_pass_fail2.py

import numpy as np
import tensorflow as tf

import random

#グローバル変数
cir = 's1488'  # 対象回路
tp_file = cir + '.vec'  # テストパターンファイル名
part_stdic_file = cir + "stdic_bi/aout_" #縮退故障辞書ファイル名の一部
part_brdic_file = cir + "brdic_bi/aout_" #ブリッジ故障辞書ファイルの一部
st_diagnosis_dir = cir + "diagnosis_st_data2/" #縮退故障診断を行うためのデータを保存するフォルダ
br_diagnosis_dir = cir + "diagnosis_br_data2/" #ブリッジ故障診断を行うためのデータを保存するフォルダ
st_target_fault_num = None  # 縮退故障診断対象の信号線の総数
br_target_fault_num = None  # ブリッジ故障診断対象の信号線の総数
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


# 故障出力値を取得してファイルに保存する関数
def get_fault_output(tp_num, fault_target_line, fault_target_type, fault_flag): # fault_flagは0なら縮退故障、1ならブリッジ故障を表す
    # 縮退故障、ブリッジ故障の辞書ファイルのパスを指定
    if fault_flag == 0:  # 縮退故障の診断を行う場合
        part_dic_file = part_stdic_file  # 縮退故障辞書ファイルのパスを指定
        diagnosis_file = st_diagnosis_dir + 'fault_output'  # 縮退故障診断用の故障出力保存ファイル名
        target_fault_num = st_target_fault_num
        fault_type_num = 2  # 縮退故障の種類は0縮退故障と1縮退故障の2種類
    else:  # ブリッジ故障の診断を行う場合
        part_dic_file = part_brdic_file  # ブリッジ故障辞書ファイルのパスを指定
        diagnosis_file = br_diagnosis_dir + 'fault_output'  # ブリッジ故障診断用の故障出力保存ファイル名
        target_fault_num = br_target_fault_num
        fault_type_num = 10  # ブリッジ故障の種類は0~9の10種類

    # 診断対象故障の出力値をファイルに保存
    for i in range(tp_num):
        with open(part_dic_file + str(i), 'r') as f:    # 故障辞書ファイルを指定
            fault_inf = f.readline()  # 故障情報
            id_num = int(fault_inf.split()[2])  # 対象回路で起こりうる故障数
            # print(f"in_num：{id_num}")
            lines = f.readlines()  # 故障辞書ファイルの全行を読み込む

        # 各辞書におけるIDと出力値を取得
        fault_output = [0 for _ in range(id_num)]  # 故障出力値を格納するリスト
        idx = 0
        # print(f"len(lines)：{len(lines)}")
        for j in range(0, len(lines), 2):
            if fault_flag == 0:  # 縮退故障の診断を行う場合
                fault_output[idx] = lines[j+1].replace('\n', '')  # 故障出力値を取得
                idx += 1
            else:  # ブリッジ故障の診断を行う場合
                fault_output[idx] = lines[j+1].replace('\n', '')  # 故障出力値を取得
                idx += 1

        with open(diagnosis_file + str(i), 'w') as f:
            for j in range(len(fault_output)):
                f.write(str(fault_output[j]) + '\n')
    
    # print(f"fault_output：{fault_output[j]}")
    # print(j)


# パスフェイル情報を取得してファイルに保存する関数
def get_pass_fail(tp_num, fault_flag):  # fault_flagは0なら縮退故障、1ならブリッジ故障を表す
    if fault_flag == 0:  # 縮退故障の診断を行う場合
        fault_output_file = st_diagnosis_dir + 'fault_output'  # 縮退故障の故障出力を保存しているファイルのパスを指定
        pass_fail_file = st_diagnosis_dir + 'pass_fail'  # 縮退故障のパスフェイル保存ファイル名
        target_fault_num = st_target_fault_num  # 縮退故障診断対象の信号線の総数を取得
        fault_type_num = 2  # 縮退故障の種類は0縮退故障と1縮退故障の2種類
    else:  # ブリッジ故障の診断を行う場合
        fault_output_file = br_diagnosis_dir + 'fault_output'  # ブリッジ故障の故障出力を保存しているファイルのパスを指定
        pass_fail_file = br_diagnosis_dir + 'pass_fail'  # ブリッジ故障のパスフェイル保存ファイル名
        target_fault_num = br_target_fault_num  # ブリッジ故障診断対象の信号線の総数を取得
        fault_type_num = 10  # ブリッジ故障の種類は0~9の10種類


    #正常な回路出力を取得する
    correct_output_value = []  # 正常な回路出力を格納するリスト
    with open(correct_output_file, 'r') as f:
        lines = f.readlines()  # 正常な回路出力の全行を読み込む
        for i in range(1, len(lines), 2):
            correct_output_value.append(lines[i].replace('\n', ''))

    # print(correct_output_value)
    
    # 各故障の種類における回路出力と正常な回路出力を比較して、パスフェイル情報を取得
    fault_output_value = [[0 for _ in range(tp_num)] for _ in range(target_fault_num*fault_type_num)]  # 故障出力値を格納する二次元リスト fault_output_value[30個ランダムに選んだ故障][テストパターン数]。fault_output_value[2][4]には、ランダムに選んだ故障のうち2番目に選んだ故障が発生した回路にテストパターン4を入力したときの出力値が格納される
    print(f"len(target_fault_num*fault_type_num)：{len(fault_output_value)}")
    for i in range(tp_num):
        with open(fault_output_file + str(i), 'r') as f:
            # print(f"fault_output_file + str(i)：{fault_output_file + str(i)}")
            lines = f.readlines()  # fault_output~ファイルから故障出力値の全行を読み込む
            # print(f"len(lines)：{len(lines)}")
            for j in range(target_fault_num*fault_type_num):
                fault_output_value[j][i] = lines[j].replace('\n', '')
            
    
    # print(f"fault_output_value{fault_output_value}")
    # with open('de', 'w') as f:
    #     for i in range(len(fault_output_value)):
    #         f.write(str(fault_output_value[i]) + '\n')
    
    # 故障出力値と正常な回路出力を比較して、パスフェイル情報を取得
    pass_fail = [[0 for _ in range(tp_num)] for _ in range(target_fault_num*fault_type_num)]  # パスフェイル情報を格納するリスト。診断を行う際に使用する
    for i in range(target_fault_num*fault_type_num):  # 故障の種類ごとにパスフェイル情報を取得
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


#  パスフェイルが同じである故障候補を取得する関数
def get_same_passfail_sum(pass_fail, fault_all_line, fault_flag): # fault_flagは0なら縮退故障、1ならブリッジ故障を表す
    if fault_flag == 0:  # 縮退故障の診断を行う場合
        fault_output_file = st_diagnosis_dir + 'fault_output'  # 縮退故障の故障出力を保存しているファイルのパスを指定
        pass_fail_file = st_diagnosis_dir + 'pass_fail'  # 縮退故障のパスフェイル保存ファイル名
        target_fault_num = st_target_fault_num  # 縮退故障診断対象の信号線の総数を取得
        fault_type_num = 2  # 縮退故障の種類は0縮退故障と1縮退故障の2種類
        fault_name = '縮退故障'
    else:  # ブリッジ故障の診断を行う場合
        fault_output_file = br_diagnosis_dir + 'fault_output'  # ブリッジ故障の故障出力を保存しているファイルのパスを指定
        pass_fail_file = br_diagnosis_dir + 'pass_fail'  # ブリッジ故障のパスフェイル保存ファイル名
        target_fault_num = br_target_fault_num  # ブリッジ故障診断対象の信号線の総数を取得
        fault_type_num = 10  # ブリッジ故障の種類は0~9の10種類
        fault_name = 'ブリッジ故障'

    same_count = [0 for _ in range(target_fault_num*fault_type_num)]  # パスフェイルが同じである故障候補の数を格納するリスト
    zero_count = 0  # 全てのパスフェイルが0の場合のカウント
    count_line_idx = 0  # 信号線番号をカウントするためのインデックス
    fault_type_count = 0  # 故障の種類をカウントするためのインデックス
    all_zero_line = []  # 全てのパスフェイルが0の場合の信号線番号を格納するリスト
    for i in range(len(pass_fail)):
        if all( x == 0 for x in pass_fail[i]) == True:  #区別できない故障の場合（＝全てのパスフェイルが0の場合）は、
            zero_count += 1  # 全てのパスフェイルが0の場合のカウントを増やす
            all_zero_line.append(fault_all_line[count_line_idx])  # 全てのパスフェイルが0の場合の信号線番号を格納

        for j in range(i+1, len(pass_fail)):
            if pass_fail[i] == pass_fail[j]:
                same_count[i] += 1
                same_count[j] += 1
        
        fault_type_count += 1  # 故障の種類をカウントする
        if fault_type_count == fault_type_num:  # 故障の種類をカウントする
            fault_type_count = 0  # 故障の種類をリセット
            count_line_idx += 1  # 信号線番号をカウントする

    all_zero_line = list(set(all_zero_line))  # 全てのパスフェイルが0の場合の信号線番号を重複を除いて取得
    print(f"{fault_name}における区別できない故障がある信号線番号：{all_zero_line}")
    print(f"{fault_name}における区別できない信号線の数：{len(all_zero_line)}")
    print(f"{fault_name}における区別できない故障の数：{zero_count}")
    print(f"{fault_name}におけるパスフェイルが同じである故障候補の数：{same_count}")
    print(f"{fault_name}におけるパスフェイルが同じである故障候補で最大の数を持つ故障候補：{max(same_count)}")



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
        print("st_fault_all_line：", st_fault_all_line)
        # print(suplit_num)
    
    st_target_fault_num = len(st_fault_all_line)  # 縮退故障診断対象の信号線の総数を取得

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
    
    br_target_fault_num = len(br_fault_all_line)  # ブリッジ故障診断対象の信号線の総数を取得

    print("len(br_fault_all_line)：", len(br_fault_all_line))
    print("br_fault_all_line：", br_fault_all_line)
    

    # 診断対象となる縮退故障、ブリッジ故障それぞれに対する対象信号線番号を30個ランダムにそれぞれ取得
    st_fault_target_type = [i%2 for i in range(len(st_fault_all_line)*2)]  # 縮退故障が発生する信号線で発生するのは0縮退故障か1縮退故障かをランダムに決定する
    br_fault_target_type = [i%10 for i in range(len(br_fault_all_line)*10)]  # ブリッジ故障が発生する信号線で発生するのは0~9のブリッジ故障かをランダムに決定する

    # 診断対象故障の出力値をファイルに保存
    get_fault_output(tp_num, st_fault_all_line, st_fault_target_type, 0) # 縮退故障時の故障出力値をファイルに保存
    get_fault_output(tp_num, br_fault_all_line, br_fault_target_type, 1) # ブリッジ故障時の故障出力値をファイルに保存


    # パスフェイル情報を取得してファイルに保存
    st_pass_fail = get_pass_fail(tp_num, 0)  # 縮退故障のパスフェイル情報を取得してファイルに保存
    print(f"get_pass_fail(tp_num, 0)の実行完了")
    br_pass_fail = get_pass_fail(tp_num, 1)  # ブリッジ故障のパスフェイル情報を取得してファイルに保存
    print(f"get_pass_fail(tp_num, 1)の実行完了")

    get_same_passfail_sum(st_pass_fail, st_fault_all_line, 0)
    get_same_passfail_sum(br_pass_fail, br_fault_all_line, 1)
    
