# 学習用データを作成するプログラム
# make_learn_data.ipynbを改良
#　故障辞書における出力と正常化路出力における出力の順序をchange_binari_s2.c、change_binari_b2.cで修正し、outvalファイルにおける出力の順番と正常回路出力の順番を一致させている
#  make_learn_data.ipynbでは逆順に対応していたが、その必要がなくなったためその点を修正。さらに、正常回路出力をsigvalファイルではなく、correct_outputディレクトリから読み込むように変更
#　山内さんの「make_learn_data.c」（学習用データを作成するプログラム）を参考に作成

# python3 make_learn_data2.py


#グローバル変数
cir = "s38584" #対象回路（cs回路もsとして記述　例 cs344の場合、s344と記述）
test_ptn_file = cir + ".vec" #正解データを生成するのに使うテストパターンファイル
signal_value_file = cir + "_sigval"
part_stdic_file = cir + "stdic_bi/aout_" #縮退故障辞書ファイル名の一部
part_brdic_file = cir + "brdic_bi/aout_" #ブリッジ故障辞書ファイルの一部
input_data_file = cir + "input" #作成する入力データを保存するファイル
correct_data_file = cir + "output" #作成する正解データを保存するファイル
correct_output_file = "correct_output/" + cir + "_correct_output" #正常回路出力を保存しているファイル

# print("%s %s %s %s %s %s %s" % (cir, test_ptn_file, signal_value_file, part_stdic_file, part_brdic_file, input_data_file, correct_data_file))


#ファイルが存在するか確認、なかったらコマンドラインに入力させる処理を実行
def check_file_existence(file_name):
    try:
        with open(file_name):
            pass
    except FileNotFoundError:
        print(file_name + "が見つかりません")
        file_name = input("ファイル名を入力してください:")
        check_file_existence(file_name)
    return file_name


def main():

    global test_ptn_file, signal_value_file, stdic_file, brdic_file

    #ファイルの存在確認
    # test_ptn_file = check_file_existence(test_ptn_file)
    # signal_value_file = check_file_existence(signal_value_file)

    #テストパターンファイルの読み込み
    with open(test_ptn_file, "r") as f:
        tpn_inf = f.readline()  # テストパターン情報
        tpn_sum = int(tpn_inf.split()[0])  # テストパターン数
        # print(tpn_inf)
        tpn_bit_sum = int(tpn_inf.split()[1])  # テストパターンのビット数
        print(f"テストパターン数：{tpn_sum}")
        print(f"テストパターンのビット数；{tpn_bit_sum}")

        tpn = [_.replace("\n", "") for _ in f.readlines()]  # テストパターンファイル全体を読み込む

    #信号線の値ファイルの読み込み
    with open(signal_value_file, "r") as f:
        signal_value_inf = f.readline()
        signal_line_sum = int(signal_value_inf.split()[0]) # 信号線数
        output_bit = int(signal_value_inf.split()[1]) # 出力ビット数
        input_bit = int(signal_value_inf.split()[2]) # 入力ビット数
        print(signal_value_inf, signal_line_sum, output_bit, input_bit)
        signal_value_temp = [_.replace("\n", "") for _ in f.readlines()]
        signal_value_temp = signal_value_temp[1:] #2行目の情報はテストパターン数でtest_ptn_fileの情報と同一であるため削除
        signal_value = signal_value_temp[1::2]  # 2行おきにテストパターンの信号値を取得=偶数インデックスの行を取得。signal_valueの各要素は、各信号線の信号値を表す文字列


    #正常な回路出力値を読み込む
    correct_output_value = []  # 正常な回路出力を格納するリスト
    with open(correct_output_file, 'r') as f:
        lines = f.readlines()  # 正常な回路出力の全行を読み込む
        for i in range(1, len(lines), 2):
            correct_output_value.append(lines[i].replace('\n', ''))

    print(f"正常な回路出力値：{correct_output_value}")


    #ファイルの読み込み確認
    print(f"テストパターン数；{tpn_sum}")
    print(tpn)
    print(signal_value)


    # 入力データを作成
    # 故障の種類を表す2進数＝計12個
    fault_type_sum = 12
    fault_type_bit_num = 4 # 故障の種類を表す2進数のビット数
    fault_type=['0000',  # 0縮退故障
            '0001',  # 1縮退故障
            '0010',  # ブリッジ故障1
            '0011',  # ブリッジ故障2
            '0100',
            '0101',
            '0110',
            '0111',
            '1000',
            '1001',
            '1010',
            '1011']  # ブリッジ故障10

    from math import log2, ceil
    tpn_bit_num = ceil(log2(tpn_sum))  # テストパターンの通し番号を表現するために必要なビット数を計算．まず，テストパターン数を2を底とする対数を取り，小数点以下を切り上げる（ceil）
    print(tpn_bit_num)
    print(fault_type[0])

  # 入力データを作成
    input_data = []  # 入力データを格納するリスト

    for i in range(tpn_sum): # 入力データは，テストパターン数×故障の種類数分ある
        for j in range(fault_type_sum):
          input_data.append('{:0>{}b}'.format(i, tpn_bit_num) + fault_type[j%fault_type_sum]) # 2進数化した通し番号に故障の種類を結合してリストに追加 
                                                                                            # '{:0>{}b}'.format(i, tpn_bit_num)は，iをtpn_bit_numビットの2進数に変換する．'{:0>ビット数b}'.format(値)のように記述します。ここで、ビット数は2進数に変換したいビット数を指定し、値は2進数に変換したい整数値です。0>は結果の文字列を指定された長さになるまで左側を0で埋めることを意味します。

    print(input_data)

    with open(input_data_file, "w") as f:  # 入力データをファイルに書き込む
        for i in range(tpn_sum*fault_type_sum): # 入力データは，テストパターン数×故障の種類数分ある
            for j in range(tpn_bit_num + fault_type_bit_num - 1): # 「,」で区切らないといけないので一文字ずつ入力データを書き込む．書き込むビット数は，通し番号のビット数＋故障の種類のビット数．最後の文字の後は，「,」は必要ないので，for文では，tpn_bit_num + fault_type_bit_num - 1までとしている
                f.write(input_data[i][j] + ",")
            f.write(input_data[i][j+1] + "\n") # 最後の文字の後は，「,」は必要ないので，j+1番目の文字を書き込んで改行する

  # 正解データを作成
    print(signal_line_sum)
    correct_data = [[0] * (signal_line_sum - output_bit) for j in range(tpn_sum * fault_type_sum)]  # 正解データを格納する2重リストcorrect_data[tpn_sum * fault_type_sum][signal_line_sum - output_bit]を作成
    # ※故障辞書には、出力信号線の故障は想定されていない⇒正解データは、出力信号線以外の信号線を格納する
    print(f"signal_line_sum - output_bit:{signal_line_sum - output_bit}")
    print("正解データの行数：",len(correct_data))  # 作成した正解データの行数を表示
    print("正解データの列数：", len(correct_data[0])) # 作成した正解データの列数を表示
    print("正解データ0：", correct_data[0]) # 作成した正解データを表示

    #故障辞書ファイルの読み込み,correct_dataに正解データを格納
    for i in range(tpn_sum):  #故障辞書ファイルはそれぞれテストパターン数分作成あるので、テストパターン数分ループ
        stdic_file = part_stdic_file + str(i)  # 縮退故障辞書ファイル名
        brdic_file = part_brdic_file + str(i) # ブリッジ故障辞書ファイル名

        with open(stdic_file, "r") as f:
            stdic_inf = f.readline()
            stdic_file_num = int(stdic_inf.split()[1])  # 縮退故障辞書のファイル番号
            st_id_sum = int(stdic_inf.split()[2]) # idの数(※id番号は0から始まるため最後のid番号は「id_sum-1」)＝各テストパターンににおいて想定される縮退故障数
            # print("sasasas",stdic_inf, stdic_file_num, st_id_sum)
            st_dic_temp = [_.replace("\n", "") for _ in f.readlines()]
            # print("dsadsa", st_dic_temp)
            # print(stdic_inf)
            st_inf = st_dic_temp[::2] # 各故障の情報を格納
            st_id = [_.split()[1] for _ in st_inf]   # 縮退故障のid, インデックスとして使う
            st_signal_num = [_.split()[3] for _ in st_inf]  # 縮退故障が生じている信号線数番号
            st_type = [int(_.split()[5]) for _ in st_inf] # 縮退故障のタイプ＝0縮退故障か1縮退故障か
            # print(st_inf)
            # print("st_id", st_id)
            print("st_id_sum；", st_id_sum)
            print("st_signal_num:", st_signal_num)
            print("st_type", st_type)
            st_outsignal_value = st_dic_temp[1::2] # 各故障の出力信号線の信号値を格納
            # print("sasasa",st_signal_value)
            # if i == 0:
            #     print("st_outsignal_value", st_outsignal_value[1])

        with open(brdic_file, "r") as f:
            brdic_inf = f.readline()
            brdic_file_num = int(brdic_inf.split()[1])    # ブリッジ故障辞書のファイル番号
            br_id_sum = int(brdic_inf.split()[2])         # idの数(※id番号は0から始まるため最後のid番号は「id_sum-1」)＝各テストパターンににおいて想定されるブリッジ故障数
            # print("sasasas",brdic_inf, brdic_file_num, id_sum)
            br_dic_temp = [_.replace("\n", "") for _ in f.readlines()]
            # print("dsadsa", br_dic_temp)
            # print(brdic_inf)
            br_inf = br_dic_temp[::2]                     # 各故障の情報を格納
            br_id = [_.split()[1] for _ in br_inf]        # ブリッジ故障のid, インデックスとして使う
            br_dominated_signal_num = [_.split()[3] for _ in br_inf]  # ブリッジ故障が生じている信号線番号＝ブリッジ故障の支配される信号線番号
            br_dominated_value = [_.split()[4] for _ in br_inf]  # ブリッジ故障で支配される信号線の信号値
            br_dominate_signal_num = [_.split()[5] for _ in br_inf]  # ブリッジ故障で支配する信号線番号
            # print(br_inf)
            # print("br_id", br_id)
            # print("br_signal_num", br_signal_num)
            br_outsignal_value = br_dic_temp[1::2] # 各故障の出力信号線の信号値を格納
            print("br_outsignal_value：",br_outsignal_value) 

        #   縮退故障辞書の情報をもとに、正解データを作成
        for j in range(st_id_sum):
            if st_outsignal_value[j] != correct_output_value[i]: # 故障出力値と正常出力値が異なるかどうかを確認
                if int(st_signal_num[j]) <= input_bit:  
                    print("故障候補", j, i+st_type[j],int(st_signal_num[j])-1)
                    correct_data[i*fault_type_sum+st_type[j]][int(st_signal_num[j])-1] = 1  # 縮退故障が生じている信号線に対応する正解データを1にする
                    #正解データは1行目は、テストパターンiの0縮退故障、テストパターンiの1縮退故障,テストパターンiのブリッジ故障1,・・・（12行目まで続く）、13行目はテストパターンi+1の0縮退故障、テストパターンi+1の1縮退故障,テストパターンi+1のブリッジ故障1,・・・（12行目まで続く）となる
                    #（上のコメントの続き）そのため、正解データのインデックスを、i*fault_type_sum+st_type[j]としている
                else: # 出力信号線の故障は想定されていないため、出力信号線の故障は無視.
                    correct_data[i*fault_type_sum+st_type[j]][int(st_signal_num[j])-output_bit-1] = 1  # 出力信号線より大きい信号線番号の信号線（入力信号線以外）の場合、出力信号線の数だけ引いた値をインデックスとする
            # if i == 0:
            #     print("correct_data", correct_data[0][0])
       
        #   ブリッジ故障辞書の情報をもとに、正解データを作成
        for j in range(br_id_sum):
            if j == 0:
                idx = 2   # 10種類のブリッジ故障を区別するために必要
            elif (j != 0) and (br_dominated_signal_num[j] != br_dominated_signal_num[j-1]):   # 正解データにおけるブリッジ故障のインデックスを操作するために必要。故障対象である支配される信号線番号が異なる場合、idxを2に初期化。ブリッジ故障は、10種類あるので、それを区別するために必要
                idx = 2      #  1つのテストパターンにつき、縮退故障数は0と1縮退故障の2つでその後に、ブリッジ故障を記載するのでidx=2としている

            if br_outsignal_value[j] != correct_output_value[i]: # 故障出力値と正常出力値が異なるかどうかを確認
                if int(br_dominated_signal_num[j]) <= input_bit:
                    correct_data[i*fault_type_sum+idx%12][int(br_dominated_signal_num[j])-1] = 1  # 1つのテストパターンにつき、故障の種類は、12種類なので、idx%12で正解データのインデックスを操作
                    if i == 0:
                        print("i*fault_type_sum+idx%12", i*fault_type_sum+idx%12)
                else:
                    correct_data[i*fault_type_sum+idx%12][int(br_dominated_signal_num[j])-output_bit-1] = 1
            
            idx += 1
    
    # if correct_output_value[0] != st_outsignal_value[0]:
    #     print("true")
    # print(correct_output_value[0], st_outsignal_value[0])
    with open(correct_data_file, "w") as f:
        for i in range(tpn_sum*fault_type_sum):
            for j in range(signal_line_sum - output_bit - 1):
                f.write(str(correct_data[i][j]) + ",")
            f.write(str(correct_data[i][j+1]) + "\n")

if __name__ == "__main__":
    main()