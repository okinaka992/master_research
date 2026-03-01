# 16進数の故障辞書を、2進数に変換するプログラム
# python3 change_binari_b2.py


#グローバル変数
cir = 's1488'  # 対象回路
outval_file_sum = 5  # 変換対象の故障辞書ファイルの数

outval_file = cir + 'brdic/' + cir + '.outval_'  # 変換対象の故障辞書ファイル
binary_outval_file =  cir + 'brdic_bi/aout_'  # 2進数に変換後の故障辞書ファイル


def main():

  with open('c' + cir, 'r') as f:
    output_bit_sum = int(f.readline().split()[1])  # 対象回路の出力値が何ビットかを取得

  for  outval_file_num in range(outval_file_sum):

    with open(outval_file + str(outval_file_num), 'r') as f:
        line = f.readline()
        start = int(line.split()[1])  # 何パターン目から何パターン目に対応したファイルか
        end = int(line.split()[3])
        target_pattern_range = end - start + 1  # 変換対象のファイルにおける総パターン数を取得
                                         # ※32パターンずつファイルは分割されているが、対象回路のテストパターンが32で割れるとは限らないため、最後のファイルには、32パターン未満のパターンが含まれることもあるため、ここで設定しておく

        print("並列に入れますよ～")
        lines = f.readlines()  # 故障辞書ファイルの内容を読み込む
        print("並列に入りましたよ～")
        
        numflt = len([line.split()[1] for line in lines[::2]])  # 全故障の総数を取得＝idの数
        fault_id_lines = lines[::2]  # 故障IDが書かれた行を取得
        pre_fault_val_temp = lines[1::2] # 故障値のリストを取得

    pre_fault_val = []
    for i, val in enumerate(pre_fault_val_temp):
        row = val.strip().split()
        row = [s[::-1] for s in row]  # 各要素の文字列を逆順にする=各要素の値は、0ビット目が0パターン目、1ビット目が1パターン目、・・・、31ビット目が31パターン目に対応しているため、これを逆順にすることで、インデックス0が0パターン目、インデックス1が1パターン目、・・・、インデックス31が31パターン目に対応するようにする
        pre_fault_val.append(row)

    # pre_fault_valの各値を2進数に変換
    bin_fault_val = [['' for _ in row] for row in pre_fault_val]  # 2進数に変換した値を格納するリストを初期化
    for i, row in enumerate(pre_fault_val):
      for j, content in enumerate(row):
        bin_row = [format(int(val, 16), '04b')[::-1] if val != '0' else '0'*4 for val in content]  # 各値を4ビットの2進数に変換
        bin_fault_val[i][j] = "".join(bin_row)  # 各行の値を結合して1つの文字列にする
        bin_fault_val[i][j] = bin_fault_val[i][j].ljust(target_pattern_range, '0')  # nビット目から32ビット目までが0の場合、0は省略されているので、nビット目以降の文字を0で埋める＝後ろに0を追加する

    print(bin_fault_val[0])  # 2進数に変換した値を表示

    per_pt_bin_fault_val = [['' for _ in range(numflt)] for _ in range(target_pattern_range)]  # bin_fault_valをパターンごとに分割するためのリスト
    # bin_fault_valは、行がID、列が回路出力線に対応しており、各要素がパターンごとの故障値を表す。変換後は、パターンごとにファイルを分割して、各出力線ごとの値を記述するため、
    for i in range(target_pattern_range):
       for j in range(numflt):
           for k in range(output_bit_sum):
               # 変換後の故障値をパターンごとに分割
               per_pt_bin_fault_val[i][j] += bin_fault_val[j][k][i]
    
    # print(per_pt_bin_fault_val[0])  # 2進数に変換した値を表示

    st = start
    for i in range(target_pattern_range):
       with open(binary_outval_file + str(st), 'w') as f:
          f.write('pattern ' + str(st) + ' ' + str(numflt) + '\n')  # パターン番号を書き込む
          st += 1
          for j in range(numflt):
              # 変換後の故障値をファイルに書き込む
              f.write(fault_id_lines[j])
              f.write(per_pt_bin_fault_val[i][j] + '\n')

    
  print("プログラムは終了しました")

  # print(bin_fault_val[0])  # 2進数に変換した値を表示


  # # x = [format(int(val, 16), '032b') for val in pre_fault_val[0]]
  # print(format(int(pre_fault_val[0][1], 16), '032b'))  # 2進数に変換した値を表示


if __name__ == "__main__":
    main()