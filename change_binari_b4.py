# change_binari_b2.pyを編集したもの
# change_binari_b2.pyではs38584を変換するとメモリ不足になっていたため、
# 16進数の故障辞書を、2進数に変換するプログラム（ストリーミング版、メモリ消費を抑える）
# python3 change_binari_b4.py

import os

# グローバル変数
cir = 's38584'  # 対象回路
outval_file_sum = 7  # 変換対象の故障辞書ファイルの数

outval_file = cir + 'brdic/' + cir + '.outval_'  # 変換対象の故障辞書ファイルプレフィックス
binary_outval_file = cir + 'brdic_bi/aout_'  # 2進数に変換後の故障辞書ファイルプレフィックス

def hex_token_to_bin_reversed(token):
    """
    token: 文字列（例: '1a0f'）を各16進文字ごとに4ビットの2進数に変換し、
    各ニブルのビット列を逆順にして連結した文字列を返す。
    """
    bits = []
    for ch in token:
        # int(ch, 16) は '0'..'f' を扱える
        try:
            b = format(int(ch, 16), '04b')[::-1]
        except ValueError:
            # 想定外の文字が来た場合はゼロ詰めで補う
            b = '0000'
        bits.append(b)
    return "".join(bits)


def count_faults_in_file(path):
    """
    ヘッダ行を既に読み飛ばした後のファイルポインタ位置を想定せずに
    ファイルを開いて故障ペア行数をカウントして故障数（id行の数）を返す。
    """
    cnt = 0
    with open(path, 'r') as f:
        # ヘッダ行を読み飛ばす
        header = f.readline()
        for _ in f:
            cnt += 1
    return cnt // 2


def main():
    # 出力ビット数を c<cir> ファイルから取得
    cfile = 'c' + cir
    with open(cfile, 'r') as f:
        # 例: "output 32 ..." のようなフォーマットを想定
        output_bit_sum = int(f.readline().split()[1])

    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(binary_outval_file), exist_ok=True)

    for outval_file_num in range(outval_file_sum):
        src_path = outval_file + str(outval_file_num)
        if not os.path.isfile(src_path):
            print(f"警告: 入力ファイルが見つかりません: {src_path} - スキップします")
            continue

        # まずヘッダを読み取り、pattern 範囲を取得
        with open(src_path, 'r') as f:
            header = f.readline()
            parts = header.split()
            if len(parts) < 4:
                print(f"警告: ヘッダの形式が予想と異なります: {src_path}")
                continue
            try:
                start = int(parts[1])
                end = int(parts[3])
            except ValueError:
                print(f"警告: ヘッダの数値がパースできません: {header.strip()}")
                continue
            target_pattern_range = end - start + 1

        # 故障総数（id行の数）をカウント（ヘッダを除く）
        numflt = count_faults_in_file(src_path)
        if numflt == 0:
            print(f"情報: 故障が無いファイル: {src_path} - スキップ")
            continue

        # 出力ファイル群を開いてヘッダを書き込む
        out_files = []
        st = start
        for i in range(target_pattern_range):
            out_path = binary_outval_file + str(st + i)
            # 出力先ディレクトリは既に作ってあるはずだが念のため
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            of = open(out_path, 'w')
            of.write(f'pattern {st + i} {numflt}\n')
            out_files.append(of)
        print(f"{src_path} -> パターン {start}..{end} を {len(out_files)} ファイルに出力（故障数={numflt}）")

        # ファイルを再度開き、ヘッダを読み飛ばしてから 2 行ずつ処理
        with open(src_path, 'r') as f:
            f.readline()  # ヘッダ読み飛ばし

            fault_index = 0
            while True:
                id_line = f.readline()
                if not id_line:
                    break
                val_line = f.readline()
                if not val_line:
                    # 奇数行で終了しているなら無視して終了
                    break

                # val_line をトークンに分割（各トークンが出力ビットに対応するはず）
                tokens = val_line.strip().split()

                # トークンごとに2進文字列を作る（逆順ニブル、連結、パディング）
                bin_strs = []
                for token in tokens:
                    # b2 の挙動を再現：トークン文字列自体を逆順にしてから変換
                    b = hex_token_to_bin_reversed(token[::-1])
                    # target_pattern_range に満たない場合は末尾を '0' で埋める
                    if len(b) < target_pattern_range:
                        b = b.ljust(target_pattern_range, '0')
                    elif len(b) > target_pattern_range:
                        b = b[:target_pattern_range]
                    bin_strs.append(b)

                # トークン数が output_bit_sum 未満なら不足分をゼロで埋める
                if len(bin_strs) < output_bit_sum:
                    for _ in range(output_bit_sum - len(bin_strs)):
                        bin_strs.append('0' * target_pattern_range)
                # 逆に長過ぎる場合は切る
                if len(bin_strs) > output_bit_sum:
                    bin_strs = bin_strs[:output_bit_sum]

                # 各パターンごとに出力ビット列を作成（出力ビットを順に連結）
                per_pattern_bits = ['' for _ in range(target_pattern_range)]
                for k in range(output_bit_sum):
                    bstr = bin_strs[k]
                    # 位置ごとにビットを追加
                    for p in range(target_pattern_range):
                        per_pattern_bits[p] += bstr[p]

                # 各出力ファイルに故障ID行と対応ビット列を書き込む
                for p, of in enumerate(out_files):
                    of.write(id_line)
                    of.write(per_pattern_bits[p] + '\n')

                fault_index += 1

        # 全て閉じる
        for of in out_files:
            of.close()

    print("プログラムは終了しました")


if __name__ == "__main__":
    main()
