# larn_data_delite3.ipynbを編集したプログラム．
# モデルのメモリサイズを計算するためだけに使用

# 正解データ数によってモデルの中間層のノード数を個別に設定するようにした
# learn_data_delite.ipynbでは全ての値が0の行を削除していたが，このプログラムでは正解データが同じものを1つに統合している．
# また，learn_data_suplit.pyと同様に正解データの統合とともに，入力データも削除している．
# ※learn_data_delite.ipynbでは全てが0の値を完全に削除しているが，このプログラムでは統合なので，値が全て0のデータも1つ残っている．
# cd workspace/research2/experiment


import shutil

def main(cir):
    #変数
    suplit_input_data_file = cir + '分割入力データ_test/' + cir + 'input'  # 分割された入力データファイル
    correct_data_folder = cir + '分割正解データ_test/' # 正解データを分割して保存するフォルダ
    correct_data_delite_folder = cir + '分割正解データ削除後_test/'  # 正解データを削除した後のデータを保存するフォルダ
    input_data_file = cir + 'input'  # 入力データファイル
    suplit_num = None # 何個ずつ分割するか = 1つのモデルの正解データ数
    suplit_num_differece_flag = False # 分割数が異なる場合にTrue
    suplit_data_num = None # 分割された正解データが何個あるか = モデルの分割数
    correct_data_file = correct_data_folder + cir + 'integrated_output'  # 分割された正解データファイル
    correct_data_integrated_inf_file = correct_data_delite_folder + cir + 'integrated_inf'  # 正解データの統合情報を保存するファイル
    correct_data_delite_file = correct_data_delite_folder + cir + 'integrated_output'  # 正解データを統合・削除した後の正解データファイル
    suplit_num_file = correct_data_folder + cir + 'suplit_num'  # 分割数を保存するファイル
    delite_sum_file = correct_data_delite_folder + cir + 'delite_inf'  # 削除行数を保存するファイル
    save_delite_inf_file = correct_data_delite_folder + cir + 'delite_inf_all'  # これまでの削除行数を保存するファイル
    save_delite_inf_file_sub = correct_data_delite_folder + cir + 'delite_inf_all2'  # 分割された正解データごとの情報は保存しないファイル
    delited_data_num_file = correct_data_delite_folder + cir + 'delited_data_num'  # 削除後のデータ数を保存するファイル
    delited_data_num_all_file = correct_data_delite_folder + cir + 'delited_data_num_all'  # これまでの削除後のデータ数を保存するファイル
    delited_data_num_all_file_sub = correct_data_delite_folder + cir + 'delited_data_num_all2'  # これまでの削除後のデータ数を保存するファイル（分割された正解データごとの情報は保存しないファイル）
    single_line_inf_file = correct_data_folder + "/" + cir + 'single_line_inf'  # 統合されていない信号線の情報を保存するファイル
    
    middle_layer_node_num_file = correct_data_delite_folder + cir + 'middle_layer_node_num'  # 中間層のノード数を保存するファイル

    shutil.copy(single_line_inf_file, correct_data_delite_folder)  # 統合されていない信号線の情報をコピーする
    
    
    with open(suplit_num_file, 'r') as f:  # データの分割数を保存するファイルを開く
        suplit_data_num = int(f.readline().replace('\n', ''))  # データの分割数を読み込む
        print("正解データの分割数:", suplit_data_num)
    
    
    # 入力データファイルを開いてデータを読み込む
    with open(input_data_file) as f:
        input_data = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]  # 各行の空白文字と改行文字を削除
    
    
    # 各分割された正解データファイルを開いてデータを読み込み、同じ正解データの行を削除し、それに伴って入力データも削除する
    # その後、削除した正解データと入力データを新しいファイルに保存する
    # 削除した行数の平均も計算して表示
    all_delite_sum = 0
    max_delite_sum = 0 #　正解データごとの削除行数の最大値
    min_delite_sum = -1 #　正解データごとの削除行数の最小値(初期値が-1なのは，最初の削除行数が0のときに正しく動作しないため)
    delite_sum_list = []
    delited_data_num = [] # 削除後の正解データ数
    for i in range(suplit_data_num):
    
        with open(correct_data_file + str(i)) as f:
            correct_data = [_.strip().split(",") for _ in f.readlines()]  # 各行の空白文字と改行文字を削除
            data_num = len(correct_data)
            if suplit_num is None:     # 最初のループで正解データ数が設定されていない場合、設定する
                suplit_num = data_num
            elif suplit_num != data_num:
                suplit_num_differece_flag = True
    
        # 行ごとに比較して同じ値なら統合・削除を行う（統合元は，最初に出現した行）
        unique_correct_data = []
        unique_input_data = []
        delite_sum = 0 #　正解データごとの削除行数
        for idx, row in enumerate(correct_data):
            if row not in unique_correct_data:
                unique_correct_data.append(row)
                unique_input_data.append(input_data[idx])
            else:
                all_delite_sum += 1
                delite_sum += 1
        
        delite_sum_list.append(delite_sum)
        delited_data_num.append(len(unique_correct_data))
    
        if delite_sum > max_delite_sum:
            max_delite_sum = delite_sum
    
        if delite_sum < min_delite_sum or min_delite_sum == -1:
            min_delite_sum = delite_sum
        # print(unique_correct_data)
        # print(unique_input_data)
        
        # 統合された正解データごとにその行番号をリストに格納（行番号は0から始まる）
        duplicate_idx = [[] for _ in range(len(unique_correct_data))]
        for idx, data in enumerate(unique_correct_data):
            for j in range(len(correct_data)):
                if data == correct_data[j]:
                    duplicate_idx[idx].append(j)
        
        # print(duplicate_idx)
    
        
        with open(suplit_input_data_file + str(i), 'w') as f:
            for j in range(len(unique_input_data)):
                f.write(unique_input_data[j] + '\n')
        
        with open(correct_data_delite_file + str(i), 'w') as f:
            for j in range(len(unique_correct_data)):
                for k in range(len(unique_correct_data[j]) - 1):
                    f.write(unique_correct_data[j][k] + ",")
                f.write(unique_correct_data[j][-1] + '\n')
        
        with open(correct_data_integrated_inf_file + str(i), 'w') as f:
            for j in range(len(duplicate_idx)):
                for k in range(len(duplicate_idx[j]) - 1):
                    f.write(str(duplicate_idx[j][k]) + ",")
                f.write(str(duplicate_idx[j][-1]) + '\n')
    
    with open(delite_sum_file, 'w') as f:
        f.write("最大削除行数:" + str(max_delite_sum) + "\n")
        f.write("最小削除行数:" + str(min_delite_sum) + "\n")
        f.write("平均削除行数:" + str(all_delite_sum / suplit_data_num) + "\n")
        f.write("各正解データごとの削除行数\n")
        for i in range(suplit_data_num):
            f.write("正解データ" + str(i) + ":" + str(delite_sum_list[i]) + "\n")
    
    with open(save_delite_inf_file, 'a') as f:
        f.write("モデル分割数:" + str(suplit_data_num) + "\n")
        if suplit_num_differece_flag:
            f.write("1つのモデルの正解データ数:" + str(suplit_num) + "," + str(suplit_num+1) + "\n")
        else:
            f.write("1つのモデルの正解データ数:" + str(suplit_num) + "\n")
        f.write("最大削除行数:" + str(max_delite_sum) + "\n")
        f.write("最小削除行数:" + str(min_delite_sum) + "\n")
        f.write("平均削除行数:" + str(all_delite_sum / suplit_data_num) + "\n")
        f.write("各正解データごとの削除行数\n")
        for i in range(suplit_data_num):
            f.write("正解データ" + str(i) + ":" + str(delite_sum_list[i]) + "\n")
        f.write("\n\n\n")
    
    with open(save_delite_inf_file_sub, 'a') as f:
        f.write("モデル分割数:" + str(suplit_data_num) + "\n")
        if suplit_num_differece_flag:
            f.write("1つのモデルの正解データ数:" + str(suplit_num) + "," + str(suplit_num+1) + "\n")
        else:
            f.write("1つのモデルの正解データ数:" + str(suplit_num) + "\n")
        f.write("最大削除行数:" + str(max_delite_sum) + "\n")
        f.write("最小削除行数:" + str(min_delite_sum) + "\n")
        f.write("平均削除行数:" + str(all_delite_sum / suplit_data_num) + "\n")
        f.write("\n\n")
    
    with open(delited_data_num_file, 'w') as f:
        f.write("平均正解データ数:" + str(sum(delited_data_num) / suplit_data_num) + "\n")
        f.write("最大正解データ数:" + str(max(delited_data_num)) + "\n")
        f.write("最小正解データ数:" + str(min(delited_data_num)) + "\n")
        f.write("各正解データごとの削除後のデータ数\n")
        for i in range(suplit_data_num):
            f.write("正解データ" + str(i) + ":" + str(delited_data_num[i]) + "\n")
    
    with open(delited_data_num_all_file, 'a') as f:
        f.write("モデル分割数:" + str(suplit_data_num) + "\n")
        f.write("平均正解データ数:" + str(sum(delited_data_num) / suplit_data_num) + "\n")
        f.write("最大正解データ数:" + str(max(delited_data_num)) + "\n")
        f.write("最小正解データ数:" + str(min(delited_data_num)) + "\n")
        f.write("各正解データごとの削除後のデータ数\n")
        f.write("\n\n")
    
    group_count = [0] * 9  # 9グループのそれぞれのカウント用リスト
    with open(middle_layer_node_num_file, 'w') as f:
        f.write("各正解データごとの中間層ノード数\n")
        f.write("以下は、N1,N2の順でノード数を記述\n")
        for i in range(suplit_data_num):
            if delited_data_num[i] <= 250:
                group_count[0] += 1
                f.write("25,50\n")
            elif delited_data_num[i] <= 500:
                group_count[1] += 1
                f.write("50,75\n")
            elif delited_data_num[i] <= 750:
                group_count[2] += 1
                f.write("75,100\n")
            elif delited_data_num[i] <= 1000:
                group_count[3] += 1
                f.write("100,125\n")
            elif delited_data_num[i] <= 1250:
                group_count[4] += 1
                f.write("125, 150\n")
            elif delited_data_num[i] <= 1500:
                group_count[5] += 1
                f.write("150,175\n")
            elif delited_data_num[i] <= 1750:
                group_count[6] += 1
                f.write("175,200\n")
            elif delited_data_num[i] <= 2000:
                group_count[7] += 1
                f.write("200,225\n")
            else:
                group_count[8] += 1
                f.write("225,250\n")
    
    
    print("最大削除行数:", max_delite_sum)
    print("最小削除行数:", min_delite_sum)
    print("平均削除行数:", all_delite_sum / suplit_data_num)        
    print("各正解データごとの削除行数:", delite_sum_list)
    print("最大正解データ数:", max(delited_data_num))
    print("最小正解データ数:", min(delited_data_num))
    print("平均正解データ数:", sum(delited_data_num) / suplit_data_num)
    print("グループ1（中間層ノード数25,50）のモデル数:", group_count[0])
    print("グループ2（中間層ノード数50,75）のモデル数:", group_count[1])
    print("グループ3（中間層ノード数75,100）のモデル数:", group_count[2])
    print("グループ4（中間層ノード数100,125）のモデル数:", group_count[3])
    print("グループ5（中間層ノード数125,150）のモデル数:", group_count[4])
    print("グループ6（中間層ノード数150,175）のモデル数:", group_count[5])
    print("グループ7（中間層ノード数175,200）のモデル数:", group_count[6])
    print("グループ8（中間層ノード数200,225）のモデル数:", group_count[7])
    print("グループ9（中間層ノード数225,250）のモデル数:", group_count[8])


if __name__ == "__main__":
    main()