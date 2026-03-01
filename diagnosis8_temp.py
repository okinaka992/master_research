# ...existing code...
import numpy as np
import tensorflow as tf
import random
import os

# グローバル変数
cir = 's5378'  # 対象回路
tp_file = cir + '.vec'  # テストパターンファイル名
part_stdic_file = cir + "stdic_bi/aout_" #縮退故障辞書ファイル名の一部
part_brdic_file = cir + "brdic_bi/aout_" #ブリッジ故障辞書ファイルの一部
st_diagnosis_dir = cir + "diagnosis_st_data/" #縮退故障診断を行うためのデータを保存するフォルダ
br_diagnosis_dir = cir + "diagnosis_br_data/" #ブリッジ故障診断を行うためのデータを保存するフォルダ
target_fault_num = 30
correct_output_file = 'correct_output/' + cir + '_correct_output'
input_data_original_file = cir + 'input'
input_data_supulit_file = cir + '分割入力データ/' + cir + 'input'
correct_data_file = cir + '分割正解データ削除後/' + cir + 'integrated_output'
correct_data_integrated_inf_file = cir + '分割正解データ削除後/' + cir + 'integrated_inf'
correct_data_value_file = cir + '分割正解データ' + '/' + cir + 'correct_value'
threshold_file = cir + '分割正解データ' + '/' + cir + 'threshold'
suplit_num_file = cir + '分割正解データ' + '/' + cir + 'suplit_num'
suplit_data_num_file = cir + '分割正解データ' + '/' + cir + 'suplit_data_num'
single_line_file = cir + '分割正解データ' + '/' + cir + 'single_line'
single_line_inf_file = cir + '分割正解データ' + '/' + cir + 'single_line_inf'
single_line_model = None
original_input_data_num = None
suplited_input_data_num = None
output_node_num = None
num_models = None
model_folder = None
fault_line_num = None
fault_type_sum = 12
correct_value = None
threshold = None
all_models_learn = True
processes = 8

def setting_original_input_data_num():
    with open(input_data_original_file) as f:
        lines = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]
    global original_input_data_num
    original_input_data_num = int(len(lines))

def mk_input_data(input_data_file):
    with open(input_data_file) as f:
        lines = [_.replace(",", "").replace("\n", "") for _ in f.readlines()]
    global suplited_input_data_num
    suplited_input_data_num = int(len(lines))
    int_lines = [list(map(int, _)) for _ in lines]
    return np.array(int_lines)

def mk_output_data(fname):
    with open(fname) as f:
        lines = [_.replace("\n", "") for _ in f.readlines()]
    lines = [value.split(",") for value in lines]
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            lines[i][j] = float(lines[i][j])
    global output_node_num
    output_node_num = int(len(lines[0]))
    print("output_node_num:", output_node_num)

def get_fault_target(fault_all_line, fault_flag):
    fault_target_line = random.sample(fault_all_line, target_fault_num)
    fault_target_line = sorted(fault_target_line)
    if fault_flag == 0:
        fault_target_type = random.choices([0, 1], k=target_fault_num)
        with open(st_diagnosis_dir + 'st_fault_line', 'w') as f:
            f.write('診断対象縮退故障数' + str(target_fault_num) + '\n')
            for i in range(target_fault_num):
                f.write(str(fault_target_line[i]) + ' ' + "sa " + str(fault_target_type[i]) + "\n")
    else:
        fault_target_type = random.choices(list(range(10)), k=target_fault_num)
        with open(br_diagnosis_dir + 'br_fault_line', 'w') as f:
            f.write('診断対象ブリッジ故障数' + str(target_fault_num) + '\n')
            for i in range(target_fault_num):
                f.write(str(fault_target_line[i]) + ' ' + "br_type " + str(fault_target_type[i]) + "\n")
    print(fault_target_line)
    print(fault_target_type)
    return fault_target_line, fault_target_type

def get_fault_output(tp_num, fault_target_line, fault_target_type, fault_flag):
    if fault_flag == 0:
        part_dic_file = part_stdic_file
        diagnosis_file = st_diagnosis_dir + 'fault_output'
    else:
        part_dic_file = part_brdic_file
        diagnosis_file = br_diagnosis_dir + 'fault_output'
        print(fault_target_line)
        print(fault_target_type)

    for i in range(tp_num):
        with open(part_dic_file + str(i), 'r') as f:
            fault_inf = f.readline()
            lines = f.readlines()
        fault_output = [0 for _ in range(target_fault_num)]
        idx = 0
        br_type_count = 0
        for j in range(0, len(lines), 2):
            if fault_flag == 0:
                if fault_target_line[idx] == int(lines[j].split()[3]) and fault_target_type[idx] == int(lines[j].split()[5]):
                    fault_output[idx] = lines[j+1].replace('\n', '')
                    idx += 1
            else:
                if fault_target_line[idx] == int(lines[j].split()[3]) and fault_target_type[idx] == br_type_count:
                    fault_output[idx] = lines[j+1].replace('\n', '')
                    idx += 1
                br_type_count += 1
                if br_type_count == 10:
                    br_type_count = 0
            if idx == target_fault_num:
                break
        with open(diagnosis_file + str(i), 'w') as f:
            for j in range(len(fault_output)):
                f.write(str(fault_output[j]) + '\n')

def get_pass_fail(tp_num, fault_flag):
    if fault_flag == 0:
        fault_output_file = st_diagnosis_dir + 'fault_output'
        pass_fail_file = st_diagnosis_dir + 'pass_fail'
    else:
        fault_output_file = br_diagnosis_dir + 'fault_output'
        pass_fail_file = br_diagnosis_dir + 'pass_fail'

    correct_output_value = []
    with open(correct_output_file, 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines), 2):
            correct_output_value.append(lines[i].replace('\n', ''))
    print(f"correct_output_value: {correct_output_value}")

    fault_output_value = [[0 for _ in range(tp_num)] for _ in range(target_fault_num)]
    for i in range(tp_num):
        with open(fault_output_file + str(i), 'r') as f:
            lines = f.readlines()
            for j in range(target_fault_num):
                fault_output_value[j][i] = lines[j].replace('\n', '')

    pass_fail = [[0 for _ in range(tp_num)] for _ in range(target_fault_num)]
    for i in range(target_fault_num):
        with open(pass_fail_file + str(i), 'w') as f:
            for j in range(tp_num):
                if correct_output_value[j] == fault_output_value[i][j]:
                    pass_fail_value = '0'
                    pass_fail[i][j] = 0
                else:
                    pass_fail_value = '1'
                    pass_fail[i][j] = 1
                f.write(pass_fail_value)
    return pass_fail

def get_fault_candidate(pass_fail, compare_model_pass_fail, fault_all_line, signal_num, br_missing_line, fault_flag):
    if fault_flag == 0:
        fault_type_num = 2
    else:
        fault_type_num = 10
        signal_num = signal_num - len(br_missing_line)

    fault_candidate = [[] for _ in range(target_fault_num)]
    fault_type = [[] for _ in range(target_fault_num)]
    for i in range(target_fault_num):
        if all( x == 0 for x in pass_fail[i]) == True:
            fault_candidate[i].append(-1)
            fault_type[i].append(-1)
            print(f"fault_candidate：{fault_candidate[i]}")
            continue
        line_count = 0
        count = 0
        for j in range(signal_num*fault_type_num):
            if compare_model_pass_fail[j] == pass_fail[i]:
                fault_candidate[i].append(fault_all_line[line_count])
                fault_type[i].append(count%fault_type_num)
            count += 1
            if count == fault_type_num:
                line_count += 1
                count = 0
        if len(fault_candidate[i]) == 0:
            fault_candidate[i].append(0)
            fault_type[i].append(0)
        print(f"fault_candidate[{i}]: {len(fault_candidate[i])}")
    return fault_candidate, fault_type

def check_fault_candidate(fault_candidate, fault_type, fault_target_line, fault_target_type, fault_flag):
    if fault_flag == 0:
        correct_fault_candidate_file = st_diagnosis_dir + 'correct_fault_candidate'
        incorrect_fault_candidate_file = st_diagnosis_dir + 'incorrect_fault_candidate'
        diagnosis_rate_file = st_diagnosis_dir + 'diagnosis_rate'
        fault_name = '縮退故障'
        fault_symbol = ' sa '
    else:
        correct_fault_candidate_file = br_diagnosis_dir + 'correct_fault_candidate'
        incorrect_fault_candidate_file = br_diagnosis_dir + 'incorrect_fault_candidate'
        diagnosis_rate_file = br_diagnosis_dir + 'diagnosis_rate'
        fault_name = 'ブリッジ故障'
        fault_symbol = ' br_type '

    print(f"{fault_name}の診断を行います")
    print(f"fault_target_line:{fault_target_line}")

    find_count = 0
    not_find_count = 0
    indistinguishable_fault_count = 0
    fault_line_candidate_sum = 0
    fault_type_candidate_sum = 0
    with open(correct_fault_candidate_file, 'w') as f:
        with open(incorrect_fault_candidate_file, 'w') as g:
            for i in range(target_fault_num):
                fault_candidate[i] = sorted(fault_candidate[i])
                if fault_candidate[i] == [-1]:
                    indistinguishable_fault_count += 1
                    print(f"区別できない故障のため、故障信号線{fault_target_line[i]} の診断は行いませんでした")
                    f.write("区別できない故障のため、故障信号線" + str(fault_target_line[i]) + " の診断は行いませんでした\n")
                    g.write("区別できない故障のため、故障信号線" + str(fault_target_line[i]) + " の診断は行いませんでした\n")
                    continue
                if fault_candidate[i] == [0]:
                    print(f"実際に選んだ故障 {fault_target_line[i]} の故障候補は0でした")
                    g.write("故障候補が0だった故障：" + str(fault_target_line[i]) + fault_symbol + str(fault_target_type[i]) + '\n')
                    not_find_count += 1
                    continue
                fault_line_candidate_sum += len(set(fault_candidate[i]))
                fault_type_candidate_sum += len(fault_candidate[i])
                for j in range(len(fault_candidate[i])):
                    if fault_candidate[i][j] == fault_target_line[i] and fault_type[i][j] == fault_target_type[i]:
                        find_count += 1
                        print(f"{fault_name}候補の中に実際に選んだ{fault_name}番号 {fault_candidate[i][j]} の {fault_type[i][j]} {fault_name}が含まれていました")
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

if __name__ == '__main__':

    if all_models_learn:
        model_folder = cir + 'sepmodel2'
    else:
        model_folder = cir + 'sepmodel_check2'

    with open(correct_data_value_file, 'r') as f:
        correct_value = [float(value) for value in f.readline().split()]

    with open(threshold_file, 'r') as f:
        threshold = [float(value) for value in f.readline().split()]

    print("正解データの種類（値）：", correct_value)
    print("閾値：", threshold)

    with open(tp_file, 'r') as f:
        line = f.readline()
        tp_num = int(line.split()[0])
        print(tp_num)

    with open(part_stdic_file + str(0), 'r') as f:
        fault_inf = f.readline()
        st_fault_num = int(fault_inf.split()[2])
        lines = f.readlines()[::2]
        st_fault_all_line = [int(line.split()[3]) for line in lines[::2]]

    with open(part_brdic_file + str(0), 'r') as f:
        fault_inf = f.readline()
        br_fault_num = int(fault_inf.split()[2])
        lines = f.readlines()[::2]
        br_fault_all_line = [int(line.split()[3]) for line in lines[::10]]

    st_fault_target_line, st_fault_target_type = get_fault_target(st_fault_all_line, 0)
    br_fault_target_line, br_fault_target_type = get_fault_target(br_fault_all_line, 1)

    get_fault_output(tp_num, st_fault_target_line, st_fault_target_type, 0)
    get_fault_output(tp_num, br_fault_target_line, br_fault_target_type, 1)

    st_pass_fail = get_pass_fail(tp_num, 0)
    br_pass_fail = get_pass_fail(tp_num, 1)

    setting_original_input_data_num()

    with open(suplit_num_file, 'r') as f:
        num_models = int(f.readline())

    with open(cir + 'output', 'r') as f:
        line = f.readline().replace(",", "").replace("\n", "")
        fault_line_num = int(len(line))

    model_pass_fail_np = np.zeros((original_input_data_num, fault_line_num), dtype=np.uint8)
    print("model_pass_fail shape:", model_pass_fail_np.shape)

    with open('c' + cir, 'r') as f:
        line_inf = f.readline()
        output_line_num = int(line_inf.split()[1])
        input_line_num = int(line_inf.split()[2])

    with open(single_line_file, 'r') as f:
        single_flag = int(f.readline().replace("\n", ""))

    with open(cir + 'pair_list2', 'r') as f:
        integration_num = int(f.readline().split()[1])
        signal_num = int(f.readline().split()[1])
        print("signal_num", signal_num)
        raw_lines = [l.strip() for l in f.readlines() if l.strip()]
        print(raw_lines)

    groups = []
    for l in raw_lines:
        toks = l.split()
        nums = [int(x) for x in toks]
        if len(nums) == integration_num or len(nums) == 1:
            groups.append(nums)
        else:
            for i in range(0, len(nums), integration_num):
                groups.append(nums[i:i+integration_num])

    line_id = []
    for g in groups:
        line_id.extend(g)

    print("len(line_id):", len(line_id))
    if len(line_id) != signal_num:
        print(f"警告: 期待する信号線数 {signal_num} と実際のトークン数 {len(line_id)} が一致しません。")

    if single_flag == 1:
        with open(single_line_inf_file, 'r') as f:
            _ = f.readline().replace("\n", "")
            single_line_model = int(f.readline().replace("\n", ""))
            single_line_suplit_idx = int(f.readline().replace("\n", ""))

    line_id = [i - 1 if isinstance(i, int) else int(i)-1 for i in line_id]
    print(line_id[0], line_id[1], line_id[-1])

    with open(suplit_data_num_file, 'r') as f:
        suplit_data_num = f.readlines()
        for i in range(len(suplit_data_num)):
            suplit_data_num[i] = suplit_data_num[i].replace("\n","")
        print(suplit_data_num)

    suplit_data_sum_count = 0
    for model_id in range(num_models):
        input_data_file = input_data_supulit_file + str(model_id)
        input_data = mk_input_data(input_data_file)

        with open(correct_data_integrated_inf_file + str(model_id), 'r') as f:
            correct_data_integrated_inf = [_.replace("\n", "").split(",") for _ in f.readlines()]

        with open(os.path.join(model_folder, cir + 'model_' + str(model_id) + '.tflite'), 'rb') as f:
            tflite_model = f.read()

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        correct_file = correct_data_file + str(model_id)
        mk_output_data(correct_file)

        # prepare column index mapping for this model (replicates original count logic)
        col_a = []
        col_b = []
        count = 0
        for j in range(output_node_num):
            if (model_id == single_line_model) and single_flag == 1 and j == single_line_suplit_idx:
                col_a.append(line_id[j + count + suplit_data_sum_count])
                col_b.append(-1)
            else:
                col_a.append(line_id[j + count + suplit_data_sum_count])
                col_b.append(line_id[j + count + suplit_data_sum_count + 1])
                count += 1
        col_a = np.array(col_a, dtype=np.int64)
        col_b = np.array(col_b, dtype=np.int64)

        # --- 追加: インデックス検査と安全化 ---
        if np.any(col_a < 0) or np.any(col_a >= fault_line_num) or np.any(col_b < -1) or np.any(col_b >= fault_line_num):
            print("警告: col_a/col_b にモデル外のインデックスが含まれています。詳細を表示します。")
            print("fault_line_num:", fault_line_num)
            print("col_a min/max:", col_a.min(), col_a.max())
            print("col_b min/max:", col_b.min(), col_b.max())
            bad_a_idx = np.where((col_a < 0) | (col_a >= fault_line_num))[0]
            bad_b_idx = np.where((col_b < 0) | (col_b >= fault_line_num))[0]
            if bad_a_idx.size:
                print("不正な col_a のインデックス位置:", bad_a_idx.tolist(), "値:", col_a[bad_a_idx].tolist())
            if bad_b_idx.size:
                print("不正な col_b のインデックス位置:", bad_b_idx.tolist(), "値:", col_b[bad_b_idx].tolist())
        valid_a_mask = (col_a >= 0) & (col_a < fault_line_num)
        valid_b_mask = (col_b >= 0) & (col_b < fault_line_num)
        col_a_valid = col_a[valid_a_mask]
        col_b_valid = col_b[valid_b_mask]

        # バッチ推論：パターンをまとめて一度に推論して速度改善
        N = suplited_input_data_num
        batch_size = 32  # 環境に合わせて調整
        input_shape = input_details[0]['shape']
        orig_input_shape = input_shape.copy()
        feat_shape = tuple(input_shape[1:]) if len(input_shape) > 1 else ()
        using_resize = False
        try:
            interpreter.resize_tensor_input(input_details[0]['index'], [batch_size] + list(feat_shape))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            using_resize = True
        except Exception:
            using_resize = False

        mask_b = col_b >= 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b = end - start
            batch_inputs = np.zeros((b,) + feat_shape, dtype=np.float32)
            for idx in range(b):
                inp = input_data[start + idx]
                if feat_shape:
                    batch_inputs[idx] = np.reshape(inp, feat_shape)
                else:
                    batch_inputs[idx] = inp

            if using_resize and b != batch_size:
                interpreter.resize_tensor_input(input_details[0]['index'], [b] + list(feat_shape))
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'], batch_inputs.astype(np.float32))
            interpreter.invoke()
            outputs = interpreter.get_tensor(output_details[0]['index'])  # shape (b, M)
            classes_batch = np.empty_like(outputs, dtype=np.int8)
            classes_batch[:] = 3
            classes_batch[outputs < threshold[2]] = 2
            classes_batch[outputs < threshold[1]] = 1
            classes_batch[outputs < threshold[0]] = 0

            if (model_id == single_line_model) and single_flag == 1:
                idx = single_line_suplit_idx
                classes_batch[:, idx] = np.where(outputs[:, idx] <= threshold[1], 0, 3)

            a_batch = ((classes_batch == 1) | (classes_batch == 3)).astype(np.uint8)
            b_batch = ((classes_batch == 2) | (classes_batch == 3)).astype(np.uint8)

            rows = np.arange(start, end)

            # --- 修正: 有効インデックスのみで安全に代入 ---
            if col_a_valid.size:
                a_batch_valid = a_batch[:, valid_a_mask]
                model_pass_fail_np[np.ix_(rows, col_a_valid)] = a_batch_valid
            if col_b_valid.size:
                b_batch_valid = b_batch[:, valid_b_mask]
                model_pass_fail_np[np.ix_(rows, col_b_valid)] = b_batch_valid

        if using_resize:
            try:
                interpreter.resize_tensor_input(input_details[0]['index'], orig_input_shape)
                interpreter.allocate_tensors()
            except Exception:
                pass

        if model_id == single_line_model and single_flag == 1:
            suplit_data_sum_count += integration_num*output_node_num - 1
        else:
            suplit_data_sum_count += integration_num*output_node_num

    model_pass_fail = model_pass_fail_np.T.tolist()

    compare_model_pass_fail = [[0 for _ in range(tp_num)] for _ in range(signal_num*fault_type_sum)]
    for i in range(signal_num):
        for j in range(fault_type_sum):
            for k in range(tp_num):
                compare_model_pass_fail[i*fault_type_sum + j][k] = model_pass_fail[i][k*fault_type_sum + j]

    st_compare_model_pass_fail = [[0 for _ in range(tp_num)] for _ in range(signal_num*2)]
    br_compare_model_pass_fail = [[0 for _ in range(tp_num)] for _ in range(signal_num*(fault_type_sum - 2))]

    st_idx = 0
    br_idx = 0
    for i in range(signal_num*fault_type_sum):
        if (i%fault_type_sum == 0) or (i%fault_type_sum == 1):
            for j in range(tp_num):
                st_compare_model_pass_fail[st_idx][j] = compare_model_pass_fail[i][j]
            st_idx += 1
        else:
            for j in range(tp_num):
                br_compare_model_pass_fail[br_idx][j] = compare_model_pass_fail[i][j]
            br_idx += 1

    br_fault_all_line = sorted(br_fault_all_line)
    with open("c" + cir, 'r') as f:
        line = f.readline()
        all_signal_num  = int(line.split()[0])
        cir_output_line_num = int(line.split()[1])
        cir_input_line_num = int(line.split()[2])

    br_missing_line = [line for line in range(1, (all_signal_num + 1)) if line not in br_fault_all_line]
    for i in range(cir_output_line_num):
        br_missing_line.remove(i + cir_input_line_num + 1)
    br_missing_line = sorted(br_missing_line, reverse=True)
    print("br_missing_line：", br_missing_line)
    print("br_missing_lineの長さ：", len(br_missing_line))

    for i in range(len(br_missing_line)):
        if br_missing_line[i] <= cir_input_line_num:
            for j in range(10):
                br_compare_model_pass_fail.pop((br_missing_line[i] - 1)*10)
        else:
            for j in range(10):
                br_compare_model_pass_fail.pop((br_missing_line[i] - cir_output_line_num - 1)*10)

    st_fault_candidate, st_fault_type = get_fault_candidate(st_pass_fail, st_compare_model_pass_fail, st_fault_all_line, signal_num, br_missing_line, 0)
    br_fault_candidate, br_fault_type = get_fault_candidate(br_pass_fail, br_compare_model_pass_fail, br_fault_all_line, signal_num, br_missing_line, 1)

    check_fault_candidate(st_fault_candidate, st_fault_type, st_fault_target_line, st_fault_target_type, 0)
    check_fault_candidate(br_fault_candidate, br_fault_type, br_fault_target_line, br_fault_target_type, 1)
# ...existing code...