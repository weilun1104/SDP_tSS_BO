import numpy as np
from sklearn.feature_selection import mutual_info_regression
import torch

def calculate_mutual_information(X, Y, n_neighbors=3, n_repeats=10):
    """
    计算输入X和输出Y之间的平均互信息。

    参数:
    X -- 输入数据，大小为(samples, n_inputs)的二维数组
    Y -- 输出数据，大小为(samples, n_outputs)的二维数组
    n_neighbors -- mutual_info_regression中使用的邻居数量
    n_repeats -- 重复计算互信息的次数以获得平均值

    返回:
    mi_avg -- 平均互信息值的数组，大小为(n_outputs, n_inputs)
    """
    n_samples, n_inputs = X.shape
    n_outputs = Y.shape[1]
    mi_avg = np.zeros((n_outputs, n_inputs))  # 初始化平均互信息数组

    for repeat in range(n_repeats):
        for output_index in range(n_outputs):
            mi = mutual_info_regression(X, Y[:, output_index], n_neighbors=n_neighbors)
            mi_avg[output_index] += mi

    mi_avg /= n_repeats  # 计算平均互信息值

    return mi_avg

def filter_two_rows(y):
    # 创建一个空的列表来保存满足条件的行
    modified_y = []
    FoM_num = 1
    # 第一列乘以 20
    y[:, 0] = y[:, 0] * 20
    # 第二列取负
    y[:, 1] = -y[:, 1]
    # 后三列取exp
    y[:, 1:] = torch.exp(y[:, 1:])

    # 遍历y的每一行，对应的索引也用于检索x中的行
    for i, row in enumerate(y):
        # 检查是否满足特定条件
        if (row[0] > 60 and row[1] * 1.8 < 1e-3 and row[2] > 60 and row[3] > 4e6):
            # 计算满足条件时的y
            modified_value = 3 + ((1e-3/1.8) / row[1]) * 50
            FoM_num += 1
            # modified_value = ((1e-3 / 1.8) / row[1]) * 5
            modified_y.append(modified_value)
        else:
            # 计算不满足条件时的y
            value_1 = min(row[0], 60) / 60
            value_2 = min((1.0e-3/1.8)/row[1], 1)
            value_3 = min(row[2], 60) / 60
            value_4 = min(row[3], 4e6) / 4e6

            # modified_value = value_1 + value_2 + value_3 + value_4
            modified_value = 0
            modified_y.append(modified_value)

    # 将结果转换回tensor形式
    modified_y = torch.tensor(modified_y) if modified_y else torch.tensor([])
    # non_zero_indices = torch.nonzero(modified_y).squeeze()
    # modified_y = modified_y[non_zero_indices]
    # x = x[non_zero_indices]
    modified_y = modified_y.unsqueeze(1)
    print("指标：", modified_y)
    return modified_y, FoM_num

def filter_three_rows(y):
    # 创建一个空的列表来保存满足条件的行
    modified_y = []
    FoM_num = 1
    # 第一列乘以 20
    y[:, 0] = y[:, 0] * 20
    # 第二列取负
    y[:, 1] = -y[:, 1]
    # 后三列取exp
    y[:, 1:] = torch.exp(y[:, 1:])

    # 遍历y的每一行，对应的索引也用于检索x中的行
    for i, row in enumerate(y):
        # 检查是否满足特定条件
        if (row[0] > 80 and row[1] * 1.8 < 1e-3 and row[2] > 60 and row[3] > 2e6):
            # 计算满足条件时的y
            modified_value = 3 + ((1e-3/1.8) / row[1]) * 50
            FoM_num += 1
            # modified_value = ((1e-3 / 1.8) / row[1]) * 5
            modified_y.append(modified_value)
        else:
            # 计算不满足条件时的y
            value_1 = min(row[0], 80) / 80
            value_2 = min((1.0e-3/1.8)/row[1], 1)
            value_3 = min(row[2], 60) / 60
            value_4 = min(row[3], 2e6) / 2e6

            # modified_value = value_1 + value_2 + value_3 + value_4
            modified_value = 0
            modified_y.append(modified_value)

    # 将结果转换回tensor形式
    modified_y = torch.tensor(modified_y) if modified_y else torch.tensor([])
    modified_y = modified_y.unsqueeze(1)
    print("指标：", modified_y)
    return modified_y, FoM_num

def filter_bandgap_rows(y, flag=0):
    # 创建一个空的列表来保存满足条件的行
    modified_y = []
    FoM_num = 1
    # 第一二列取exp
    if flag == 0:       # 需要取负
        y[:, 0] = torch.exp(-y[:, 0])
        y[:, 1] = torch.exp(-y[:, 1])
        y[:, 2] = y[:, 2]
    # else:               # 无需取负
    #     y[:, 0] = torch.exp(y[:, 0])
    #     y[:, 1] = torch.exp(y[:, 1])
    # 第三列直接赋值


    # 遍历y的每一行，对应的索引也用于检索x中的行
    for i, row in enumerate(y):
        # 检查是否满足特定条件
        if row[0] < 200 and row[1] < 5e-5 and row[2] > 60:
            # 计算满足条件时的y
            modified_value = 2 + (200 / row[0]) * 50
            FoM_num += 1
            # modified_value = ((1e-3 / 1.8) / row[1]) * 5
            modified_y.append(modified_value)
        else:
            modified_value = 0
            modified_y.append(modified_value)

    # 将结果转换回tensor形式
    modified_y = torch.tensor(modified_y) if modified_y else torch.tensor([])
    modified_y = modified_y.unsqueeze(1)
    print("指标：", modified_y)
    return modified_y, FoM_num


def cal_score(mi_results, index, weights):
    return sum(mi_results[i][index] * weight for i, weight in enumerate(weights))


# 计算二级、三级每个变量铭感分数
def calculate_scores(dbx, dby, FoM_num, I_num, gain_num, GBW_num, phase_num, iter, init_num, n_neighbors=3, n_repeats=10, input_dim=12):
    gain_weight = ((iter + init_num + 1) - gain_num) / (iter + init_num + 1)
    # gain_weight = 0
    # I_weight = 0
    I_weight = ((iter + init_num + 1) - I_num) / (iter + init_num + 1)
    GBW_weight = ((iter + init_num + 1) - GBW_num) / (iter + init_num + 1)
    phase_weight = ((iter + init_num + 1) - phase_num) / (iter + init_num + 1)
    # GBW_weight = 0
    # phase_weight = 0
    FoM_weight = ((iter + init_num + 1) - FoM_num) / (iter + init_num + 1) + 1
    # FoM_weight = ((iter + init_num + 1) - FoM_num) / (iter + init_num + 1)
    # FoM_weight = 1
    if isinstance(dbx, torch.Tensor):
        dbx = dbx.numpy()
    if isinstance(dby, torch.Tensor):
        dby = dby.numpy()
    # 调用函数计算互信息
    mi_results = calculate_mutual_information(dbx, dby, n_neighbors=n_neighbors, n_repeats=n_repeats)

    weights = [gain_weight, I_weight, phase_weight, GBW_weight, FoM_weight]

    indices = range(input_dim)
    scores_list = [cal_score(mi_results, i, weights) for i in indices]

    scores = torch.tensor(scores_list)

    # 打印结果
    for i, mi in enumerate(mi_results):
        print(f"Average Mutual Information with output {i + 1}: {mi}")

    # # 更新权重
    # all_weight = ScalarizedObjective(weights=torch.tensor([gain_weight, I_weight, phase_weight, GBW_weight]))
    return scores


def calculate_bandgap_scores(dbx, dby, FoM_num, ppm_num,I_num, psrr_num, iter, init_num, n_neighbors=3, n_repeats=10,
                             input_dim=20):
    ppm_weight = 1
    # I_weight = ((iter + init_num + 1) - I_num) / (iter + init_num + 1)
    # psrr_weight = ((iter + init_num + 1) - psrr_num) / (iter + init_num + 1)
    # FoM_weight = ((iter + init_num + 1) - FoM_num) / (iter + init_num + 1) + 1
    I_weight = 0
    psrr_weight = 0
    FoM_weight  = 0
    if isinstance(dbx, torch.Tensor):
        dbx = dbx.numpy()
    if isinstance(dby, torch.Tensor):
        dby = dby.numpy()
    # 调用函数计算互信息
    mi_results = calculate_mutual_information(dbx, dby, n_neighbors=n_neighbors, n_repeats=n_repeats)

    weights = [ppm_weight, I_weight, psrr_weight, FoM_weight]

    indices = range(input_dim)
    scores_list = [cal_score(mi_results, i, weights) for i in indices]

    scores = torch.tensor(scores_list)

    # 打印结果
    for i, mi in enumerate(mi_results):
        print(f"Average Mutual Information with output {i + 1}: {mi}")

    return scores
