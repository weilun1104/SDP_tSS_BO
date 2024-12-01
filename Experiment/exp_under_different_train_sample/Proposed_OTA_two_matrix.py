# 二级运放全解绑+MI+逐渐解绑
import random

import torch

from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_V3 import *
from botorch.acquisition.objective import ScalarizedObjective, ScalarizedPosteriorTransform
from botorch.models.transforms.outcome import Standardize
from Model.Point_search.CONBO import BayesianOptimization, plot
from MI_calc import calculate_mutual_information
from config_OTA_two import init_OTA_two

# 将参数都设置维double类型
torch.set_default_dtype(torch.double)

# 设置实验种子
SEED = 1
random.seed(SEED)
np.random.seed(SEED)  # 设置 NumPy 的种子
torch.manual_seed(SEED)


# 计算每个变量铭感分数
def calculate_scores(dbx, dby, gain_num, GBW_num, phase_num, iter, n_neighbors=3, n_repeats=10):
    gain_weight = ((iter + 1) - gain_num) / (iter + 1)
    I_weight = 1
    GBW_weight = ((iter + 1) - GBW_num) / (iter + 1)
    phase_weight = ((iter + 1) - phase_num) / (iter + 1)
    dbx = dbx.numpy()
    dby = dby.numpy()
    # 调用函数计算互信息
    mi_results = calculate_mutual_information(dbx, dby, n_neighbors=n_neighbors, n_repeats=n_repeats)

    cap_score = mi_results[0][0] * gain_weight + mi_results[1][0] * I_weight + mi_results[2][0] * phase_weight + \
                mi_results[3][0] * GBW_weight
    l1_score = mi_results[0][1] * gain_weight + mi_results[1][1] * I_weight + mi_results[2][1] * phase_weight + \
               mi_results[3][1] * GBW_weight
    l2_score = mi_results[0][2] * gain_weight + mi_results[1][2] * I_weight + mi_results[2][2] * phase_weight + \
               mi_results[3][2] * GBW_weight
    l3_score = mi_results[0][3] * gain_weight + mi_results[1][3] * I_weight + mi_results[2][3] * phase_weight + \
               mi_results[3][3] * GBW_weight
    l4_score = mi_results[0][4] * gain_weight + mi_results[1][4] * I_weight + mi_results[2][4] * phase_weight + \
               mi_results[3][4] * GBW_weight
    l5_score = mi_results[0][5] * gain_weight + mi_results[1][5] * I_weight + mi_results[2][5] * phase_weight + \
               mi_results[3][5] * GBW_weight
    r_score = mi_results[0][6] * gain_weight + mi_results[1][6] * I_weight + mi_results[2][6] * phase_weight + \
              mi_results[3][6] * GBW_weight
    w1_score = mi_results[0][7] * gain_weight + mi_results[1][7] * I_weight + mi_results[2][7] * phase_weight + \
               mi_results[3][7] * GBW_weight
    w2_score = mi_results[0][8] * gain_weight + mi_results[1][8] * I_weight + mi_results[2][8] * phase_weight + \
               mi_results[3][8] * GBW_weight
    w3_score = mi_results[0][9] * gain_weight + mi_results[1][9] * I_weight + mi_results[2][9] * phase_weight + \
               mi_results[3][9] * GBW_weight
    w4_score = mi_results[0][10] * gain_weight + mi_results[1][10] * I_weight + mi_results[2][10] * phase_weight + \
               mi_results[3][10] * GBW_weight
    w5_score = mi_results[0][11] * gain_weight + mi_results[1][11] * I_weight + mi_results[2][11] * phase_weight + \
               mi_results[3][11] * GBW_weight

    # 打印结果
    for i, mi in enumerate(mi_results):
        print(f"Average Mutual Information with output {i + 1}: {mi}")

    print('cap_score:', cap_score)
    print('L1_score:', l1_score)
    print('L2_score:', l2_score)
    print('L3_score:', l3_score)
    print('L4_score:', l4_score)
    print('L5_score:', l5_score)
    print('R_score:', r_score)
    print('W1_score:', w1_score)
    print('W2_score:', w2_score)
    print('W3_score:', w3_score)
    print('W4_score:', w4_score)
    print('W5_score:', w5_score)

    # 更新权重
    all_weight = ScalarizedObjective(weights=torch.tensor([gain_weight, I_weight, phase_weight, GBW_weight]))
    return (
    cap_score, l1_score, l2_score, l3_score, l4_score, l5_score, r_score, w1_score, w2_score, w3_score, w4_score,
    w5_score, all_weight)


# 获取阶段解绑参数标签和范围
def get_indices_and_ranges(param_name, init_val, *params_percentage):
    params_idx = []
    param_range = []
    for param, percentage in params_percentage:
        idx = param_name.index(param)
        params_idx.append(idx)
        param_range.append(set_param_ranges(init_val[idx], percentage))
    return params_idx, param_range


# 根据敏感性进行排序
def sort_and_group(scores):
    # 使用zip将参数名称与它们的分数关联起来，然后排序
    param_scores = zip(["cap", "L1", "L2", "L3", "L4", "L5", "r", "W1", "W2", "W3", "W4", "W5"], scores)
    # 根据分数从高到低进行排序
    sorted_params = sorted(param_scores, key=lambda x: x[1], reverse=True)
    # 分组，每两个为一组
    grouped_params = [(sorted_params[i], sorted_params[i + 1], sorted_params[i + 2], sorted_params[i + 3]) for i in
                      range(0, len(sorted_params), 4)]
    return grouped_params


# 定义参数范围设置函数
def set_param_ranges(param_initial_value, percentage=0.35):
    param_min = param_initial_value * (1 - percentage)
    param_max = param_initial_value * (1 + percentage)
    return param_min, param_max


# 挑选满足约束的x和y值用于MI分析
def filter_rows(x, y):
    # 创建一个空的列表来保存满足条件的行
    selected_x = []
    selected_y = []
    # 第一列乘以 20
    y[:, 0] = y[:, 0] * 20
    y[:, 1] = -y[:, 1]
    # 后三列取exp
    y[:, 1:] = torch.exp(y[:, 1:])

    # 遍历y的每一行，对应的索引也用于检索x中的行
    for i, y in enumerate(y):
        if (y[0] > 60 and y[1] * 1.8 < 1e-3 and y[2] > 60 and y[3]
                > 8e6):
            selected_x.append(x[i])
            selected_y.append(y)

    # 将结果转换回tensor形式
    selected_x = torch.stack(selected_x) if selected_x else torch.tensor([])
    selected_y = torch.stack(selected_y) if selected_y else torch.tensor([])

    return selected_x, selected_y


if __name__ == "__main__":
    # param_ranges, thresholds, valid_x, valid_y, dbx_alter, dby_alter, last_valid_x, last_valid_y = init_OTA_two()
    #
    # # 定义初始满足约束的点的个数
    # gain_num = 1
    # I_num = 1
    # GBW_num = 1
    # phase_num = 1
    #
    # # 初步设定权重，四个数分别是gian，I,phase,GBW，用于四个参数同一成一个，进行后续的gp训练
    # objective = ScalarizedObjective(weights=torch.tensor([0.9, 1.0, 0.5, 0.8]))
    # posterior_transform = Standardize(m=1)  # 将后验的均值和方差标准化
    #
    # # 初步全解绑BO迭代次数,用于后续MI分析
    # iter_1 = 40
    #
    # # 创建BayesianOptimization实例
    # bo = BayesianOptimization(
    #     param_ranges=param_ranges,  # 参数范围
    #     n=iter_1,  # 迭代次数
    #     simulation_function=OTA_two_simulation_all,  # ngspice
    #     objective=objective,  # y的参数权重
    #     posterior_transform=posterior_transform,
    #     mode='collect_all',  # 全解绑过程
    #     best_y=last_valid_y[1],  # 设置为饱和区搜索出来的最好电流
    #     dbx_alter=dbx_alter,
    #     dby_alter=dby_alter,
    #     valid_x=valid_x,
    #     valid_y=valid_y,
    #     last_valid_x=last_valid_x,
    #     last_valid_y=last_valid_y,
    #     gain_num=gain_num,
    #     I_num=I_num,
    #     GBW_num=GBW_num,
    #     phase_num=phase_num,
    #     thresholds=thresholds,
    #     stage='first'  # 最初阶段，用于记录全解绑gp
    # )
    #
    # # iter_times = [x + 1 for x in range(iter+1)]
    # # 设置RS初始点个数
    # init_num = 10
    # # # 运行优化过程
    # best_params, best_simulation_result, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num = bo.find(
    #     init_num=init_num)
    # initial_values = best_params[-1]  # 当前最好的参数列表
    # print('stage1_result:', best_simulation_result)
    # # 使用函数筛选
    # # selected_x, selected_y = filter_rows(dbx, dby)
    # # 计算每个参数的SA分数和更新权重
    # *scores, _ = calculate_scores(dbx=dbx, dby=dby, gain_num=gain_num, GBW_num=GBW_num, phase_num=phase_num,
    #                               iter=iter_1, n_neighbors=1, n_repeats=100)
    # # 进行分数排序，从大到小
    # groups = sort_and_group(scores)
    #
    # # 阶段二  取出前四个参数
    # param1, score1 = groups[0][0]
    # param2, score2 = groups[0][1]
    # param3, score3 = groups[0][2]
    # param4, score4 = groups[0][3]
    # print('第二阶段 param1:', param1, 'param2:', param2, 'param3', param3, 'param4', param4)
    #
    # # 参数名称列表
    # param_names = ['cap', 'L1', 'L2', 'L3', 'L4', 'L5', 'r', 'W1', 'W2', 'W3', 'W4', 'W5']
    # # 定义两个范围
    # percentage1 = 0.35
    # percentage2 = 0.2
    # # 获取解绑参数的索引和范围
    # params_indices, param_ranges = get_indices_and_ranges(
    #     param_names,
    #     initial_values,
    #     (param1, percentage1),
    #     (param2, percentage1),
    #     (param3, percentage1),
    #     (param4, percentage1)
    # )
    # # 阶段二迭代次数
    # iter_2 = 100
    # # 创建BayesianOptimization实例
    # bo = BayesianOptimization(
    #     param_ranges=param_ranges,
    #     n=iter_2,  # 迭代次数
    #     simulation_function=OTA_two_simulation_all,
    #     objective=objective,
    #     posterior_transform=posterior_transform,
    #     mode='collect_stage',
    #     best_y=best_simulation_result[-1][1],  # 设置为饱和区搜索出来的最好电流
    #     valid_x=[],
    #     valid_y=[],
    #     last_valid_x=last_x,
    #     last_valid_y=last_y,
    #     last_all_x=[last_x],  # 为了使得后续操作
    #     params_indices=params_indices,
    #     all_x=best_params,
    #     thresholds=thresholds
    # )
    # # 运行优化过程
    # stage_init = 20
    # (best_params, best_simulation_result2, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num, last_all_x,
    #  all_x) = bo.find(stage_init_num=stage_init)
    # initial_values = last_all_x[-1]  # 参数更新
    # # 结果合成
    # best_simulation_result = best_simulation_result + best_simulation_result2
    # print('stage2_result:', best_simulation_result2)
    #
    # # 阶段三  取出中间4个参数
    # param5, score5 = groups[1][0]
    # param6, score6 = groups[1][1]
    # param7, score7 = groups[1][2]
    # param8, score8 = groups[1][3]
    # print('第三阶段 param5:', param5, 'param6:', param6, 'param7:', param7, 'param8', param8)
    #
    # # 获取解绑参数的索引和范围
    # params_indices, param_ranges = get_indices_and_ranges(
    #     param_names,
    #     initial_values,
    #     (param1, percentage2),
    #     (param2, percentage2),
    #     (param3, percentage2),
    #     (param4, percentage2),
    #     (param5, percentage1),
    #     (param6, percentage1),
    #     (param7, percentage1),
    #     (param8, percentage1)
    # )
    #
    # # 阶段三迭代次数
    # iter_3 = 100
    # # 创建BayesianOptimization实例
    # bo = BayesianOptimization(
    #     param_ranges=param_ranges,
    #     n=iter_3,  # 迭代次数
    #     simulation_function=OTA_two_simulation_all,
    #     objective=objective,
    #     posterior_transform=posterior_transform,
    #     mode='collect_stage',
    #     best_y=best_simulation_result[-1][1],  # 设置为饱和区搜索出来的最好电流
    #     valid_x=[],
    #     valid_y=[],
    #     last_valid_x=last_x,
    #     last_valid_y=last_y,
    #     last_all_x=last_all_x,
    #     params_indices=params_indices,
    #     thresholds=thresholds,
    #     all_x=all_x
    # )
    # # 运行优化过程
    # best_params, best_simulation_result3, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num, last_all_x, all_x = (
    #     bo.find(stage_init_num=stage_init))
    # initial_values = last_all_x[-1]  # 参数更新
    # best_simulation_result = best_simulation_result + best_simulation_result3
    # print('stage3_result:', best_simulation_result3)
    #
    # # 阶段四  取出最后四个参数
    # param9, score9 = groups[2][0]
    # param10, score10 = groups[2][1]
    # param11, score11 = groups[2][2]
    # param12, score12 = groups[2][3]
    # print('第四阶段 param9:', param9, 'param10:', param10, 'param11:', param11, 'param12', param12)
    #
    # # 获取解绑参数的索引和范围
    # params_indices, param_ranges = get_indices_and_ranges(
    #     param_names,
    #     initial_values,
    #     (param1, percentage2),
    #     (param2, percentage2),
    #     (param3, percentage2),
    #     (param4, percentage2),
    #     (param5, percentage2),
    #     (param6, percentage2),
    #     (param7, percentage2),
    #     (param8, percentage2),
    #     (param9, percentage1),
    #     (param10, percentage1),
    #     (param11, percentage1),
    #     (param12, percentage1),
    # )
    #
    # # 阶段四迭代次数
    # iter_4 = 100
    # # 创建BayesianOptimization实例
    # bo = BayesianOptimization(
    #     param_ranges=param_ranges,
    #     n=iter_4,  # 迭代次数
    #     simulation_function=OTA_two_simulation_all,
    #     objective=objective,
    #     posterior_transform=posterior_transform,
    #     mode='collect_stage',
    #     best_y=best_simulation_result[-1][1],  # 设置为饱和区搜索出来的最好电流
    #     valid_x=[],
    #     valid_y=[],
    #     last_valid_x=last_x,
    #     last_valid_y=last_y,
    #     last_all_x=last_all_x,
    #     params_indices=params_indices,
    #     thresholds=thresholds,
    #     all_x=all_x,
    #     stage='last'  # 最后阶段，用于加载全解绑gp
    # )
    # # 运行优化过程
    # best_params, best_simulation_result4, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num, last_all_x, all_x = (
    #     bo.find(stage_init_num=stage_init))
    # best_simulation_result = best_simulation_result + best_simulation_result4

    # 实验结果保存路径
    file_path = ('C:/Users/icelab01/Desktop/ZhuohuaLiu_2024_BYA_jiebang/BYA_jiebang/BYA_jiebang/Knowledge_unbinding'
                 '/Experiment/exp_under_different_train_sample/exp_design_1/ourModel_v1/Proposed_OTA_two_matrix_seed_{}.csv').format(
        SEED)
    # 实验结果计算路径
    cal_path = (
        "C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\BYA_jiebang\\BYA_jiebang\\Knowledge_unbinding"
        "\\Experiment\\exp_under_different_train_sample\\exp_design_1\\ourModel_v1\\Proposed_OTA_two_matrix_seed_")
    # 均值方差计算结果保存路径
    to_path = (
        'C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\BYA_jiebang\\BYA_jiebang\\Knowledge_unbinding'
        '\\Experiment\\exp_under_different_train_sample\\exp_design_1_report\\Proposed_OTA_two_matrix_current_mean_var_strand.csv')

    # # 迭代次数列表，用于生成csv数据文件中的迭代次数索引，加20是因为有1（初始点）+19（总共19次RS）
    # iter_times = list(range(1, len(best_simulation_result) + 1))
    # # bo.print_results(best_params, best_simulation_result)
    # bo.save_data(best_simulation_result, iter_times, file_path)

    # 第五个种子时使用
    plot(cal_path, to_path)
