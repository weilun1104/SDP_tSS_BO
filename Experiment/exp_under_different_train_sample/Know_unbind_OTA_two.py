# 专家知识逐渐解绑，验证L2W2W4W5的
# 二级运放全解绑+MI+逐渐解绑
import random
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_V3 import *
from botorch.acquisition.objective import ScalarizedObjective, ScalarizedPosteriorTransform
from botorch.models.transforms.outcome import Standardize
from Model.Point_search.CONBO import BayesianOptimization, plot
from MI_calc import calculate_mutual_information

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


# 设定范围的函数
def set_bounds(value, percentage):
    return (value * (1 - percentage), value * (1 + percentage))


if __name__ == "__main__":
    # 饱和区搜索值(单位微米)
    cap = 3e-12
    L1 = 1e-6
    L2 = 1e-6
    L3 = 1e-6
    L4 = 1e-06
    L5 = 1e-06
    r = 3000
    W1 = 3.4e-6
    W2 = 1.0e-4
    W3 = 12.5e-6
    W4 = 6.8e-6
    W5 = 2.6e-5

    # 设置问题字典
    param_ranges = [
        set_bounds(L2, 0.5),
        set_bounds(W2, 0.5),
        set_bounds(W4, 0.5),
        set_bounds(W5, 0.5)
    ]
    params_indices = [2, 8, 10, 11]
    # 定义一个包含所有约束的字典
    thresholds = {
        'gain': 60,
        'i_multiplier': 1.8,
        'i': 1e-3,
        'phase': 60,
        'gbw': 8e6
    }

    # 用于保存初始满足饱和区的值
    valid_x = dbx_alter = [[3e-12, 1e-6, 1e-6, 1e-6, 1.0e-6, 1.0e-6, 3000, 3.4e-6, 1.0e-4, 12.5e-6, 6.80e-6, 2.6e-5]]
    valid_y = dby_alter = [[78.6, 1.9755e-4, 77.2494, 9.225389e6]]

    # 用于记录上一次满足条件的值
    last_valid_x = [3e-12, 1e-6, 1e-6, 1e-6, 1.0e-6, 1.0e-6, 3000, 3.4e-6, 1.0e-4, 12.5e-6, 6.80e-6, 2.6e-5]
    last_valid_y = [78.6, 1.9755e-4, 77.2494, 9.225389e6]

    # 定义初始满足约束的点的个数
    gain_num = 1
    I_num = 1
    GBW_num = 1
    phase_num = 1

    # 初步设定权重，四个数分别是gian，I,phase,GBW，用于四个参数同一成一个，进行后续的gp训练
    objective = ScalarizedObjective(weights=torch.tensor([0.9, 1.0, 0.5, 0.8]))
    posterior_transform = Standardize(m=1)  # 将后验的均值和方差标准化

    # 初步全解绑BO迭代次数,用于后续MI分析
    iter_1 = 120

    # 创建BayesianOptimization实例
    bo = BayesianOptimization(
        param_ranges=param_ranges,  # 参数范围
        n=iter_1,  # 迭代次数
        simulation_function=OTA_two_simulation_all,  # ngspice
        objective=objective,  # y的参数权重
        posterior_transform=posterior_transform,
        mode='collect_stage',  # 全解绑过程
        best_y=1.9755e-4,  # 设置为饱和区搜索出来的最好电流
        valid_x=valid_x,
        valid_y=valid_y,
        last_valid_x=last_valid_x,
        last_valid_y=last_valid_y,
        last_all_x=valid_x,
        gain_num=gain_num,
        I_num=I_num,
        GBW_num=GBW_num,
        phase_num=phase_num,
        params_indices=params_indices,
        all_x=[last_valid_x],
        thresholds=thresholds
    )

    # iter_times = [x + 1 for x in range(iter+1)]
    # 设置RS初始点个数
    stage_init_num = 20
    # # 运行优化过程
    best_params, best_simulation_result, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num, last_all_x, all_x = bo.find(
        stage_init_num=stage_init_num)
    print('stage1_result:', best_simulation_result)
    L2 = last_all_x[-1][2]
    W2 = last_all_x[-1][8]
    W4 = last_all_x[-1][10]
    W5 = last_all_x[-1][11]

    # 创建新的param_ranges列表
    param_ranges = [
        set_bounds(L2, 0.2),
        set_bounds(L3, 0.5),
        set_bounds(L4, 0.5),
        set_bounds(L5, 0.5),
        set_bounds(W2, 0.2),
        set_bounds(W3, 0.5),
        set_bounds(W4, 0.2),
        set_bounds(W5, 0.2)
    ]
    params_indices = [2, 3, 4, 5, 8, 9, 10, 11]
    # 阶段二迭代次数
    iter_2 = 50
    # 创建BayesianOptimization实例
    bo = BayesianOptimization(
        param_ranges=param_ranges,
        n=iter_2,  # 迭代次数
        simulation_function=OTA_two_simulation_all,
        objective=objective,
        posterior_transform=posterior_transform,
        mode='collect_stage',
        best_y=best_simulation_result[-1][1],  # 设置为饱和区搜索出来的最好电流
        valid_x=[],
        valid_y=[],
        last_valid_x=last_x,
        last_valid_y=last_y,
        last_all_x=last_all_x,  # 为了使得后续操作
        params_indices=params_indices,
        all_x=all_x,
        thresholds=thresholds
    )
    # 运行优化过程
    best_params, best_simulation_result2, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num, last_all_x, all_x = (
        bo.find(stage_init_num=stage_init_num))
    L2 = last_all_x[-1][2]
    L3 = last_all_x[-1][3]
    L4 = last_all_x[-1][4]
    L5 = last_all_x[-1][5]
    W2 = last_all_x[-1][8]
    W3 = last_all_x[-1][9]
    W4 = last_all_x[-1][10]
    W5 = last_all_x[-1][11]
    # 结果合成
    best_simulation_result = best_simulation_result + best_simulation_result2
    print('stage2_result:', best_simulation_result2)

    # 阶段三  取出中间4个参数
    params_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # 创建新的param_ranges列表
    param_ranges = [
        set_bounds(cap, 0.5),
        set_bounds(L1, 0.5),
        set_bounds(L2, 0.2),
        set_bounds(L3, 0.2),
        set_bounds(L4, 0.2),
        set_bounds(L5, 0.2),
        set_bounds(r, 0.5),
        set_bounds(W1, 0.5),
        set_bounds(W2, 0.2),
        set_bounds(W3, 0.2),
        set_bounds(W4, 0.2),
        set_bounds(W5, 0.2)
    ]

    # 阶段三迭代次数
    iter_3 = 50
    # 创建BayesianOptimization实例
    bo = BayesianOptimization(
        param_ranges=param_ranges,
        n=iter_3,  # 迭代次数
        simulation_function=OTA_two_simulation_all,
        objective=objective,
        posterior_transform=posterior_transform,
        mode='collect_stage',
        best_y=best_simulation_result[-1][1],  # 设置为饱和区搜索出来的最好电流
        valid_x=[],
        valid_y=[],
        last_valid_x=last_x,
        last_valid_y=last_y,
        last_all_x=last_all_x,
        params_indices=params_indices,
        all_x=all_x,
        thresholds=thresholds
    )
    # 运行优化过程
    best_params, best_simulation_result3, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num, last_all_x, all_x = (
        bo.find(stage_init_num=stage_init_num))
    best_simulation_result = best_simulation_result + best_simulation_result3

    # 设置数据保存路径
    file_path = ('C:/Users/icelab01/Desktop/ZhuohuaLiu_2024_BYA_jiebang/BYA_jiebang/BYA_jiebang/Knowledge_unbinding'
                 '/Experiment/exp_under_different_train_sample/exp_design_1/baseline_Model_v1/Know_unbind'
                 '/Know_unbind_OTA_two_seed_{}.csv').format(SEED)

    # 设置计算均值，方差路径
    cal_path = ("C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\BYA_jiebang\\BYA_jiebang"
                "\\Knowledge_unbinding\\Experiment\\exp_under_different_train_sample\\exp_design_1\\baseline_Model_v1"
                "\\Know_unbind\\Know_unbind_OTA_two_seed_")

    # 设置保存路径
    to_path =('C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\BYA_jiebang\\BYA_jiebang\\Knowledge_unbinding'
              '\\Experiment\\exp_under_different_train_sample\\exp_design_1_report'
              '\\Know_unbind_OTA_two_current_mean_var_strand.csv')

    # 迭代次数列表，用于生成csv数据文件中的迭代次数索引
    iter_times = [x + 1 for x in range(iter_1 + iter_2 + iter_3 + 1 + stage_init_num*3)]
    # bo.print_results(best_params, best_simulation_result)
    bo.save_data(best_simulation_result, iter_times, file_path)

    # 第五个种子时使用
    # plot(cal_path, to_path)


