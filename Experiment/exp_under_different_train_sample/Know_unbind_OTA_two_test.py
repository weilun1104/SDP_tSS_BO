# 二级运放全解绑+MI+逐渐解绑
# 专家知识逐渐解绑，验证L4L5W4W5的
import random
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_V3 import *
from botorch.acquisition.objective import ScalarizedObjective, ScalarizedPosteriorTransform
from botorch.models.transforms.outcome import Standardize
from Model.Point_search.CONBO import BayesianOptimization, plot
from MI_calc import calculate_mutual_information

# 将参数都设置维double类型
torch.set_default_dtype(torch.double)

# 设置实验种子
SEED = 5
random.seed(SEED)
np.random.seed(SEED)  # 设置 NumPy 的种子
torch.manual_seed(SEED)

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


# 定义参数范围设置函数
def set_param_ranges(param_initial_value, percentage=0.35):
    param_min = param_initial_value * (1 - percentage)
    param_max = param_initial_value * (1 + percentage)
    return param_min, param_max


# 设定范围的函数
def set_bounds(value, percentage):
    return (value * (1 - percentage), value * (1 + percentage))


if __name__ == "__main__":
    # 饱和区搜索值(单位微米)
    cap = 2.4e-11
    L1 = 6.02e-6
    L2 = 3.38e-6
    L3 = 1.22e-6
    L4 = 1e-06
    L5 = 1e-06
    r = 5339
    W1 = 1.806e-5
    W2 = 6.02e-4
    W3 = 1.83e-5
    W4 = 6e-6
    W5 = 3.564e-5

    # 定义一个包含所有约束的字典
    thresholds = {
        'gain': 60,
        'i_multiplier': 1.8,
        'i': 1e-3,
        'phase': 60,
        'gbw': 4e6
    }

    # 用于保存初始满足饱和区的值
    valid_x = dbx_alter = [[2.4e-11, 6.02e-6, 3.38e-6, 1.22e-6, 1.0e-6, 1.0e-6, 5339, 1.806e-5, 6.02e-4, 1.83e-5, 6e-6, 3.564e-5]]
    valid_y = dby_alter = [[83.16, 2.89e-4, 132.974, 9.156e6]]

    # 用于记录上一次满足条件的值
    last_valid_x = [2.4e-11, 6.02e-6, 3.38e-6, 1.22e-6, 1.0e-6, 1.0e-6, 5339, 1.806e-5, 6.02e-4, 1.83e-5, 6e-6,
                    3.564e-5]
    last_valid_y = [83.16, 2.89e-4, 132.974, 9.156e6]

    # 定义初始满足约束的点的个数
    gain_num = 1
    I_num = 1
    GBW_num = 1
    phase_num = 1

    # 初步设定权重，四个数分别是gian，I,phase,GBW，用于四个参数同一成一个，进行后续的gp训练
    objective = ScalarizedObjective(weights=torch.tensor([0.9, 1.0, 0.5, 0.8]))
    posterior_transform = Standardize(m=1)  # 将后验的均值和方差标准化

    # 设置问题字典
    param_ranges = [
        set_bounds(cap, 0.5),
        set_bounds(L1, 0.5),
        set_bounds(L2, 0.5),
        set_bounds(L3, 0.5),
        set_bounds(L4, 0.5),
        set_bounds(L5, 0.5),
        set_bounds(r, 0.5),
        set_bounds(W1, 0.5),
        set_bounds(W2, 0.5),
        set_bounds(W3, 0.5),
        set_bounds(W4, 0.5),
        set_bounds(W5, 0.5)
    ]

    iter_1 = 40
    # 创建BayesianOptimization实例
    bo = BayesianOptimization(
        param_ranges=param_ranges,  # 参数范围
        n=iter_1,  # 迭代次数
        simulation_function=OTA_two_simulation_all,  # ngspice
        objective=objective,  # y的参数权重
        posterior_transform=posterior_transform,
        mode='collect_all',  # 全解绑过程
        best_y=2.89e-4,  # 设置为饱和区搜索出来的最好电流
        dbx_alter=dbx_alter,
        dby_alter=dby_alter,
        valid_x=valid_x,
        valid_y=valid_y,
        last_valid_x=last_valid_x,
        last_valid_y=last_valid_y,
        gain_num=gain_num,
        I_num=I_num,
        GBW_num=GBW_num,
        phase_num=phase_num,
        thresholds=thresholds
    )
    # 设置RS初始点个数
    init_num = 10
    # # 运行优化过程
    best_params, best_simulation_result, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num = bo.find(
        init_num=init_num)
    print('stage1_result:', best_simulation_result)
    # selected_x, selected_y = filter_rows(dbx, dby)
    cap = best_params[-1][0]
    L1 = best_params[-1][1]
    L2 = best_params[-1][2]
    L3 = best_params[-1][3]
    L4 = best_params[-1][4]
    L5 = best_params[-1][5]
    r = best_params[-1][6]
    W1 = best_params[-1][7]
    W2 = best_params[-1][8]
    W3 = best_params[-1][9]
    W4 = best_params[-1][10]
    W5 = best_params[-1][11]
    # 设置问题字典
    param_ranges = [
        set_bounds(L4, 0.5),
        set_bounds(L5, 0.5),
        set_bounds(W4, 0.5),
        set_bounds(W5, 0.5)
    ]
    params_indices = [4, 5, 10, 11]

    # 初步全解绑BO迭代次数,用于后续MI分析
    iter_2 = 100

    # 创建BayesianOptimization实例
    bo = BayesianOptimization(
        param_ranges=param_ranges,  # 参数范围
        n=iter_2,  # 迭代次数
        simulation_function=OTA_two_simulation_all,  # ngspice
        objective=objective,  # y的参数权重
        posterior_transform=posterior_transform,
        mode='collect_stage',  # 全解绑过程
        best_y=best_simulation_result[-1][1],  # 设置为饱和区搜索出来的最好电流
        valid_x=[],
        valid_y=[],
        last_valid_x=last_x,
        last_valid_y=last_y,
        last_all_x=[last_x],
        gain_num=gain_num,
        I_num=I_num,
        GBW_num=GBW_num,
        phase_num=phase_num,
        params_indices=params_indices,
        all_x=best_params,
        thresholds=thresholds
    )


    # iter_times = [x + 1 for x in range(iter+1)]
    # 设置RS初始点个数
    stage_init_num = 20
    # # 运行优化过程
    best_params, best_simulation_result2, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num, last_all_x, all_x = bo.find(
        stage_init_num=stage_init_num)
    # selected_x, selected_y = filter_rows(dbx, dby)
    best_simulation_result = best_simulation_result + best_simulation_result2
    L4 = last_all_x[-1][4]
    L5 = last_all_x[-1][5]
    W4 = last_all_x[-1][10]
    W5 = last_all_x[-1][11]

    # 创建新的param_ranges列表
    param_ranges = [
        set_bounds(L2, 0.5),
        set_bounds(L3, 0.5),
        set_bounds(L4, 0.2),
        set_bounds(L5, 0.2),
        set_bounds(W2, 0.5),
        set_bounds(W3, 0.5),
        set_bounds(W4, 0.2),
        set_bounds(W5, 0.2)
    ]
    params_indices = [2, 3, 4, 5, 8, 9, 10, 11]
    # 阶段二迭代次数
    iter_3 = 100
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
        last_all_x=last_all_x,  # 为了使得后续操作
        params_indices=params_indices,
        thresholds=thresholds,
        all_x=all_x
    )
    # 运行优化过程
    best_params, best_simulation_result3, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num, last_all_x, all_x = (
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
    best_simulation_result = best_simulation_result + best_simulation_result3
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
    iter_4 = 100
    # 创建BayesianOptimization实例
    bo = BayesianOptimization(
        param_ranges=param_ranges,
        n=iter_4,  # 迭代次数
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
        thresholds=thresholds,
        all_x=all_x
    )
    # 运行优化过程
    best_params, best_simulation_result4, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num, last_all_x, all_x = (
        bo.find(stage_init_num=stage_init_num))
    best_simulation_result = best_simulation_result + best_simulation_result4

    # 设置数据保存路径
    file_path = ('C:/Users/icelab01/Desktop/ZhuohuaLiu_2024_BYA_jiebang/BYA_jiebang/BYA_jiebang/Knowledge_unbinding'
                 '/Experiment/exp_under_different_train_sample/exp_design_1/baseline_Model_v1/Know_unbind'
                 '/Know_unbind_OTA_two_test_seed_{}.csv').format(SEED)

    # 设置计算均值，方差路径
    cal_path = ("C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\BYA_jiebang\\BYA_jiebang"
                "\\Knowledge_unbinding\\Experiment\\exp_under_different_train_sample\\exp_design_1\\baseline_Model_v1"
                "\\Know_unbind\\Know_unbind_OTA_two_test_seed_")

    # 设置保存路径
    to_path =('C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\BYA_jiebang\\BYA_jiebang\\Knowledge_unbinding'
              '\\Experiment\\exp_under_different_train_sample\\exp_design_1_report'
              '\\Know_unbind_OTA_two_test_current_mean_var_strand.csv')

    # 迭代次数列表，用于生成csv数据文件中的迭代次数索引
    iter_times = [x + 1 for x in range(iter_1 + iter_2 + iter_3 + iter_4 + 1 + init_num + stage_init_num*3)]
    # bo.print_results(best_params, best_simulation_result)
    bo.save_data(best_simulation_result, iter_times, file_path)

    # 第五个种子时使用
    plot(cal_path, to_path)


