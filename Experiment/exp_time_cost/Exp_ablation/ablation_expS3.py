# 专家知识逐渐解绑消融实验三 W3,L3---->L1,L2---->W1---->W2

import torch
import numpy as np
import random
from Simulation.Data.lyngspice_master.lyngspice_master.examples.ablation_S3 import *
from botorch.acquisition.objective import ScalarizedObjective
from botorch.models.transforms.outcome import Standardize
from Model.Point_search.CONBO import BayesianOptimization,plot


# example:
SEED = 5
random.seed(SEED)
np.random.seed(SEED)  # 设置 NumPy 的种子
torch.manual_seed(SEED)

# 饱和区搜索值(单位微米)
L1 = 2.27
L2 = 4.57
L3 = 1
W1 = 117.68
W2 = 105.29
W3 = 46.8

# 将微米转换为米
um_to_m = 1e-6

# 四个性能指标权重
objective = ScalarizedObjective(weights=torch.tensor([1.0, 1.0, 1.0, 1.0]))
posterior_transform = Standardize(m=1)

# 阶段一 W3,L3上下50%
# 管子三
L3_min, L3_max = L3 * (1 - 0.5), L3 * (1 + 0.5)
W3_min, W3_max = W3 * (1 - 0.5), W3* (1 + 0.5)

# 创建param_ranges列表
param_ranges = [
    (L3_min * um_to_m, L3_max * um_to_m),
    (W3_min * um_to_m, W3_max * um_to_m),
]


# 迭代次数,这里需要减去1次，用于初始化
iter_1 = 19

# 用于保存初始满足饱和区的值
valid_x = [[2.27e-6,4.57e-6,1e-6,1,1.1768e-4,1.0529e-4,4.68e-5]]
valid_y = [[44.45, 3.6e-4, 62.28, 43.17e6]]

# 用于记录上一次满足条件的值
last_valid_x = [2.27e-6,4.57e-6,1e-6,1,1.1768e-4,1.0529e-4,4.68e-5]
last_valid_y = [44.45, 3.6e-4, 62.28, 43.17e6]


# 创建BayesianOptimization实例
bo = BayesianOptimization(
    param_ranges=param_ranges,
    n=iter_1,                                   # 迭代次数
    simulation_function=OTA5_simulation_one,    # 使用的ngspice接口函数
    objective=objective,
    posterior_transform=posterior_transform,
    mode='collect_stage',                       # 设置模式，这里为分段解绑模式
    best_y=3.6e-4,                              # 设置为饱和区搜索出来的最好电流
    valid_x=valid_x,                            # 将前面定义的初始输入进去
    valid_y=valid_y,
    last_valid_x=last_valid_x,
    last_valid_y=last_valid_y,
    stage=1,                                    # 设置为阶段1
)

# 运行优化过程
best_params, best_simulation_result, last_x, last_y, dbx, dby = bo.find()
# L3,W3进行更新
L3 = best_params[-1][0]
W3 = best_params[-1][1]

# 阶段二 L3,W3上下20%,L1,L2上下50%
# 管子一
L1_min, L1_max = L1 * (1 - 0.5), L1 * (1 + 0.5)

# 管子二
L2_min, L2_max = L2 * (1 - 0.5), L2 * (1 + 0.5)

# 管子三
L3_min, L3_max = L3 * (1 - 0.2), L3 * (1 + 0.2)
W3_min, W3_max = W3 * (1 - 0.2), W3 * (1 + 0.2)

# 创建param_ranges列表
param_ranges = [
    (L1_min * um_to_m, L1_max * um_to_m),
    (L2_min * um_to_m, L2_max * um_to_m),
    (L3_min, L3_max),
    (W3_min, W3_max),
]


# 迭代次数
iter_2 = 20

# 创建BayesianOptimization实例
bo = BayesianOptimization(
    param_ranges=param_ranges,
    n=iter_2,                                       # 迭代次数
    simulation_function=OTA5_simulation_two,
    objective=objective,
    posterior_transform=posterior_transform,
    mode='collect_stage',
    best_y=best_simulation_result[-1][1],           # 设置为饱和区搜索出来的最好电流
    valid_x=[],                                     # 此处因为输入参数个数变了，故不可沿用之前的x、y、gp,只能从头开始训练
    valid_y=[],
    last_valid_x=last_x,
    last_valid_y=last_y,
    stage=2,
)

# 运行优化过程
best_params, best_simulation_result2, last_x, last_y, dbx, dby = bo.find()
# 数据保存
best_simulation_result = best_simulation_result+best_simulation_result2
# L1,L3,W3进行更新
L1 = best_params[-1][0]
L2 = best_params[-1][1]
L3 = best_params[-1][2]
W3 = best_params[-1][3]

# 阶段三 L3,W3,L1,L2上下20%,W1上下50%
# 管子一
L1_min, L1_max = L1 * (1 - 0.2), L1 * (1 + 0.2)
W1_min, W1_max = W1 * (1 - 0.5), W1 * (1 + 0.5)

# 管子二
L2_min, L2_max = L1 * (1 - 0.2), L1 * (1 + 0.2)

# 管子三
L3_min, L3_max = L3 * (1 - 0.2), L3 * (1 + 0.2)
W3_min, W3_max = W3 * (1 - 0.2), W3 * (1 + 0.2)



# 创建param_ranges列表
param_ranges = [
    (L1_min, L1_max),
    (L2_min, L2_max),
    (L3_min, L3_max),
    (W1_min * um_to_m, W1_max * um_to_m),
    (W3_min, W3_max),
]


# 迭代次数
iter_3 = 20

bo = BayesianOptimization(
    param_ranges=param_ranges,
    n=iter_3,           # 迭代次数
    simulation_function=OTA5_simulation_three,
    objective=objective,
    posterior_transform=posterior_transform,
    mode='collect_stage',
    best_y=best_simulation_result[-1][1],          # 设置为饱和区搜索出来的最好电流
    valid_x=[],
    valid_y=[],
    last_valid_x=last_x,
    last_valid_y=last_y,
    stage=3,
)

# 运行优化过程
best_params, best_simulation_result3, last_x, last_y, dbx, dby = bo.find()
best_simulation_result = best_simulation_result+best_simulation_result3
# L1,L2,L3,W1,W3进行更新
L1 = best_params[-1][0]
L2 = best_params[-1][1]
L3 = best_params[-1][2]
W1 = best_params[-1][3]
W3 = best_params[-1][4]


# 阶段四 L1,L2,L3,W1,W3上下20%, W2上下50%
# 管子一
L1_min, L1_max = L1 * (1 - 0.2), L1 * (1 + 0.2)
W1_min, W1_max = W1 * (1 - 0.2), W1 * (1 + 0.2)

# 管子二
L2_min, L2_max = L2 * (1 - 0.2), L2 * (1 + 0.2)
W2_min, W2_max = W2 * (1 - 0.5), W2 * (1 + 0.5)

# 管子三
L3_min, L3_max = L3 * (1 - 0.2), L3 * (1 + 0.2)
W3_min, W3_max = W3 * (1 - 0.2), W3 * (1 + 0.2)


# 现在创建param_ranges列表
param_ranges = [
    (L1_min, L1_max),
    (L2_min, L2_max),
    (L3_min, L3_max),
    (W1_min, W1_max),
    (W2_min * um_to_m, W2_max * um_to_m),
    (W3_min, W3_max),
]


# 迭代次数
iter_4 = 20

# 创建BayesianOptimization实例
bo = BayesianOptimization(
    param_ranges=param_ranges,
    n=iter_4,           # 迭代次数
    simulation_function=OTA5_simulation_all,
    objective=objective,
    posterior_transform=posterior_transform,
    mode='collect_stage',
    best_y=best_simulation_result[-1][1],          # 设置为饱和区搜索出来的最好电流
    valid_x=[],
    valid_y=[],
    last_valid_x=last_x,
    last_valid_y=last_y,
    stage=4,
)

# 设置数据保存路径
file_path = ('C:/Users/icelab01/Desktop/ZhuohuaLiu_2024_BYA_jiebang/BYA_jiebang/BYA_jiebang'
             '/Knowledge_unbinding/Experiment/exp_time_cost/Exp_ablation/ResultData/Ablation_S3_seed_{}.csv').format(SEED)

# 设置计算均值，方差路径
cal_path = ("C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\BYA_jiebang\\BYA_jiebang\\Knowledge_unbinding"
            "\\Experiment\\exp_time_cost\\Exp_ablation\\ResultData\\Ablation_S3_seed_")

# 设置保存路径
to_path =('C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\BYA_jiebang\\BYA_jiebang\\Knowledge_unbinding'
          '\\Experiment\\exp_time_cost\\Exp_ablation\\FigureGen\\Ablation_S3_current_mean_var_strand.csv')

# 迭代次数列表，用于生成csv数据文件中的迭代次数索引
iter_times = [x + 1 for x in range(iter_1+iter_2+iter_3+iter_4 + 1)]
# 运行优化过程
best_params, best_simulation_result4, last_x, last_y, dbx, dby = bo.find()
best_simulation_result = best_simulation_result+best_simulation_result4

# 数据打印
bo.print_results(best_params, best_simulation_result)
# 数据保存
bo.save_data(best_simulation_result, iter_times,file_path)

# 在第五个种子使用
# plot(cal_path, to_path)

