# 三级全解绑
import torch
import numpy as np
import random
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_V4 import *
from botorch.acquisition.objective import ScalarizedObjective, ScalarizedPosteriorTransform
from botorch.models.transforms.outcome import Standardize
from Model.Point_search.CONBO import BayesianOptimization, plot, save_data
# from BO_test import BayesianOptimization, plot
from config_OTA_three import init_OTA_three

# 将参数都设置维double类型
torch.set_default_dtype(torch.double)

# 种子设置
SEED = 1
random.seed(SEED)
np.random.seed(SEED)  # 设置 NumPy 的种子
torch.manual_seed(SEED)


# param_ranges, thresholds, valid_x, valid_y, dbx_alter, dby_alter, last_valid_x, last_valid_y = init_OTA_three()
#
#
# # 定义初始满足约束的点的个数
# gain_num = 1
# I_num = 1
# GBW_num = 1
# phase_num = 1
#
# # 初步设定权重，四个数分别是gian，I,phase,GBW，用于四个参数同一成一个，进行后续的gp训练
# objective = ScalarizedObjective(weights=torch.tensor([0.5, 1.0, 0.9, 0.5]))
# posterior_transform = Standardize(m=1)  # 将后验的均值和方差标准化
#
# # 初步全解绑BO迭代次数,用于后续MI分析
# iter_1 = 379
# # iter_1 = 50
#
# # 创建BayesianOptimization实例
# bo = BayesianOptimization(
#     param_ranges=param_ranges,  # 参数范围
#     n=iter_1,  # 迭代次数
#     simulation_function=OTA_three_simulation_all,  # ngspice
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
#     thresholds=thresholds
# )
#
# # iter_times = [x + 1 for x in range(iter+1)]
# # 设置RS初始点个数
# init_num = 20

file_path = ('C:/Users/icelab01/Desktop/ZhuohuaLiu_2024_BYA_jiebang/BYA_jiebang/BYA_jiebang/Knowledge_unbinding'
            '/Experiment/exp_under_different_train_sample/exp_design_1/baseline_Model_v1/Unbind_all/Unbind_all_OTA_three_seed_{}.csv').format(SEED)
cal_path = ("C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\BYA_jiebang\\BYA_jiebang\\Knowledge_unbinding"
                "\\Experiment\\exp_under_different_train_sample\\exp_design_1\\baseline_Model_v1\\Unbind_all\\Unbind_all_OTA_three_seed_")
to_path =('C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\BYA_jiebang\\BYA_jiebang\\Knowledge_unbinding'
              '\\Experiment\\exp_under_different_train_sample\\exp_design_1_report\\Unbind_all_OTA_three_current_mean_var_strand.csv')

# # # 运行优化过程
# best_params, best_simulation_result, last_x, last_y, dbx, dby, gain_num, I_num, GBW_num, phase_num = bo.find(
#     init_num=init_num)
# iter_times = list(range(1, len(best_simulation_result) + 1))
# bo.print_results(best_params, best_simulation_result)
# save_data(best_simulation_result, iter_times, file_path)
# #
# # 第五个种子时使用
plot(cal_path, to_path, [1, 2, 3, 4, 5])

