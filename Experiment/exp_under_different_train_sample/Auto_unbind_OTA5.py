# 五管sobol算法进行SA分析
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_V2 import *
import random
from Sens_analysis import sobol_sensitivity_analysis, morris_sensitivity_analysis

# example:
SEED = 1
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

# 管子一
L1_min, L1_max = L1 * (1 - 0.5), L1 * (1 + 0.5)
W1_min, W1_max = W1 * (1 - 0.5), W1 * (1 + 0.5)

# 管子二
L2_min, L2_max = L2 * (1 - 0.5), L2* (1 + 0.5)
W2_min, W2_max = W2 * (1 - 0.5), W2 * (1 + 0.5)

# 管子三
L3_min, L3_max = L3 * (1 - 0.5), L3 * (1 + 0.5)
W3_min, W3_max = W3 * (1 - 0.5), W3* (1 + 0.5)

# 将微米转换为米
um_to_m = 1e-6


problem = {
    'num_vars': 6,
    'names': ['l1', 'l2', 'l3', 'w1', 'w2', 'w3'],
    'bounds': [
        [L1_min * um_to_m, L1_max * um_to_m],
        [L2_min * um_to_m, L2_max * um_to_m],
        [L3_min * um_to_m, L3_max * um_to_m],
        [W1_min * um_to_m, W1_max * um_to_m],
        [W2_min * um_to_m, W2_max * um_to_m],
        [W3_min * um_to_m, W3_max * um_to_m]
    ]
}

# 此参数sobol和
num_samples = 8

num_levels = 4   # 网格中的级别数

# 指定分析的性能指标编号（对应模型输出的索引）
target_index = 1

# morris 总仿真次数 = 轨迹数量 (N) × (变量数量 (num_vars) + 1)
# Si = morris_sensitivity_analysis(OTA5_simulation_all, problem, num_samples, target_index, num_levels)

# sobol 总仿真次数 = (2 + num_vars * 2) × N
first_order, total_order = sobol_sensitivity_analysis(OTA5_simulation_all, problem, num_samples, target_index)


# sobol方法的
print("一阶Sobol指数:", first_order)
print("总Sobol指数: ", total_order)


# 打印morris分析结果
# print("Morris Method Analysis Results:")
# print("mu (or mean):", Si['mu'])
# print("sigma:", Si['sigma'])
# print("mu_star:", Si['mu_star'])