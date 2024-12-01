# 二级sobol算法进行SA分析
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_V3 import *
import random
from Sens_analysis import sobol_sensitivity_analysis, morris_sensitivity_analysis

# example:
SEED = 1
random.seed(SEED)
np.random.seed(SEED)  # 设置 NumPy 的种子
torch.manual_seed(SEED)

# 已知参数值
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


# 设定范围的函数
def set_bounds(value):
    return [value * (1 - 0.5), value * (1 + 0.5)]


cap_min, cap_max = set_bounds(cap)

# 管子一
L1_min, L1_max = set_bounds(L1)
W1_min, W1_max = set_bounds(W1)

# 管子二
L2_min, L2_max = set_bounds(L2)
W2_min, W2_max = set_bounds(W2)

# 管子三
L3_min, L3_max = set_bounds(L3)
W3_min, W3_max = set_bounds(W3)

# 管子四
L4_min, L4_max = set_bounds(L4)
W4_min, W4_max = set_bounds(W4)

# 管子五
L5_min, L5_max = set_bounds(L5)
W5_min, W5_max = set_bounds(W5)

R_min, R_max = set_bounds(r)

# 将微米转换为米的单位转换因子
# um_to_m = 1e-6

# 更新问题字典
problem = {
    'num_vars': 12,
    'names': ['cap', 'l1', 'l2', 'l3', 'l4', 'l5', 'r', 'w1', 'w2', 'w3', 'w4', 'w5'],
    'bounds': [
        [cap_min, cap_max],
        [L1_min, L1_max],
        [L2_min, L2_max],
        [L3_min, L3_max],
        [L4_min, L4_max],
        [L5_min, L5_max],
        [R_min, R_max],
        [W1_min, W1_max],
        [W2_min, W2_max],
        [W3_min, W3_max],
        [W4_min, W4_max],
        [W5_min, W5_max]
    ]
}


# 此参数sobol和
num_samples = 32

num_levels = 4   # 网格中的级别数

# 指定分析的性能指标编号（对应模型输出的索引）
target_index = 3

# morris 总仿真次数 = 轨迹数量 (N) × (变量数量 (num_vars) + 1)
# Si = morris_sensitivity_analysis(OTA_two_simulation_all, problem, num_samples, target_index, num_levels)
#
# sobol 总仿真次数 = (2 + num_vars * 2) × N
first_order, total_order = sobol_sensitivity_analysis(OTA_two_simulation_all, problem, num_samples, target_index)


# sobol方法的
print("一阶Sobol指数:", first_order)
print("总Sobol指数: ", total_order)


# 打印morris分析结果
# print("Morris Method Analysis Results:")
# print("mu (or mean):", Si['mu'])
# print("sigma:", Si['sigma'])
# print("mu_star:", Si['mu_star'])