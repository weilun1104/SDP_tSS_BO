
# 设定范围的函数
def set_bounds(value):
    return [value * (1 - 0.5), value * (1 + 0.5)]

def init_OTA_two():
    # 饱和区搜索值(单位微米)
    cap = 4.66e-11
    L1 = 1.52e-6
    L2 = 4.32e-7
    L3 = 1.33e-6
    L4 = 1e-06
    L5 = 1e-06
    r = 9056
    W1 = 1.9608e-5
    W2 = 1.944e-5
    W3 = 8.5785e-5
    W4 = 2.58e-5
    W5 = 9e-6   

    # 电容
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

    # 设置问题字典
    param_ranges = [
        (cap_min, cap_max),
        (L1_min, L1_max),
        (L2_min, L2_max),
        (L3_min, L3_max),
        (L4_min, L4_max),
        (L5_min, L5_max),
        (R_min, R_max),
        (W1_min, W1_max),
        (W2_min, W2_max),
        (W3_min, W3_max),
        (W4_min, W4_max),
        (W5_min, W5_max)
    ]
    # 定义一个包含所有约束的字典
    thresholds = {
        'gain': 60,
        'i_multiplier': 1.8,
        'i': 1e-3,
        'phase': 60,
        'gbw': 4e6
    }

    # 用于保存初始满足饱和区的值
    valid_x = dbx_alter = [[4.66e-11, 1.52e-6, 4.32e-7, 1.33e-6, 1e-6, 1e-6, 9056, 1.9608e-5, 1.944e-5, 8.5785e-5, 2.58e-5, 9e-6]]
    valid_y = dby_alter = [[77.64, 2.345e-4, 67.2136, 38e6]]

    # 用于记录上一次满足条件的值
    last_valid_x = [4.66e-11, 1.52e-6, 4.32e-7, 1.33e-6, 1e-6, 1e-6, 9056, 1.9608e-5, 1.944e-5, 8.5785e-5, 2.58e-5, 9e-6]
    last_valid_y = [77.64, 2.345e-4, 67.2136, 38e6]


    return param_ranges, thresholds, valid_x, valid_y, dbx_alter, dby_alter, last_valid_x, last_valid_y

