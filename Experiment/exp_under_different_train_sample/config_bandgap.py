
# 设定范围的函数
def set_bounds(value):
    return [value * (1 - 0.5), value * (1 + 0.5)]


def init_bandgap():
    # 饱和区搜索值(单位微米)
    L1 = 7.64e-07
    L2 = 4.83e-06
    L3 = 4.16e-06
    L4 = 3.66e-06
    L5 = 5.29e-07
    L6 = 3.06e-06
    L7 = 2.34e-06
    L8 = 2.33e-06
    L9 = 3.06e-06
    R1 = 2559018
    R2 = 261904
    W1 = 1e-05
    W2 = 7.45e-05
    W3 = 6.37e-05
    W4 = 1.46e-05
    W5 = 1.75e-06
    W6 = 2.43e-05
    W7 = 7.93e-05
    W8 = 2.10e-05
    W9 = 2.79e-05


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

    # 管子六
    L6_min, L6_max = set_bounds(L6)
    W6_min, W6_max = set_bounds(W6)

    # 管子七
    L7_min, L7_max = set_bounds(L7)
    W7_min, W7_max = set_bounds(W7)

    # 管子八
    L8_min, L8_max = set_bounds(L8)
    W8_min, W8_max = set_bounds(W8)

    # 管子九
    L9_min, L9_max = set_bounds(L9)
    W9_min, W9_max = set_bounds(W9)
    
    # R1 和 R2
    R1_min, R1_max = set_bounds(R1)
    R2_min, R2_max = set_bounds(R2)

    # 设置问题字典
    param_ranges = [
        (L1_min, L1_max),
        (L2_min, L2_max),
        (L3_min, L3_max),
        (L4_min, L4_max),
        (L5_min, L5_max),
        (L6_min, L6_max),
        (L7_min, L7_max),
        (L8_min, L8_max),
        (L9_min, L9_max),
        (R1_min, R1_max),
        (R2_min, R2_max),
        (W1_min, W1_max),
        (W2_min, W2_max),
        (W3_min, W3_max),
        (W4_min, W4_max),
        (W5_min, W5_max),
        (W6_min, W6_max),
        (W7_min, W7_max),
        (W8_min, W8_max),
        (W9_min, W9_max),
    ]
    # 定义一个包含所有约束的字典
    thresholds = {
        'ppm': 200,
        'dc_current': 3.3e-5,
        'psrr': 60
    }

    # 用于保存初始满足饱和区的值
    valid_x = dbx_alter = [[7.64e-07, 4.83e-06, 4.16e-06, 3.66e-06, 5.29e-07, 3.06e-06,
                     2.34e-06, 2.33e-06, 3.06e-06, 2559018, 261904, 1e-05,
                     7.45e-05, 6.37e-05, 1.46e-05, 1.75e-06, 2.43e-05, 7.93e-05,
                     2.10e-05, 2.79e-05]]
    valid_y = dby_alter = [[39.3, 3.98e-5, 69.64]]

    # 用于记录上一次满足条件的值
    last_valid_x = [7.64e-07, 4.83e-06, 4.16e-06, 3.66e-06, 5.29e-07, 3.06e-06,
                     2.34e-06, 2.33e-06, 3.06e-06, 2559018, 261904, 1e-05,
                     7.45e-05, 6.37e-05, 1.46e-05, 1.75e-06, 2.43e-05, 7.93e-05,
                     2.10e-05, 2.79e-05]
    last_valid_y = [39.3, 3.98e-5, 69.64]

    return param_ranges, thresholds, valid_x, valid_y, dbx_alter, dby_alter, last_valid_x, last_valid_y


