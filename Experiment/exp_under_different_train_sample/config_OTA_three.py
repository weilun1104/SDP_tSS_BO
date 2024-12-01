
# 设定范围的函数
def set_bounds(value):
    return [value * (1 - 0.5), value * (1 + 0.5)]


def init_OTA_three():
    # 饱和区搜索值(单位微米)
    c1 = 4.5e-12
    c2 = 5.2768e-12
    L1 = 7.7044e-7
    L2 = 8.92e-7
    L3 = 1.391e-6
    L4 = 1.67251e-6
    L5 = 1e-6
    L6 = 1e-6
    L7 = 1e-6
    L8 = 1e-6
    W1 = 7.65e-6
    W2 = 1.771512e-6
    W3 = 7.12e-5
    W4 = 3.179e-5
    W5 = 3.972e-6
    W6 = 1.0239e-5
    W7 = 5.1195e-5
    W8 = 9.5e-5


    # 电容
    c1_min, c1_max = set_bounds(c1)
    c2_min, c2_max = set_bounds(c2)

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

    # 设置问题字典
    param_ranges = [
        (c1_min, c1_max),
        (c2_min, c2_max),
        (L1_min, L1_max),
        (L2_min, L2_max),
        (L3_min, L3_max),
        (L4_min, L4_max),
        (L5_min, L5_max),
        (L6_min, L6_max),
        (L7_min, L7_max),
        (L8_min, L8_max),
        (W1_min, W1_max),
        (W2_min, W2_max),
        (W3_min, W3_max),
        (W4_min, W4_max),
        (W5_min, W5_max),
        (W6_min, W6_max),
        (W7_min, W7_max),
        (W8_min, W8_max),

    ]
    # 定义一个包含所有约束的字典
    thresholds = {
        'gain': 80,
        'i_multiplier': 1.8,
        'i': 1e-3,
        'phase': 60,
        'gbw': 2e6
    }

    # 用于保存初始满足饱和区的值
    valid_x = dbx_alter = [[4.5e-12, 5.2768e-12, 7.7044e-7, 8.92e-7, 1.391e-6, 1.67251e-6, 1e-6, 1e-6, 1e-6, 1e-6, 7.65e-6, 1.771512e-6, 7.12e-5, 3.179e-5,
                            3.972e-6, 1.0239e-5, 5.1195e-5, 9.5e-5]]
    valid_y = dby_alter = [[117.7462, 2.968e-4, 72, 4.494e6]]

    # 用于记录上一次满足条件的值
    last_valid_x = [4.5e-12, 5.2768e-12, 7.7044e-7, 8.92e-7, 1.391e-6, 1.67251e-6, 1e-6, 1e-6, 1e-6, 1e-6, 7.65e-6, 1.771512e-6, 7.12e-5, 3.179e-5,
                    3.972e-6, 1.0239e-5, 5.1195e-5, 9.5e-5]
    last_valid_y = [117.7462, 2.968e-4, 72, 4.494e6]

    return param_ranges, thresholds, valid_x, valid_y, dbx_alter, dby_alter, last_valid_x, last_valid_y


