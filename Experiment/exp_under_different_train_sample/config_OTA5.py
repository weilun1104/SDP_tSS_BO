
def init_OTA5():
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
    L2_min, L2_max = L2 * (1 - 0.5), L2 * (1 + 0.5)
    W2_min, W2_max = W2 * (1 - 0.5), W2 * (1 + 0.5)

    # 管子三
    L3_min, L3_max = L3 * (1 - 0.5), L3 * (1 + 0.5)
    W3_min, W3_max = W3 * (1 - 0.5), W3 * (1 + 0.5)

    # 将微米转换为米
    um_to_m = 1e-6

    # 现在创建param_ranges列表
    param_ranges = [
        (L1_min * um_to_m, L1_max * um_to_m),
        (L2_min * um_to_m, L2_max * um_to_m),
        (L3_min * um_to_m, L3_max * um_to_m),
        (W1_min * um_to_m, W1_max * um_to_m),
        (W2_min * um_to_m, W2_max * um_to_m),
        (W3_min * um_to_m, W3_max * um_to_m),
    ]
    # 定义一个包含所有约束的字典
    thresholds = {
        'gain': 40,
        'i_multiplier': 3.3,
        'i': 2e-3,
        'phase': 60,
        'gbw': 1e7
    }
    # 用于保存初始满足饱和区的值
    valid_x = dbx_alter = [[2.27e-6, 4.57e-6, 1e-6, 1.1768e-4, 1.0529e-4, 4.68e-5]]
    valid_y = dby_alter = [[44.45, 3.6e-4, 62.28, 43.17e6]]

    # 用于记录上一次满足条件的值
    last_valid_x = [2.27e-6, 4.57e-6, 1e-6, 1.1768e-4, 1.0529e-4, 4.68e-5]
    last_valid_y = [44.45, 3.6e-4, 62.28, 43.17e6]

    return param_ranges, thresholds, valid_x, valid_y, dbx_alter, dby_alter, last_valid_x, last_valid_y

