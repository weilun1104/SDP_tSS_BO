import numpy as np
import torch
from SALib.sample import saltelli, sobol_sequence, morris as morris_sample
from SALib.analyze import sobol, morris


def sobol_sensitivity_analysis(model_func, problem, num_samples, target):
    """
    执行敏感性分析的函数。

    参数:
    model_func - 一个接受一个参数数组并返回单个数值输出的模型函数。
    problem - 一个字典，定义了变量的数量、名字和范围。
    num_samples - 生成样本点的数量。
    target - 要分析的性能指标编号

    返回:
    (Si['S1'], Si['ST']) - 第一阶Sobol指数和总Sobol指数。
    """
    # 生成样本点
    # param_values = sobol_sequence.sample(num_samples, problem['num_vars'])
    param_values = saltelli.sample(problem, num_samples)
    param_values = torch.from_numpy(param_values)

    # 计算模型输出，这里后续要改为可以同时分析`全部性能指标
    Y = np.array([model_func(params.reshape(1, -1))[0, target].item() for params in param_values])

    # 进行敏感性分析
    # Si = sobol.analyze(problem, Y, calc_second_order=False)
    Si = sobol.analyze(problem, Y)

    # 返回第一阶和总Sobol指数
    return Si['S1'], Si['ST']


def morris_sensitivity_analysis(model_func, problem, num_samples, target, num_levels):
    """
    执行 Morris 敏感性分析的函数。

    参数:
    model_func - 一个接受一个参数数组并返回单个数值输出的模型函数。
    problem - 一个字典，定义了变量的数量、名字和范围。
    num_samples - 每个参数生成的轨迹数量。
    target - 要分析的性能指标编号。
    num_levels - 网格中的级别数。
    grid_jump - 网格间隔。

    返回:
    Si - Morris 方法分析结果的字典。
    """
    # 生成样本点
    param_values = morris_sample.sample(problem, N=num_samples, num_levels=num_levels)
    param_values = torch.from_numpy(param_values).float()

    # 计算模型输出
    Y = np.array([model_func(params.reshape(1, -1))[0, target].item() for params in param_values])

    # 进行敏感性分析
    Si = morris.analyze(problem, param_values.numpy(), Y, num_levels=num_levels)

    return Si