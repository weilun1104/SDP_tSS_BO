import sys
import os
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,pythonpath)
from tSS_BO import tSubspace
from util import *
import numpy as np
import torch
import torch.multiprocessing as mp
import math
import time
import pickle
import pandas as pd
from utils.util import seed_set
from Experiment.exp_under_different_train_sample.config_OTA_three import init_OTA_three
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_V4 import OTA_three_simulation_all
from Model.Point_search.CONBO import plot, save_data

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

mp.set_sharing_strategy('file_system')  #设置多进程共享文件系统
strat_time = time.time()
def print_err(err):
    print(err)
    
def parallel_simulate(i, f, x, fX_pool, pid_set, output = 1):   #并行仿真函数
    pid = os.getpid()
    if pid in pid_set:
        index = pid_set[pid]
    else:
        pid_set[pid] = None
        index = list(pid_set.keys()).index(pid)
        pid_set[pid] = index
    fX_pool[i] = [f(x, index, output)]

def save_data(best_all_y, iter_times, save_path):
    best_all_y1 = [item[0][0].item() for item in best_all_y]
    best_all_y2 = [item[0][1].item() for item in best_all_y]
    best_all_y3 = [item[0][2].item() for item in best_all_y]
    best_all_y4 = [item[0][3].item() for item in best_all_y]
    # 将5个list保存到一个CSV文件中
    df = pd.DataFrame({
        'iter_times': iter_times,
        'gain(db)': best_all_y1,
        'dc_current': best_all_y2,
        'phase': best_all_y3,
        'GBW(MHZ)': best_all_y4,
    })
    df.to_csv(save_path, index=False)

def objective_function_constrained(x):
    x_tensor = torch.tensor(x, dtype=torch.float32) # 需要转换为 PyTorch tensor
    results = OTA_three_simulation_all(x_tensor)
    gain, dc_current, phase, GBW = results[0]
    print(f"Gain: {gain}, DC Current: {dc_current}, Phase: {phase}, GBW: {GBW}")
    return [gain.item(), dc_current.item(), phase.item(), GBW.item()]

# 定义约束函数
def constraint_1(y):
    return (y[0, 0] > (80)).item()

# def constraint_2(y):
#     return (y[0, 1] < (1e-3 / 1.8)).item()

def constraint_3(y):
    return (y[0, 2] > (60)).item()

def constraint_4(y):
    return (y[0, 3] > (2e6)).item()

def all_constraints(y):
    return constraint_1(y) and constraint_3(y) and constraint_4(y)
            
def main_solver_constrained(funct, dim, bounds, init_x = None, init_y = None,
                sigma = 0.2, mu = 0.5, c1 = None, c2 = None, allround_flag = False, greedy_flag = False,
                n_training = None, batch_size = 40, n_candidates = 200, n_resample = 10, nMax = 3000 , k = 100, dataset_file = './dataset_tTs_bo.pkl', use_BO = True, use_TS = True, outdim = 4, calculate_model_gradient = False
                 ):
    t_funct = lambda x: torch.tensor(funct(x)).float()      #返回目标值和不同的约束值
    #funct returns [obj, cons1, cons2, ... ,] for each sample in one line

    if n_training is None:
        n_training = min(dim * 2, 23)

    subspace = tSubspace(dim, bounds, sigma = sigma, mu = mu, c1 =   c1, c2 = c2, allround_flag = allround_flag, greedy_flag = greedy_flag, k = k)

    evaluation_hist = [torch.empty(size = torch.Size([0, dim])), torch.empty(size = torch.Size([0, outdim]))]
    t0 = time.time()
    # 初始化最佳值列表，并将最小电流值设为无穷大
    best_all_y = []
    best_y = None
    min_current_value = float('inf')
    if init_x is not None:
        x = torch.tensor(init_x)
        x_exp = torch.exp(x)        
        # 对每个样本的 x 进行处理
        for i in range(x.size(0)):
            y = t_funct(x_exp[i].unsqueeze(0))  # 计算 t_funct 的输出  
            y = y.unsqueeze(0)

            # 检查 y 是否满足所有约束条件
            if all_constraints(y):
                if y[0][0].item() < min_current_value:
                    min_current_value = y[0][0].item()
                    best_y = y.clone()
                # 如果 init_y 为空，则初始化为第一个满足条件的样本的结果
                if init_y is None:
                    init_y = y
                    new_x = x[i].unsqueeze(0)  # 初始化 new_x
                else:
                    # 将满足条件的 y 和对应的 x 添加到 init_y 和 new_x 中
                    init_y = torch.cat((init_y, y), dim=0)
                    new_x = torch.cat((new_x, x[i].unsqueeze(0)), dim=0)
            best_all_y.append(best_y)

        print('current length of best_all_y:', len(best_all_y))
        # 处理 init_y 中的第二列
        init_y[:, 1] = torch.log(init_y[:, 1])

        # 对 new_x 和 init_y 进行排序
        new_x = new_x[torch.argsort(init_y[:, 1]), :]  # 根据 init_y[:, 1] 排序 new_x

        evaluation_hist[0] = torch.cat([evaluation_hist[0], new_x], 0)
        evaluation_hist[1] = torch.cat([evaluation_hist[1], init_y], 0)
        weights_prime = torch.tensor(
            [
                math.log((new_x.size(0) + 1) * mu) - math.log(i + 1)
                for i in range(new_x.size(0))
            ]
        )


        mu_num = math.floor(new_x.size(0) * mu)
        mu_eff = (torch.sum(weights_prime[:mu_num]) ** 2) / torch.sum(weights_prime[:mu_num] ** 2)

        positive_sum = torch.sum(weights_prime[weights_prime > 0])
        negative_sum = torch.sum(torch.abs(weights_prime[weights_prime < 0]))

        weights = torch.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            0.9 / negative_sum * weights_prime,
        )

        X_mu = new_x[:mu_num,:]
        weights = weights.to(device)
        
        mean_prior = torch.sum(X_mu * weights[:mu_num].view([-1, 1]), 0)
        mean_prior_exp = torch.exp(mean_prior)
        f_mean_prior = torch.tensor(funct(mean_prior_exp.unsqueeze(0))).float().ravel()

        f_mean_prior[1] = torch.log(f_mean_prior[1])

        subspace.set_new_mean(mean_prior, f_mean_prior[1])   #子空间的更新不仅依据目标函数的值，还参考了解是否满足约束

        J0 = subspace._get_prior_gradient(new_x.t(), init_y[:, 1])

        subspace.prior = J0
        evaluation_hist[0] = torch.cat([evaluation_hist[0], mean_prior.view([1, -1])], 0)
        evaluation_hist[1] = torch.cat([evaluation_hist[1], f_mean_prior.view([1, -1])], 0)
    t1 = time.time()
    print('initialization time, ', (t1 - t0))
    g = 0
    while len(best_all_y) < 400:
        g = g + 1
        print('iteration ,', g)
        t0 = time.time()
        X_candidates = subspace.sample_candidates(n_candidates, n_resample).t()

        t1 = time.time()
        print('candidate generation time, ', (t1 - t0))

        if evaluation_hist[0].size(0) and use_BO:
            X = evaluation_hist[0]
            Y = evaluation_hist[1]
            X_center = subspace.mean.ravel()

            if X.size(0) >= n_training:
                D, B = subspace._eigen_decomposition()
                X, Y = select_training_set(
                        X_center,
                        X,
                        Y,
                        B,
                        D,
                        n_training = n_training
                    )

            if use_TS:
                X_cand, model_list, m_n_s = select_candidate_TS_constrained(
                        X,
                        Y,
                        X_candidates,
                        batch_size = batch_size
                    )
            else:
                X_cand, model_list, m_n_s = select_candidate_EI_constrained(
                        X,
                        Y,
                        X_candidates,
                        batch_size = batch_size
                    )
        else:
            X_cand = X_candidates[:batch_size, :]
        t2 = time.time()
        print('candidate selection time, ', (t2 - t1))
        if not calculate_model_gradient:
            model_list, m_n_s = None, None
        
        if subspace.mean_f is None:
            dby_alter_train = torch.empty((0,), dtype=torch.double).to(device)
            X_cand_filtered = torch.empty((0, X_cand.size(1)), dtype=torch.double).to(device)  # 初始化用于存储符合条件的 x
            for i in range(X_cand.size(0)):
                X_cand_exp = torch.exp(X_cand[i])
                Y_cand = t_funct(X_cand_exp.unsqueeze(0))
                Y_cand = Y_cand.unsqueeze(0)
                
                # 检查是否满足所有约束条件
                if all_constraints(Y_cand):
                    # 满足条件时，将 Y_cand 添加到 dby_alter_train
                    dby_alter_train = torch.cat((dby_alter_train, Y_cand), dim=0)
                    X_cand_filtered = torch.cat((X_cand_filtered, X_cand[i].unsqueeze(0)), dim=0)  # 将满足条件的 x 添加到 X_cand_filtered

                    # 检查当前的 Y_cand[0][1] 是否为最小电流值
                    if Y_cand[0][1].item() < min_current_value:
                        min_current_value = Y_cand[0][1].item()  # 更新最小电流值
                        best_y = Y_cand.clone()  # 更新最小电流值对应的 y

                # 将最佳的 Y_cand 添加到列表中
                best_all_y.append(best_y)
            
            print('current length of best_all_y:', len(best_all_y))
            t3 = time.time()
            
            dby_alter_train[:, 1] = torch.log(dby_alter_train[:, 1])
            subspace.set_mean_f(dby_alter_train[0, 1])
            subspace.update_subspace(X_cand_filtered[1:, :].t(), dby_alter_train[1:, 1], GP_model_list=model_list, mean_and_std=m_n_s)
            t4 = time.time()

        else:
            dby_alter_train = torch.empty((0,), dtype=torch.double).to(device)
            X_cand_filtered = torch.empty((0, X_cand.size(1)), dtype=torch.double).to(device)  # 初始化用于存储符合条件的 x
            for i in range(X_cand.size(0)):
                X_cand_exp = torch.exp(X_cand[i])
                Y_cand = t_funct(X_cand_exp.unsqueeze(0))
                Y_cand = Y_cand.unsqueeze(0)
                
                if all_constraints(Y_cand):
                    # 满足条件时，添加 Y_cand 和对应的 x
                    dby_alter_train = torch.cat((dby_alter_train, Y_cand), dim=0)
                    X_cand_filtered = torch.cat((X_cand_filtered, X_cand[i].unsqueeze(0)), dim=0)  # 添加满足条件的 x

                    # 检查当前的 Y_cand[0][1] 是否为最小电流值
                    if Y_cand[0][1].item() < min_current_value:
                        min_current_value = Y_cand[0][1].item()  # 更新最小电流值
                        best_y = Y_cand.clone()  # 更新最小电流值对应的 y

                best_all_y.append(best_y)  # 将最佳的 Y_cand 添加到列表中

            print('current length of best_all_y:', len(best_all_y))
            dby_alter_train[:, 1] = torch.log(dby_alter_train[:, 1])
            t3 = time.time()
            subspace.update_subspace(X_cand_filtered.t(), dby_alter_train[:, 1], GP_model_list=model_list, mean_and_std=m_n_s)
            t4 = time.time()

        print('simulation time, ', (t3 -t2))
        print('update subspace time, ', (t4 -t3))

        evaluation_hist[0] = torch.cat([evaluation_hist[0], X_cand_filtered], 0)
        evaluation_hist[1] = torch.cat([evaluation_hist[1], dby_alter_train], 0)
        print('best y', torch.min(evaluation_hist[1][:, 0]))
                                                  
    return best_all_y

if __name__=='__main__':
    for m in range(100, 106):
        seed = m
        seed_set(seed)

        param_ranges, thresholds, valid_x, valid_y, dbx_alter, dby_alter, last_valid_x, last_valid_y = init_OTA_three()
        dbx_alter = torch.tensor(dbx_alter, dtype=torch.double).to(device)
        init_num = 40
        new_x_values = torch.empty((init_num, len(param_ranges)), dtype=torch.double).to(device)
        for i in range(len(param_ranges)):
            low, high = param_ranges[i]
            new_x_values[:, i] = torch.tensor(np.random.uniform(low, high, init_num), dtype=torch.double).to(device)
        dbx_alter = torch.cat((dbx_alter, new_x_values), dim=0)
        dbx_alter = torch.log(dbx_alter)
        # dby_alter = torch.log(dby_alter)
        log_param_ranges = [(np.log(low), np.log(high)) for low, high in param_ranges]
        dim = len(param_ranges) 
        best_all_y = main_solver_constrained(objective_function_constrained, dim, log_param_ranges, dbx_alter, init_y = None,
                sigma = 0.2, mu = 0.5, c1 = None, c2 = None, allround_flag = False, greedy_flag = False,
                n_training = None, batch_size = 20, n_candidates = 200, n_resample = 10, nMax = 3000 , k = 100, dataset_file = './dataset_tTs_bo.pkl', use_BO = True, use_TS = True, outdim = 4, calculate_model_gradient = False
                 )
        # 二级保存路径
        # 实验结果保存路径  
        file_path = ('C:/DAC/tSS-BO-main/Experiment/exp_under_different_train_sample/exp_design_1/baseline_Model_v1/tSS-BO/tSS-BO_OTA_three_seed_{}.csv').format(m)
        # 实验结果计算路径
        cal_path = (
                "C:\\DAC\\tSS-BO-main\\Experiment\\exp_under_different_train_sample\\exp_design_1\\baseline_Model_v1\\tSS-BO\\tSS-BO_OTA_three_seed_")
        # 均值方差计算结果保存路径
        to_path = (
            'C:\\DAC\\tSS-BO-main\\Experiment\\exp_under_different_train_sample\\exp_design_1_report\\tSS-BO_OTA_three_current_mean_var_strand.csv')

        iter_times = list(range(1, len(best_all_y) + 1))
        save_data(best_all_y, iter_times, file_path)

        plot(cal_path, to_path, [100, 102, 101, 104, 103])