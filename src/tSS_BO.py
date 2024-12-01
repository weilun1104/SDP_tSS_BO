import numpy as np
import torch
import math
from util import *

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
class tSubspace:
    def __init__(self, 
                dim,
                bounds,
                W_prior = None,     # 先验权重矩阵
                mean = None,
                start_f = None,
                gradient_prior = None,
                sigma = 0.2,    # 变异步长
                mu = 0.5,   # 权重调整参数
                c1 = None,
                c2 = None,
                allround_flag = False,  # 是否进行全面更新
                greedy_flag = False,    # 是否进行贪婪更新
                k = 100     # 用于更新步长的缩放因子
                ):
        self.dim = dim
        if bounds is None:
            self.bounds = bounds
        else:
            self.bounds = torch.tensor(bounds)
            self.lb = self.bounds[:, 0]
            self.ub = self.bounds[:, 1]
        if mean is None:
            self.mean = torch.rand(self.dim) * (self.ub - self.lb) + self.lb
        else:
            self.mean = torch.tensor(mean)

        self.mean = self.mean.view([self.dim, 1])

        if sigma is None:
            self.sigma = (torch.mean(self.ub) - torch.mean(self.lb)) / 5
        else:
            self.sigma = sigma

        self.mean_f = start_f   # 初始化函数值

        self.mu = mu    # 缩放因子mu
        # 初始化梯度先验
        if gradient_prior is None:
            self.prior = torch.zeros([dim, 1])
        else:
            self.prior = torch.tensor(gradient_prior).view([dim, 1])

        if c1 is None:
            c1 = (1 - math.exp(math.log(0.01)/k)) / 2

        if c2 is None:
            c2 = (1 - math.exp(math.log(0.01)/k)) / 2

        self._c_j = c1
        self._c_p = c2
        self._c_W = 1 - self._c_j - self._c_p   # 更新矩阵W的权重

        # 初始化W矩阵
        if W_prior is None:
            self._W = torch.eye(self.dim)
        else:
            self._W = torch.tensor(W_prior)

        # 计算更新步长的常数
        self._chi_n = math.sqrt(self.dim) * (
            1.0 - (1.0 / (4.0 * self.dim)) + 1.0 / (21.0 * (self.dim**2))
        )

        self.allround_flag = allround_flag
        self.greedy_flag = greedy_flag
        self.value_sqrt = None  # 存储W矩阵特征值的平方根
        self.Q = None   # 存储W矩阵的特征向量

    # 设置均值函数值
    def set_mean_f(self, mean_f):
        self.mean_f = torch.tensor(mean_f)

    # 设置新的均值
    def set_new_mean(self, mean, mean_f = None):
        self.mean = torch.tensor(mean)
        self.mean_f = mean_f

    # 计算先验梯度，使用最小二乘法
    def _get_prior_gradient(self, prior_x, prior_y, alpha = 0.01):
        assert self.mean_f is not None

        X_torch = torch.tensor(prior_x).view([self.dim, -1])
        Y_torch = torch.tensor(prior_y).view([-1, 1])

        # 计算输入与均值的差异
        Sk = X_torch - self.mean.view([self.dim, -1])

        Yk = Y_torch - self.mean_f
        # 计算最小二乘解
        J = torch.pinverse(Sk.mm(Sk.t()) + alpha * torch.eye(self.dim).to(device)).mm(Sk)
        J = J.mm(Yk.to(torch.double))

        return J

    # 计算desketch梯度
    def desketch_gradient(self, X_torch, Y_torch):
        assert self.mean_f is not None
        X_torch = X_torch.to(device)
        Y_torch = Y_torch.to(device)
        Sk = X_torch - self.mean.view([self.dim, -1])   # 计算输入差异

        Yk = Y_torch - self.mean_f  # 计算输出差异

        # 计算梯度Ak和Jk
        Ak = Sk.mm(torch.pinverse(Sk.t().mm(Sk)))

        Jk = self.prior + Ak.mm(Yk - Sk.t().mm(self.prior))

        return Jk.view([self.dim, -1])

    # 计算Pk值，用于更新方向
    def compute_pk(self, X_torch, Y_torch):
        Y_arg = torch.argsort(Y_torch.ravel())  # 对目标值进行排序
        X_torch = X_torch[:, Y_arg]     # 根据排序结果重新排列输入

        if self.greedy_flag:
            # 使用贪婪更新策略
            weights = torch.zeros_like(Y_arg)
            weights[0] = 1
            self.mu_num = 1
            self.mu_eff = 1
        else:
            # 计算CMA-ES权重
            weights_prime = torch.tensor(
                [
                    math.log((X_torch.size(1) + 1) * self.mu) - math.log(i + 1)
                    for i in range(X_torch.size(1))
                ]
            )


            self.mu_num = math.floor(X_torch.size(1) * self.mu)
            self.mu_eff = (torch.sum(weights_prime[:self.mu_num]) ** 2) / torch.sum(weights_prime[:self.mu_num] ** 2)

            positive_sum = torch.sum(weights_prime[weights_prime > 0])
            negative_sum = torch.sum(torch.abs(weights_prime[weights_prime < 0]))
            weights = torch.where(
                weights_prime >= 0,
                1 / positive_sum * weights_prime,
                0.9 / negative_sum * weights_prime,
            )

        if self.allround_flag:
            # 全面更新策略
            X_delta = (X_mu - self.mean.view([self.dim , -1])) / self.sigma
            p_mu = torch.sum(X_delta * weights.view([1, -1]), 1)
        else:
            # 使用CMA-ES更新策略
            X_mu = X_torch[:, :self.mu_num]
            X_mu = X_mu.to(device)
            X_mu_delta = (X_mu - self.mean.view([self.dim , -1])) / self.sigma
            weights = weights.to(device)
            p_mu = torch.sum(X_mu_delta * weights[:self.mu_num].view([1, -1]), 1)

        

        return p_mu.view([self.dim, -1])
            
    # 更新子空间
    def update_subspace(self, new_x, new_y, new_mean_f = None, GP_model_list = None, mean_and_std = None):
        X_torch = torch.tensor(new_x).view([self.dim, -1])
        Y_torch = torch.tensor(new_y).view([-1, 1])

        if new_mean_f is not None:
            self.mean_f = new_mean_f
            
        Gk = None

        # 如果有GP模型，则计算梯度    
        if GP_model_list is not None and mean_and_std is not None:
            Gk = sample_model_gradient(GP_model_list, mean_and_std[0], mean_and_std[1], self.mean, delta = 0.01).view([-1, 1])
            Gk = Gk / Gk.norm() * self._chi_n  # 归一化

        # 计算梯度Jk和更新方向Pk
        Jk = self.desketch_gradient(X_torch, Y_torch)

        Pk = self.compute_pk(X_torch, Y_torch)

        self.prior = Jk # 更新梯度先验

        Jk = Jk / Jk.norm() * self._chi_n   # 归一化Jk
        # 更新均值
        self.mean = self.mean.ravel() + Pk.ravel() * self.sigma
        self.mean_f = None
        # 特征分解W矩阵
        D, Q = self._eigen_decomposition()
        # 计算归一化的Pk
        W_2 = Q.mm(torch.diag(1/D).mm(Q.t())).to(device)
        Pk_normalized_norm = W_2.mm(Pk).norm()
        #print('previous W eigenvalues, ')
        #print(D)
        print('pk norm, ', Pk.norm())
        print('normalized pk norm, ', Pk_normalized_norm / self._chi_n - 1)
        print('another normalized pk norm, ', Pk_normalized_norm * math.sqrt(self.mu_eff) / self._chi_n)
        # 更新sigma
        #c = (self.mu_eff + 2) / (self.mu_eff + X_torch.size(1) + 5)
        c = (self.mu_eff + 2) / (self.mu_eff + self.dim + 5)
        #self.sigma = self.sigma * 0.96
        #self.sigma = self.sigma * math.exp( c / (1 + c) * (Pk_normalized_norm * math.sqrt(self.mu_eff)/ self._chi_n - 1))
        self.sigma = self.sigma * math.exp( c / (1 + c) * (Pk_normalized_norm / self._chi_n - 1))
        print('sigma, ', self.sigma)

        # 更新W矩阵
        #Pk = Pk * self.mu_eff
        self._W = (self._W).to(device)
        if Gk is None:
            self._W = self._c_W * self._W + self._c_j * Jk.mm(Jk.t()) + self._c_p * Pk.mm(Pk.t())
        else:
            Gk = Gk.to(device)
            self._W = self._c_W * self._W + self._c_j * Jk.mm(Jk.t()) * 3 / 5 + self._c_p * Pk.mm(Pk.t()) * 4 / 5 + self._c_j * Gk.mm(Gk.t()) * 3 / 5
        self.value_sqrt = None  # 清除特征值缓存
        self.Q = None   # 清除特征向量缓存

    # 对W矩阵进行特征值分解
    def _eigen_decomposition(self):
        if self.value_sqrt is not None and self.Q is not None:
            return self.value_sqrt, self.Q

        W = self._W/2 + self._W.t() / 2     # 保证W是对称矩阵
        value, Q = torch.linalg.eigh(W)     # 计算W的特征值和特征向量
        value_sqrt = torch.sqrt(torch.where(value > 1e-12, value, 1e-12))       # 计算特征值的平方根

        self._W = Q.mm(torch.diag(value_sqrt ** 2)).mm(Q.t())   # 更新W矩阵
        self.value_sqrt = value_sqrt    # 保存特征值
        self.Q = Q      # 保存特征向量

        return value_sqrt, Q

    # 从更新的子空间中采样候选解
    def sample_candidates(self, n_candidate = 100, n_resample=10):
        D, B = self._eigen_decomposition()  # 对W矩阵进行特征分解
        x = torch.empty(size = torch.Size([self.dim, 0]))   # 初始化空矩阵
        for i in range(n_resample):     # 进行多次采样
            if x.size(1) >= n_candidate:
                break
            z = torch.randn(self.dim, n_candidate)      # 生成随机噪声
            z = z.to(device)
            D = D.to(device)
            B = B.to(device)
            y = B.mm(torch.diag(D)).mm(z)   # 生成扰动
            y = y.to(device)
            x_candidate = self.mean.view([self.dim, 1]) + y * self.sigma    # 生成候选解
            if self.bounds is None:
                x = x_candidate
            else:
                # 过滤掉超出边界的候选解
                inbox = torch.all(x > self.lb.view([-1,1]), 0).multiply(
                        torch.all(x < self.ub.view([-1,1]), 0)
                    )
                if inbox.size(0):
                    x = torch.cat([x, x_candidate[:, inbox]], 1)

        # 如果候选解不足，继续采样
        if x.size(1) < n_candidate:
            n_sample = n_candidate - x.size(1)
            z = torch.randn(self.dim, n_sample)
            z = z.to(device)
            D = D.to(device)
            B = B.to(device)
            y = B.mm(torch.diag(D)).mm(z)
            y = y.to(device)
            x_candidate = self.mean.view([self.dim, 1]) + y * self.sigma
            x = x.to(device)
            x = torch.cat([x, x_candidate], 1)
        # 将候选解限制在边界内
        x = x.to('cpu')
        x = x.clip(min = self.lb.view([-1, 1]), max = self.ub.view([-1, 1]))
        x = x[:, :n_candidate]  # 只返回需要数量的候选解
        return x









