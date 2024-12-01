import gpytorch
import torch
import torch.optim as optim
import math
from gpytorch.constraints.constraints import Interval
from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

# 设置设备，使用GPU如果可用，否则使用CPU
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class NARGPKernel(gpytorch.kernels.Kernel):
    def __init__(self, n, kernel='RBF'):
        super(NARGPKernel, self).__init__()
        self.n = n
        if kernel == 'RBF':
            self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.n-1, active_dims=torch.arange(self.n-1, device=device))
            self.rbf2 = gpytorch.kernels.RBFKernel(ard_num_dims=self.n-1, active_dims=torch.arange(self.n-1, device=device))
            self.rbf3 = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([self.n - 1], device=device))
            self.mykernel = gpytorch.kernels.ScaleKernel(self.rbf2 * self.rbf3) + gpytorch.kernels.ScaleKernel(self.base_kernel)
        elif kernel == 'MAT52':
            self.base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.n-1, active_dims=torch.arange(self.n-1, device=device))
            self.rbf2 = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.n-1, active_dims=torch.arange(self.n-1, device=device))
            self.rbf3 = gpytorch.kernels.MaternKernel(nu=2.5, active_dims=torch.tensor([self.n - 1], device=device))
            self.mykernel = gpytorch.kernels.ScaleKernel(self.rbf2 * self.rbf3) + gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x1, x2, **params):
        return self.mykernel(x1, x2)

class GP(ExactGP, GPyTorchModel):
    def __init__(self, dataset, kernel='RBF', inner_kernel='RBF', k=1):
        # 确保所有输入张量在同一设备上
        self.x_train = dataset['train_x'].to(device)
        self.y_train = dataset['train_y'].view([-1]).to(device)
        self.n_train = self.x_train.size(0)
        self.n_dim = self.x_train.size(1)
        self.sigma_train = dataset['train_sigma'].to(device) if dataset['train_sigma'] is not None else None

        if self.sigma_train is not None:
            super(GP, self).__init__(self.x_train, self.y_train, gpytorch.likelihoods.FixedNoiseGaussianLikelihood(self.sigma_train))
        else:
            super(GP, self).__init__(self.x_train, self.y_train, gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(5e-4, 0.2)))

        if kernel == 'RBF':
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.n_dim)
        elif kernel == 'MAT52':
            base_kernel = gpytorch.kernels.MaternKernel(lengthscale_constraint=Interval(0.005, math.sqrt(self.n_dim)),
                                                        nu=2.5, ard_num_dims=self.n_dim)

        self.mean_module = gpytorch.means.ConstantMean().to(device)
        if k == 1:
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_constraint=Interval(0.05, 20)).to(device)
        else:
            self.covar_module = NARGPKernel(self.n_dim, kernel=inner_kernel).to(device)

    def fit(self):
        self.train()
        self.likelihood.train()
        
        optimizer = optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        num_epochs = 100
        for _ in range(num_epochs):
            optimizer.zero_grad()
            output = self(self.x_train)
            loss = -mll(output, self.y_train)

            loss.backward()
            optimizer.step()

        self.eval()
        self.likelihood.eval()

    def forward(self, x):
        x = x.to(device)  # 确保输入张量在正确的设备上
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return dist

    def predict(self, x, full_cov=0):
        x = x.to(device)  # 确保输入张量在正确的设备上
        pred = self(x)
        if full_cov:
            return pred.mean.detach(), pred.covariance_matrix.detach()
        else:
            return pred.mean.detach(), pred.variance.detach()
