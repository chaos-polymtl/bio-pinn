# ============================================================================
# Physics-informed Neural Network functions using PyTorch
# Goal : Predict the kinetic constants of artificial data.
# Author : Valérie Bibeau, Polytechnique Montréal, 2023
# ============================================================================

# ---------------------------------------------------------------------------
# Libraries
import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
# ---------------------------------------------------------------------------

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# PINN architecture
class PINeuralNet(nn.Module):

    def __init__(self, device, k, neurons):
        """Constructor

        Args:
            device (string): CPU or CUDA
            k (list): Estimates of kinetic constants
            neurons (int): Number of neurons in hidden layers
        """

        super().__init__()

        self.activation = nn.Tanh()

        self.f1 = nn.Linear(1, neurons)
        self.f2 = nn.Linear(neurons, neurons)
        self.f3 = nn.Linear(neurons, neurons)
        self.out = nn.Linear(neurons, 4)

        self.k1 = torch.tensor(k[0], requires_grad=True).float().to(device)
        self.k2 = torch.tensor(k[1], requires_grad=True).float().to(device)
        self.k3 = torch.tensor(k[2], requires_grad=True).float().to(device)
        self.k4 = torch.tensor(k[3], requires_grad=True).float().to(device)

        self.k1 = nn.Parameter(self.k1)
        self.k2 = nn.Parameter(self.k2)
        self.k3 = nn.Parameter(self.k3)
        self.k4 = nn.Parameter(self.k4)

    def forward(self, x):
        """Forward pass

        Args:
            x (tensor): Input tensor

        Returns:
            tensor: Output tensor
        """

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        
        a = x.float()
        
        z_1 = self.f1(a)
        a_1 = self.activation(z_1)
        z_2 = self.f2(a_1)
        a_2 = self.activation(z_2)
        z_3 = self.f3(a_2)
        a_3 = self.activation(z_3)

        a_4 = self.out(a_3)
        
        return a_4

# Full PINN to discover k
class Curiosity():

    def __init__(self, X, Y, idx, idx_y0, f_hat, learning_rate, k, neurons, regularization, device):
        """Constructor

        Args:
            X (tensor): Input tensor
            Y (tensor): Output tensor
            idx (list): Index of data points in collocation points
            idx_y0 (list): Index of IC points in collocation points
            f_hat (tensor): Null tensor (for residual)
            learning_rate (float): Learning rate for the gradient descent algorithm
            k (list): Estimates of the kinetic constants
            neurons (int): Number of neurons in hidden layers
            regularization (float): Regularization parameter on the MSE of the ODEs
            device (string): CPU or CUDA
        """
        
        def loss_function_ode(output, target):
            
            loss = torch.mean((output - target)**2)

            return loss
        
        def loss_function_data(output, target):

            loss = torch.mean((output[idx] - target[idx])**2)

            return loss
        
        def loss_function_IC(output, target):

            loss = torch.mean((output[idx_y0] - target[idx_y0])**2)

            return loss
        
        self.PINN = PINeuralNet(device, k, neurons).to(device)

        self.PINN.register_parameter('k1', self.PINN.k1)
        self.PINN.register_parameter('k2', self.PINN.k2)
        self.PINN.register_parameter('k3', self.PINN.k3)
        self.PINN.register_parameter('k4', self.PINN.k4)

        self.x = X
        self.y = Y

        self.loss_function_ode = loss_function_ode
        self.loss_function_data = loss_function_data
        self.loss_function_IC = loss_function_IC
        self.f_hat = f_hat
        self.regularization = regularization

        self.device = device
        self.lr = learning_rate
        self.params = list(self.PINN.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def loss(self, x, y_train):
        """Loss function

        Args:
            x (tensor): Input tensor
            y_train (tensor): Output tensor

        Returns:
            float: Evaluation of the loss function
        """

        g = x.clone()
        g.requires_grad = True
        
        y = self.PINN(g)
        cA = y[:,0].reshape(-1,1)
        cB = y[:,1].reshape(-1,1)
        cC = y[:,2].reshape(-1,1)
        cD = y[:,3].reshape(-1,1)

        k1 = self.PINN.k1
        k2 = self.PINN.k2
        k3 = self.PINN.k3
        k4 = self.PINN.k4
        
        grad_cA = autograd.grad(cA, g, torch.ones(x.shape[0], 1).to(self.device), \
                                retain_graph=True, create_graph=True)[0]
        grad_cB = autograd.grad(cB, g, torch.ones(x.shape[0], 1).to(self.device), \
                                retain_graph=True, create_graph=True)[0]
        grad_cC = autograd.grad(cC, g, torch.ones(x.shape[0], 1).to(self.device), \
                                retain_graph=True, create_graph=True)[0]
        grad_cD = autograd.grad(cD, g, torch.ones(x.shape[0], 1).to(self.device), \
                                retain_graph=True, create_graph=True)[0]

        self.loss_cA_ode = self.loss_function_ode(grad_cA + k1*cA - k2*cB*cC, self.f_hat)
        self.loss_cB_ode = self.loss_function_ode(grad_cB - k1*cA + k2*cB*cC, self.f_hat)
        self.loss_cC_ode = self.loss_function_ode(grad_cC - k1*cA + k2*cB*cC \
                                                          + k3*cC - k4*cD, self.f_hat)
        self.loss_cD_ode = self.loss_function_ode(grad_cD - k3*cC + k4*cD, self.f_hat)
        
        
        self.loss_cA_data = self.loss_function_data(cA, y_train[:,0].reshape(-1,1))
        self.loss_cB_data = self.loss_function_IC(cB, y_train[:,1].reshape(-1,1))
        self.loss_cC_data = self.loss_function_data(cC, y_train[:,2].reshape(-1,1))
        self.loss_cD_data = self.loss_function_IC(cD, y_train[:,3].reshape(-1,1))

        self.loss_c_data = self.loss_cA_data + self.loss_cB_data + self.loss_cC_data + self.loss_cD_data
        self.loss_c_ode = self.loss_cA_ode + self.loss_cB_ode + self.loss_cC_ode + self.loss_cD_ode
        
        self.total_loss = self.regularization * self.loss_c_ode + self.loss_c_data
        
        return self.total_loss
    
    def closure(self):
        """Forward and backward pass

        Returns:
            float: Evaluation of the loss function
        """

        self.optimizer.zero_grad()
        
        loss = self.loss(self.x, self.y)
        
        loss.backward()
        
        return loss