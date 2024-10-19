import numpy as np
import torch
import torch.nn as nn


class CovarianceLossv2(nn.Module):
    def __init__(self,
                 projector: str = '4096-4096-128',
                 R_ini: float = 1.0,
                 la_R: float = 0.01,
                 la_mu: float = 0.01,
                 R_eps_weight: float = 1e-08
                 ):
        super(CovarianceLossv2, self).__init__()
        sizes = [512] + list(map(int, projector.split('-')))
        proj_output_dim = sizes[-1]
        self.register_buffer('R1', R_ini * torch.eye(proj_output_dim, dtype=torch.float64, requires_grad=False))
        self.register_buffer('mu1', torch.zeros(proj_output_dim, dtype=torch.float64, requires_grad=False))
        self.register_buffer('R2', R_ini * torch.eye(proj_output_dim, dtype=torch.float64, requires_grad=False))
        self.register_buffer('mu2', torch.zeros(proj_output_dim, dtype=torch.float64, requires_grad=False))
        self.register_buffer('new_R1', torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64,
                                                   requires_grad=False))
        self.register_buffer('new_mu1', torch.zeros(proj_output_dim, dtype=torch.float64,
                                                    requires_grad=False))
        self.register_buffer('new_R2', torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64,
                                                   requires_grad=False, ))
        self.register_buffer('new_mu2', torch.zeros(proj_output_dim, dtype=torch.float64,
                                                    requires_grad=False))
        self.la_R = la_R
        self.la_mu = la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = R_eps_weight
        self.register_buffer('R_eps', self.R_eps_weight * torch.eye(proj_output_dim, dtype=torch.float64,
                                                                    requires_grad=False))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        la_R = self.la_R
        la_mu = self.la_mu

        N, D = z1.size()
        # z = torch.cat((z1, z2), 0)

        # z_hat =  z - self.mu   #Previous version
        # R_update = (z_hat.T @ z_hat) / (2*N)   #Previous version
        mu_update1 = torch.mean(z1, 0)
        mu_update2 = torch.mean(z2, 0)

        # self.new_R = la_R*(self.R) + (1-la_R)*(R_update)   #Previous version
        self.new_mu1 = la_mu * (self.mu1) + (1 - la_mu) * (mu_update1)
        self.new_mu2 = la_mu * (self.mu2) + (1 - la_mu) * (mu_update2)

        z1_hat = z1 - self.new_mu1
        z2_hat = z2 - self.new_mu2
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N
        self.new_R1 = la_R * (self.R1) + (1 - la_R) * (R1_update)
        self.new_R2 = la_R * (self.R2) + (1 - la_R) * (R2_update)

        cov_loss = - (torch.logdet(self.new_R1 + self.R_eps) + torch.logdet(self.new_R2 + self.R_eps)) / D

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.mu1 = self.new_mu1.detach()
        self.R2 = self.new_R2.detach()
        self.mu2 = self.new_mu2.detach()

        return cov_loss

    def save_eigs(self) -> np.array:
        with torch.no_grad():
            R_eig = torch.linalg.eigvals(self.R1).unsqueeze(0)
            R_eig_arr = np.real(R_eig.cpu().detach().numpy())
            R_eig_arr = np.sort(R_eig_arr)  # sorted eigenvalues (1, 128)
        return R_eig_arr
