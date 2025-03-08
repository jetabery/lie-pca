import torch
import torch.nn as nn
from torus_rep import create_dataset, reorder_dataset, get_irreps, construct_orbit, plot_results
import numpy as np


class TorusRep(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor, omega: torch.Tensor, x0: torch.Tensor, period: float = 1.0):
        """
        A: torch.Tensor (NxN complex matrix)
        B: torch.Tensor (NxN complex matrix)
        omega: torch.Tensor (N-dimensional integer vector)
        x0: torch.Tensor (N-dimensional complex vector)
        period: float (real scalar)
        """
        super().__init__()
        self.A = nn.Parameter(A)  # Store A as a learnable parameter if needed
        self.B = nn.Parameter(B)  # Store B as a learnable parameter if needed
        self.omega = nn.Parameter(omega, requires_grad=False)  # Fixed frequencies
        self.x0 = nn.Parameter(x0, requires_grad=False)  # Fixed initial vector
        self.period = nn.Parameter(torch.tensor(period), requires_grad=False)  # Fixed period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: torch.Tensor (scalar or tensor of shape (batch_size, 1))
        Returns: torch.Tensor (complex vector of shape (batch_size, N))
        """
        # Compute the diagonal complex exponentials
        exp_diag = torch.exp(2j * torch.pi * self.omega * t / self.period)  # Shape: (batch_size, N)

        # Reshape for matrix multiplication
        exp_diag_matrix = torch.diag_embed(exp_diag)  # Shape: (batch_size, N, N)

        # Perform matrix operations: A * exp(D) * B * x0
        out = self.A @ exp_diag_matrix @ self.B @ self.x0.unsqueeze(-1)  # Shape: (batch_size, N, 1)

        return out.squeeze(-1)  # Shape: (batch_size, N)

if __name__=='__main__':
    N = 3  # Dimension of the matrices
    A = torch.randn(N, N, dtype=torch.cfloat)
    B = torch.randn(N, N, dtype=torch.cfloat)
    omega = torch.randint(low=-5, high=5, size=(N,), dtype=torch.int)  # Integer frequencies
    x0 = torch.randn(N, dtype=torch.cfloat)

    model = TorusRep(A, B, omega, x0)

    t = torch.arange(0, 1, 0.001).unsqueeze(-1)
    output = model(t).detach().numpy().T.astype(np.complex128)
    print(output.dtype)
    plot_results(output, output, projection='random')