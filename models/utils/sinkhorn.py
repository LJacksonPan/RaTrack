import torch
import torch.nn as nn


class SinkhornSolver(nn.Module):
    """
    Optimal Transport solver under entropic regularisation.
    Based on the code of Gabriel Peyré.
    """

    def __init__(self, epsilon, iterations=100, ground_metric=lambda x: torch.pow(x, 2)):
        super(SinkhornSolver, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.ground_metric = ground_metric

    def forward(self, x, y):
        num_x = x.size(-2)
        num_y = y.size(-2)

        batch_size = 1 if x.dim() == 2 else x.size(0)

        # Marginal densities are empirical measures
        a = x.new_ones((batch_size, num_x), requires_grad=False) / num_x
        b = y.new_ones((batch_size, num_y), requires_grad=False) / num_y

        a = a.squeeze()
        b = b.squeeze()

        # Initialise approximation vectors in log domain
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)

        # Stopping criterion
        threshold = 1e-1

        # Cost matrix
        C = self._compute_cost(x, y)

        # Sinkhorn iterations
        for i in range(self.iterations):
            u0, v0 = u, v

            # u^{l+1} = a / (K v^l)
            K = self._log_boltzmann_kernel(u, v, C)
            u_ = torch.log(a + 1e-8) - torch.logsumexp(K, dim=1)
            u = self.epsilon * u_ + u

            # v^{l+1} = b / (K^T u^(l+1))
            K_t = self._log_boltzmann_kernel(u, v, C).transpose(-2, -1)
            v_ = torch.log(b + 1e-8) - torch.logsumexp(K_t, dim=1)
            v = self.epsilon * v_ + v

            # Size of the change we have performed on u
            diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
            mean_diff = torch.mean(diff)

            if mean_diff.item() < threshold:
                break

        print("Finished computing transport plan in {} iterations".format(i))

        # Transport plan pi = diag(a)*K*diag(b)
        K = self._log_boltzmann_kernel(u, v, C)
        pi = torch.exp(K)

        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        return cost, pi

    def _compute_cost(self, x, y):
        x_ = x.unsqueeze(-2)
        y_ = y.unsqueeze(-3)
        C = torch.sum(self.ground_metric(x_ - y_), dim=-1)
        return C

    def _log_boltzmann_kernel(self, u, v, C=None):
        C = self._compute_cost(x, y) if C is None else C
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= self.epsilon
        return kernel