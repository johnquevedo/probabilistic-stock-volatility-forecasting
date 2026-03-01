from __future__ import annotations

from dataclasses import dataclass

import gpytorch
import numpy as np
import torch


class VolatilityExactGP(gpytorch.models.ExactGP):
    """Exact GP for realized volatility forecasting."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        use_periodic_kernel: bool = True,
        periodic_dim: int | None = None,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        if use_periodic_kernel:
            active_dim = periodic_dim if periodic_dim is not None else train_x.shape[-1] - 1
            periodic = gpytorch.kernels.PeriodicKernel(active_dims=[active_dim])
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel + periodic)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@dataclass(slots=True)
class GPFitResult:
    """Container for a trained GP and basic fit diagnostics."""

    model: VolatilityExactGP
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    train_loss: float
    epochs_run: int


def fit_exact_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int,
    lr: float,
    patience: int,
    min_improvement: float,
    use_periodic_kernel: bool,
    periodic_dim: int | None,
    device: torch.device,
) -> GPFitResult:
    """Fit an exact GP with Adam and simple early stopping."""
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = VolatilityExactGP(
        train_x,
        train_y,
        likelihood,
        use_periodic_kernel=use_periodic_kernel,
        periodic_dim=periodic_dim,
    ).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    marginal_ll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float("inf")
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -marginal_ll(output, train_y)
        loss.backward()
        optimizer.step()

        current_loss = float(loss.item())
        if best_loss - current_loss > min_improvement:
            best_loss = current_loss
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "likelihood": {k: v.detach().cpu().clone() for k, v in likelihood.state_dict().items()},
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])

    model.eval()
    likelihood.eval()
    return GPFitResult(model=model, likelihood=likelihood, train_loss=best_loss, epochs_run=epoch)


def predict_distribution(
    model: VolatilityExactGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    features: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictive mean and standard deviation for feature rows."""
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(features.to(device)))
        mean = posterior.mean.detach().cpu().numpy()
        std = posterior.stddev.detach().cpu().numpy()
    return mean, std
