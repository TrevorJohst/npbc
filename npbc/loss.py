from typing import Literal
import torch
import torch.nn as nn
from .model import NormalWishart

class BayesianLoss(nn.Module):
    """
    The Bayesian loss computes an uncertainty-aware loss based on the parameters of a conjugate
    prior of the target distribution. Modified to use the analytical entropy weighting derived in
    https://ieeexplore.ieee.org/document/10611342
    """

    def __init__(
        self, 
        reduction: Literal["mean", "sum", "none"] = "mean"
    ):
        """
        Args:
            reduction: The reduction to apply to the loss. Must be one of "mean", "sum", "none".
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self, 
        y_pred: NormalWishart, 
        log_evidence: torch.Tensor, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the loss of the prediction with respect to the target.

        Args:
            y_pred: The posterior distribution predicted by the Natural Posterior Network.
            log_evidence: The scaled predicted density for `y_pred`.
            y_true: The true targets. Must have the same batch shape as `y_pred`.

        Returns:
            The loss, processed according to `self.reduction`.
        """
        nll = -y_pred.expected_log_likelihood(y_true)
        # loss = nll - log_evidence.reciprocal() * y_pred.entropy()
        loss = nll - 1e-5 * y_pred.entropy()

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
