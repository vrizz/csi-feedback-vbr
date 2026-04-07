import math

import torch
import torch.nn as nn


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss(reduction="sum")
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

        print("I am using custom RD")

    def forward(self, output, target):
        N, C, H, W = target.size()
        out = {}

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * N))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = self.metric(output["x_hat"], target) / N
        distortion = out["mse_loss"]

        out["loss"] = distortion + self.lmbda * out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
