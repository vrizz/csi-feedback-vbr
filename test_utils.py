import math
import torch
import torch.nn as nn


def test_average_compressed_size(net, test_dataloader, device, mode="theoretical"):
    total_bits = 0
    total_samples = 0

    net.eval()
    with torch.no_grad():
        for batch, batch_raw in test_dataloader:
            x = batch
            x = x.to(device)

            batch_size = x.size(0)
            total_samples += batch_size

            if mode == "theoretical":
                returns = net.forward(x)
                total_bits += torch.log(returns["likelihoods"]["y"]).sum().item()

            elif mode == "empirical":
                compressed_result = net.compress(x)
                y_strings = compressed_result["strings"][0]

                bits = [len(s) * 8 for s in y_strings]
                total_bits += sum(bits)

    if mode == "theoretical":
        average_compressed_size = total_bits / (-math.log(2) * total_samples)
    elif mode == "empirical":
        average_compressed_size = total_bits / total_samples
    return average_compressed_size


class NormalizedMeanSquareError(nn.Module):
    def __init__(self, epsilon=1e-10):
        super(NormalizedMeanSquareError, self).__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        """
        Calculate the normalized mean square error (NMSE) between two 4D tensors along dim 0.

        Parameters:
        outputs (torch.Tensor): The model's output 4D tensor.
        targets (torch.Tensor): The target or reference 4D tensor.

        Returns:
        torch.Tensor: The NMSE value.
        """
        if outputs.shape != targets.shape:
            raise ValueError("The input tensors must have the same shape.")

        outputs = outputs - 0.5
        targets = targets - 0.5

        # Calculate the MSE along the first dimension
        mse = torch.sum((outputs - targets) ** 2, dim=(1, 2, 3))

        # Calculate the variance of the target tensor
        power = torch.sum(targets ** 2, dim=(1, 2, 3))

        # Avoid division by zero by adding a small epsilon where variance is zero
        nmse = mse / (power + self.epsilon)

        return nmse.sum()


def get_avg_nmse(net, test_dataloader, device):
    nmse_acc = 0
    total_samples = 0

    nmse_func = NormalizedMeanSquareError()
    net.eval()
    with torch.no_grad():
        for batch, batch_raw in test_dataloader:
            x = batch
            x = x.to(device)

            batch_size = x.size(0)
            total_samples += batch_size

            outputs = net.forward(x)

            nmse = nmse_func(outputs["x_hat"], x)

            nmse_acc += nmse.item()

    return nmse_acc / total_samples
