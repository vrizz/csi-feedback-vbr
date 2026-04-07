import os
import shutil

import torch
import torch.nn as nn

from compressai.optimizers import net_aux_optimizer


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device
    running_loss = 0.0
    total_samples = 0

    running_mse_loss = 0.0
    running_bpp_loss = 0.0
    running_aux_loss = 0.0

    for i, (d,) in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        running_loss += out_criterion["loss"].item() * d.size(0)
        total_samples += d.size(0)


        running_mse_loss += out_criterion["mse_loss"].item() * d.size(0)
        running_bpp_loss += out_criterion["bpp_loss"].item() * d.size(0)
        running_aux_loss += aux_loss.item() * d.size(0)

        if i % 256 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.6f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

    # Return average loss for this epoch
    return running_loss / total_samples, running_mse_loss / total_samples, running_bpp_loss / total_samples, running_aux_loss / total_samples


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for (d,) in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.6f} |"
        f"\tBpp loss: {bpp_loss.avg:.3f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg.item(), mse_loss.avg.item(), bpp_loss.avg.item(), aux_loss.avg.item()


def save_checkpoint(state, is_best, prefix="checkpoint"):
    os.makedirs("checkpoints", exist_ok=True)
    filename = f"checkpoints/{prefix}.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"checkpoints/{prefix}_best_loss.pth.tar")
