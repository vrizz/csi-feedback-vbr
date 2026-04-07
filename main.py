import argparse
import random

import math
import torch.optim as optim

from cost_loader import get_cost_dataset
from model_vbr import CSIFactorizedPriorVbr
from loss import RateDistortionLoss
from train_utils import *

import numpy as np

import torch.backends.cudnn as cudnn
import wandb

import os
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser(description="COST2100 main script")

    parser.add_argument(
        '-train',
        '--train',
        action='store_true',
        help="Enable training if set."
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=5e-4/11.8,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "-N",
        "--N",
        type=int,
        default=128,
        help="N parameter of the CSIFactorizedPriorVbr model (default: %(default)s)"
    )
    parser.add_argument(
        "-M",
        "--M",
        type=int,
        default=64,
        help="M parameter of the CSIFactorizedPriorVbr model (default: %(default)s)"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=1000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save model to disk"
    )
    parser.add_argument(
        "--seed",
        default=42,
        help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Additional name suffix for the run (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    run_name = f"lambda-5e-4" + (f"_{args.name}" if args.name else "")
    wandb.init(project="csi-feedback-vbr", name=run_name, config=vars(args))

    print("training: ", args.train)
    print("lambda: ", args.lmbda)
    print("M: ", args.M)
    print("N: ", args.N)
    print("epochs: ", args.epochs)

    if args.seed is not None:
        print("Initializing seed...")
        seed = int(args.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        np.random.seed(seed)
        random.seed(seed)

        # Ensure deterministic behavior in CUDA
        cudnn.deterministic = True
        cudnn.benchmark = False

        # Optional: Set environment variable for further determinism (rare)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # for CUDA >= 10.2

    train_dataloader, val_dataloader, test_dataloader = get_cost_dataset(scenario='in', batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = CSIFactorizedPriorVbr(N=args.N, M=args.M)

    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    training = args.train

    if training:

        print("Training...")

        train_loss_history = []
        val_loss_history = []

        best_loss = float("inf")
        for epoch in range(args.epochs):
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

            # Train and get training loss
            train_loss, train_mse_loss, train_bpp_loss, train_aux_loss = train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                args.clip_max_norm,
            )
            train_loss_history.append(train_loss)  # Track train loss

            # Validate and get validation loss
            val_loss, val_mse_loss, val_bpp_loss, val_aux_loss = test_epoch(epoch, val_dataloader, net, criterion)
            val_loss_history.append(val_loss)  # Track validation loss

            wandb.log({
                "train_loss": train_loss,
                "train_mse_loss": train_mse_loss,
                "train_bpp_loss": train_bpp_loss,
                "train_aux_loss": train_aux_loss,
                "val_loss": val_loss,
                "val_mse_loss": val_mse_loss,
                "val_bpp_loss": val_bpp_loss,
                "val_aux_loss": val_aux_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
            }, step=epoch)

            # Adjust learning rate based on validation loss
            lr_scheduler.step(val_loss)

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": val_loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    prefix=run_name,
                )



    print("Testing...")

    checkpoint = torch.load(f"checkpoints/{run_name}_best_loss.pth.tar", map_location=torch.device(device))

    net = net.to(device)
    net.load_state_dict(checkpoint['state_dict'])

    print("model loaded")
    net.update(force=True)

    # TEST THEORETICAL AND EMPIRICAL AVERAGE COMPRESSED SIZE FOR THE TEST SET
    from test_utils import test_average_compressed_size

    average_theo = test_average_compressed_size(net, test_dataloader, device, mode="theoretical")
    average_emp = test_average_compressed_size(net, test_dataloader, device, mode="empirical")

    print(f"Average compressed size theoretical: {average_theo} bits per sample")
    print(f"Average compressed size empirical: {average_emp} bits per sample")

    # TEST NMSE
    from test_utils import get_avg_nmse

    average_nmse = get_avg_nmse(net, test_dataloader, device)
    print(f"Average NMSE [dB]: {10 * math.log10(average_nmse)}")

    wandb.log({
        "average_compressed_size_theoretical": average_theo,
        "average_compressed_size_empirical": average_emp,
        "average_nmse_db": 10 * math.log10(average_nmse),
    })

    artifact = wandb.Artifact(name=f"{run_name}_best", type="model")
    artifact.add_file(f"checkpoints/{run_name}_best_loss.pth.tar")
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
