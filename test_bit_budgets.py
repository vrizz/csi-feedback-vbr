"""
How to run:
  python3 test_bit_budgets.py --run lambda-5e-04_myname
"""

import argparse
import os
import sys

import math
# import numpy as np
import pandas as pd
import torch

from compression_utils import precompute_bit_budgets, generate_with_bit_budget
from cost_loader import get_cost_dataset
from metrics import normalized_mean_square_error
from model_vbr import CSIFactorizedPriorVbr


def parse_args(argv):
    parser = argparse.ArgumentParser(description="COST2100 test script")

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
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run name (e.g. lambda-5e-04_myname) to load the best checkpoint from checkpoints/"
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    train_dataloader, val_dataloader, test_dataloader = get_cost_dataset(scenario='in', batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = CSIFactorizedPriorVbr(N=args.N, M=args.M)

    ckpt_path = f"checkpoints/{args.run}_best_loss.pth.tar"
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))

    net = net.to(device)
    net.load_state_dict(checkpoint['state_dict'])

    print("Model loaded")
    net.update(force=True)

    print("Precompute the bit budgets for all samples and bin widths ...")

    bin_widths = [0.125, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32]

    lookup_table = precompute_bit_budgets(model=net, test_loader=test_dataloader,
                                          bin_widths=bin_widths, device=device)
    print("\nLook-up table precomputed")

    results = []
    for b in range(100, 1001, 100):

        print(f"\rEvaluate metrics for B = {b}", end="", flush=True)

        outputs = generate_with_bit_budget(
            model=net,
            test_loader=test_dataloader,
            b=b,
            bin_widths=bin_widths,
            lookup_table=lookup_table,
            device=device
        )

        if outputs is not None:
            reco = outputs["decoded"]
            inputs = outputs["inputs"]
            # inputs_raw = outputs["raw_inputs"]

            nmse = normalized_mean_square_error(reco - 0.5, inputs - 0.5)
            nmse_db = 10 * math.log10(nmse)
            nmse_db = round(nmse_db, 2)

            # outputs_transformed = transform_data(reco)
            # gcs = cosine_similarity(outputs_transformed, inputs_raw)

        else:
            nmse_db = "not available"
            # gcs = "not available"

        # results.append({'B': b, 'NMSE [dB]': nmse_db, 'GCS': gcs})
        results.append({'B': b, 'NMSE [dB]': nmse_db})

    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    csv_name = args.run
    results_df.to_csv(f'results/{csv_name}.csv', index=False)

    # Compute results_variable: for each bin_width, avg compressed size and avg NMSE
    import numpy as np
    results_variable = []
    all_keys = list(lookup_table.keys())
    for bw_idx, bin_width in enumerate(bin_widths):
        print(f"\rEvaluate metrics for bin_width = {bin_width:<6}", end="", flush=True)

        avg_bits = np.mean([lookup_table[k][bw_idx] for k in all_keys])

        net.update(force=True, bin_width=bin_width)
        decoded, inputs_list = [], []
        with torch.no_grad():
            for batch, _ in test_dataloader:
                x = batch.to(device)
                comp = net.compress(x, bin_width)
                result = net.decompress(comp["strings"], comp["shape"], bin_width=bin_width)
                decoded.append(result["x_hat"].cpu())
                inputs_list.append(batch)

        reco = torch.cat(decoded).numpy()
        inp = torch.cat(inputs_list).numpy()
        nmse = normalized_mean_square_error(reco - 0.5, inp - 0.5)
        nmse_db = round(10 * math.log10(nmse), 2)

        results_variable.append({'bin_width': bin_width, 'avg_bits': round(avg_bits, 2), 'NMSE [dB]': nmse_db})

    results_variable_df = pd.DataFrame(results_variable)
    results_variable_df.to_csv(f'results/{csv_name}_variable.csv', index=False)

    print("\nResults are ready.")


if __name__ == "__main__":
    main(sys.argv[1:])
