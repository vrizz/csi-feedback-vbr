import numpy as np
import torch


def precompute_bit_budgets(model, test_loader, bin_widths, device):
    """
    Returns:
        dict: {(batch_idx, sample_idx): [bits_for_bin1, bits_for_bin2, ...]}
        where list corresponds to bin_widths order.
    """
    num_bin_widths = len(bin_widths)
    lookup_table = {}

    model.eval()
    with torch.no_grad():
        for bin_width_idx, bin_width in enumerate(bin_widths):
            model.update(force=True, bin_width=bin_width)
            msg = f"Precomputing bit budgets for bin width {bin_width}"
            print(f"\r{msg:<60}", end="", flush=True)

            for batch_idx, (batch, _) in enumerate(test_loader):
                x = batch.to(device)

                compressed_result = model.compress(x, bin_width)
                y_strings = compressed_result["strings"][0]
                bits = [len(s) * 8 for s in y_strings]

                for sample_idx, bit in enumerate(bits):
                    key = (batch_idx, sample_idx)

                    if key not in lookup_table:
                        lookup_table[key] = [0] * num_bin_widths  # Pre-allocate full-size list

                    lookup_table[key][bin_width_idx] = bit

    return lookup_table


def generate_with_bit_budget(model, test_loader, b, bin_widths, lookup_table, device):
    """
    Generate reconstructions under a bit budget constraint.
    """
    decoded = []
    inputs = []
    raw_inputs = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (batch, batch_raw) in enumerate(test_loader):
            x = batch.to(device)
            reco = []

            for sample_idx, sample in enumerate(x):
                # Lookup bit estimates for current sample
                bit_budgets = lookup_table.get((batch_idx, sample_idx))

                # Find indices that meet the bit budget
                valid_indices = np.where(np.array(bit_budgets) <= b)[0]

                if valid_indices.size == 0:
                    print(
                        f"\nCould not find a suitable bin width for sample {sample_idx} in batch {batch_idx} within bit budget {b}."
                    )
                    return None

                # Use the smallest bin width that satisfies the budget
                selected_bin_width = bin_widths[valid_indices[0]]
                model.update(force=True, bin_width=selected_bin_width)

                compression = model.compress(
                    sample.unsqueeze(0), bin_width=selected_bin_width
                )

                y_strings = compression["strings"][0]

                result = model.decompress(
                    [[y_strings[0]]],
                    compression["shape"],
                    bin_width=selected_bin_width
                )

                reco.append(result["x_hat"])

            # Convert to tensors once per batch and collect
            x_hat_batch = torch.cat([x_hat.cpu() for x_hat in reco], dim=0)
            x_batch = batch.cpu()
            x_raw_batch = batch_raw.cpu()

            decoded.append(x_hat_batch)
            inputs.append(x_batch)
            raw_inputs.append(x_raw_batch)

        return {
            "decoded": torch.cat(decoded, dim=0).numpy(),
            "inputs": torch.cat(inputs, dim=0).numpy(),
            "raw_inputs": torch.cat(raw_inputs, dim=0).numpy()
        }