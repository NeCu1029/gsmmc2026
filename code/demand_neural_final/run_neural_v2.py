"""Reproduce or train the complete public-signature neural ensemble."""
import argparse
from pathlib import Path
import bike_signature_net as signature
import ensemble_signature_net as ensemble
import torch

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train','predict'], nargs='?', default='predict')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1])
    parser.add_argument('--public-dir', type=Path, default=ROOT.parents[1] / 'open_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results_v2')
    parser.add_argument('--previous-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b = signature.base.read_data(args.data_dir, 'B')
    public_days, _, _ = signature.transfer.read_public(args.public_dir, b)
    if args.action == 'train':
        a = signature.base.read_data(args.data_dir, 'A')
        # An existing prior checkpoint and its validation report are reused as-is.
        required = ['public_inventory.pt', 'public_transfer_validation.json']
        if not all((args.previous_dir / name).is_file() for name in required):
            args.previous_dir.mkdir(parents=True, exist_ok=True)
            signature.transfer.train(a, public_days, args.previous_dir)
        signature.train(a, public_days, args.output_dir)
        ensemble.validate(a, args.previous_dir, args.output_dir)
    ensemble.export(b, args.public_dir, args.previous_dir, args.output_dir)


if __name__ == '__main__':
    main()
