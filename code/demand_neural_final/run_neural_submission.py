"""Train both neural candidates or reproduce the selected B submission."""
import argparse
from pathlib import Path
import bike_neural_final as forecast
import bike_inventory_net as inventory
import bike_public_transfer as transfer
import torch

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"], nargs="?", default="predict")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--public-dir", type=Path, default=ROOT.parents[1] / "open_data")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_b = forecast.read_data(args.data_dir, "B")
    public_days, _, _ = transfer.read_public(args.public_dir, data_b)
    if args.action == "train":
        data = forecast.read_data(args.data_dir, "A")
        forecast.train(data, args.output_dir)
        inventory.train(data, args.output_dir)
        transfer.train(data, public_days, args.output_dir)
    # Check every checkpoint and the public-day protocol before writing outputs.
    for name in ("neural_ensemble.pt", "reference.npz", "inventory_ensemble.pt", "public_inventory.pt"):
        if not (args.output_dir / name).is_file():
            raise FileNotFoundError(f"Missing {name}: run the train command first.")
    saved = torch.load(args.output_dir / "public_inventory.pt", map_location="cpu", weights_only=True)
    if saved.get("model_type") != "public_marginal_inventory_v1" or saved["public_days"] != public_days.tolist():
        raise ValueError("Public days changed; retrain before generating the final submission.")
    forecast.export(data_b, args.output_dir)
    inventory.export(data_b, args.output_dir)
    transfer.export(data_b, args.public_dir, args.output_dir)


if __name__ == "__main__":
    main()
