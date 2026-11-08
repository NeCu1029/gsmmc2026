"""Adapt the neural marginal-cost model with officially revealed B station days."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import bike_neural_final as forecast
import bike_inventory_net as inventory
import numpy as np
import pandas as pd
import torch
from neural_allocation import scenario_costs, optimize, realized_score

ROOT = Path(__file__).resolve().parent
WEIGHTS = (0.0, 0.5, 1.0)


def read_public(folder, data):
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No official public-day CSV in {folder}")
    frames, provenance = [], []
    for file in files:
        frame = pd.read_csv(file, encoding="utf-8-sig").rename(columns={"날짜": "date", "시": "hour"})
        frame = frame[["date", "hour", "station_id", "rentals", "returns"]]
        frame["date"] = pd.to_datetime(frame.date)
        frames.append(frame)
        provenance.append({"file": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    frame = pd.concat(frames, ignore_index=True)
    days = data.city.dates.get_indexer(sorted(frame.date.unique()))
    if (days < 0).any():
        raise ValueError("Public dates outside evaluation month")
    index = pd.MultiIndex.from_product([data.city.dates[days], data.city.stations, range(24)], names=["date", "station_id", "hour"])
    indexed = forecast.base.unique_index(frame, index.names)
    if len(indexed) != len(index) or len(indexed.index.difference(index)):
        raise ValueError("Public data must include all stations and all 24 hours per revealed day")
    values = indexed.loc[index, ["rentals", "returns"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Public counts must be nonnegative finite integers")
    truth = (values[:, 1] - values[:, 0]).reshape(len(days), len(data.city.stations), 24)
    return days, truth, provenance


def increments(truth, caps):
    costs = scenario_costs(truth[:, :, None, :].astype(float), np.ones(truth.shape[:2] + (1,)), caps)
    return np.diff(np.where(np.isfinite(costs), costs, 0), axis=-1)


def adjusted(predicted, public_predicted, public_actual, off, public_off, weekday_weight, offday_weight):
    result = predicted.copy()
    for kind, weight in ((False, weekday_weight), (True, offday_weight)):
        matched = public_off == kind
        if not matched.any():
            matched = np.ones(len(public_off), dtype=bool)
        residual = (public_actual[matched] - public_predicted[matched]).mean(0)
        result[off == kind] += weight * residual
    return result


def select(data, target, sites, days, public_days):
    past, validation = days[:-7], days[-7:]
    if not set(public_days).issubset(set(past)):
        raise ValueError("Simulated public days must precede inner validation")
    groups = np.array_split(sites, 3)
    candidates = list(itertools.product(WEIGHTS, repeat=2))
    records = {c: [] for c in candidates}
    epochs = []
    for i, dev in enumerate(groups):
        core = np.concatenate(groups[:i] + groups[i+1:])
        fitted = [inventory.fit(data, target, past, core, seed, validation=(validation, dev)) for seed in inventory.SEEDS]
        models = [m for m, _ in fitted]
        epochs.append([e for _, e in fitted])
        pred = inventory.ensemble(models, data, validation, dev)
        public_pred = inventory.ensemble(models, data, public_days, dev)
        for c in candidates:
            calibrated = adjusted(pred, public_pred, target[public_days][:, dev], data.city.off_day[validation], data.city.off_day[public_days], *c)
            alloc, _ = optimize(inventory.cost_table(calibrated, data.capacity[dev]), data.capacity[dev])
            records[c].append(realized_score(data.city.net_flow[validation][:, dev].numpy(), alloc, data.capacity[dev])["score"])
        print(f"  Public-transfer inner fold {i+1}/3 done", flush=True)
    scores = [{"weekday_weight": c[0], "offday_weight": c[1], "selection_score": sum(v)} for c, v in records.items()]
    best = min(scores, key=lambda x: x["selection_score"])
    return {**best, "epochs": np.median(epochs, axis=0).astype(int).tolist()}, scores


def train(data, public_days, output):
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    past, future = days[:-7], days[-7:]
    target = increments(data.city.net_flow.numpy(), data.capacity)
    order = np.random.default_rng(forecast.SEED).permutation(len(sites))
    groups = np.split(order, [i * (len(sites) // 5) for i in range(1, 5)])
    allocation = np.empty((len(future), len(sites)), dtype=np.int64)
    records = []
    for i, test in enumerate(groups):
        sources = np.concatenate(groups[i+1:] + groups[:i])
        print(f"Public-transfer outer fold {i+1}/5", flush=True)
        config, search = select(data, target, sources, past, public_days)
        models = [inventory.fit(data, target, past, sources, seed, epoch)[0] for seed, epoch in zip(inventory.SEEDS, config["epochs"])]
        pred = inventory.ensemble(models, data, future, test)
        pub = inventory.ensemble(models, data, public_days, test)
        calibrated = adjusted(pred, pub, target[public_days][:, test], data.city.off_day[future], data.city.off_day[public_days], config["weekday_weight"], config["offday_weight"])
        allocation[:, test], _ = optimize(inventory.cost_table(calibrated, data.capacity[test]), data.capacity[test])
        result = realized_score(data.city.net_flow[future][:, test].numpy(), allocation[:, test], data.capacity[test])
        records.append({"test_station_ids": data.city.stations[test].tolist(), "config": config, "search": search, "actual_score": result})
        print(json.dumps({"config": config, "actual_score": result}), flush=True)
        (output / "public_transfer_progress.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    config, search = select(data, target, order, past, public_days)
    models = [inventory.fit(data, target, days, sites, seed, epoch)[0] for seed, epoch in zip(inventory.SEEDS, config["epochs"])]
    torch.save({"model_type": "public_marginal_inventory_v1", "ns": data.similarity.shape[-1], "nt": inventory.day_features(data).shape[-1],
                "states": [m.state_dict() for m in models], "config": config, "public_days": public_days.tolist()}, output / "public_inventory.pt")
    report = {"protocol": "5 station folds. Train other stations on days 1-23. Reveal target stations on days 9/10 only. Test days 24-30; never used for fitting or calibration. Inner selection uses days 17-23 and matching public days.",
              "public_dates": data.city.dates[public_days].strftime("%Y-%m-%d").tolist(), "final_config": config, "final_search": search, "folds": records,
              "actual_score": realized_score(data.city.net_flow[future].numpy(), allocation, data.capacity)}
    (output / "public_transfer_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(allocation.T, index=data.city.stations, columns=[f"D{d+1:02d}" for d in future]).rename_axis("station_id").to_csv(output / "A_공개일활용_검증배치.csv", encoding="utf-8-sig")
    print(json.dumps({"final_config": config, "actual_score": report["actual_score"]}), flush=True)


def export(data, public_dir, output):
    public_days, truth, provenance = read_public(public_dir, data)
    saved = torch.load(output / "public_inventory.pt", map_location="cpu", weights_only=True)
    if saved.get("model_type") != "public_marginal_inventory_v1" or saved["public_days"] != public_days.tolist():
        raise ValueError("Public days changed; retrain to validate the updated transfer protocol")
    models = []
    for state in saved["states"]:
        model = inventory.InventoryNet(saved["ns"], saved["nt"])
        model.load_state_dict(state)
        models.append(model)
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    pred = inventory.ensemble(models, data, days, sites)
    actual_inc = increments(truth, data.capacity)
    config = saved["config"]
    calibrated = adjusted(pred, pred[public_days], actual_inc, data.city.off_day, data.city.off_day[public_days], config["weekday_weight"], config["offday_weight"])
    # Revealed days remain in the official 30-day score. Their exact costs are known.
    calibrated[public_days] = actual_inc
    allocation, relative = optimize(inventory.cost_table(calibrated, data.capacity), data.capacity)
    pd.DataFrame(allocation.T, index=data.city.stations, columns=[f"D{d+1:02d}" for d in days]).rename_axis("station_id").to_csv(output / forecast.ALLOCATION_NAME, encoding="utf-8-sig")
    report = json.loads((output / "B_summary.json").read_text(encoding="utf-8"))
    report.pop("estimated_scenario_score", None)
    report.update({"allocation_method": "public_calibrated_inventory_net", "N": int(allocation[0].sum()), "relative_objective": relative,
                   "public_config": config, "public_dates": data.city.dates[public_days].strftime("%Y-%m-%d").tolist(), "public_files": provenance,
                   "public_actual_score": realized_score(truth, allocation[public_days], data.capacity),
                   "note": "Public-day score is in-sample. Unrevealed B demand/score are unknown. Relative objective omits C(0); it is NOT an actual-score estimate."})
    (output / "B_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"], nargs="?", default="train")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1])
    parser.add_argument("--public-dir", type=Path, default=ROOT.parents[1] / "open_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data_b = forecast.read_data(args.data_dir, "B")
    if args.action == "train":
        public_days, _, _ = read_public(args.public_dir, data_b)
        train(forecast.read_data(args.data_dir, "A"), public_days, args.output_dir)
    export(data_b, args.public_dir, args.output_dir)
