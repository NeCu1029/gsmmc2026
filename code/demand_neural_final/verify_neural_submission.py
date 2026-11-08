"""Verify final CSV, held-out costs, and exact parity with the official scorer."""
import argparse
import importlib.util
import json
from pathlib import Path
import bike_neural_final as forecast
import bike_public_transfer as transfer
import bike_inventory_net as inventory
import numpy as np
import pandas as pd
from neural_allocation import optimize, realized_score

ROOT = Path(__file__).resolve().parent


def official_score(data, days, allocation, counts, official):
    labels = [f"D{d+1:02d}" for d in days]
    ids = data.city.stations
    truth = {"hours": list(range(24)), "stations": [{"id": s, "docks": int(c)} for s, c in zip(ids, data.capacity)],
             "days": {d: {"rent": {s: counts[i, j, :, 0].astype(int).tolist() for j, s in enumerate(ids)},
                           "ret": {s: counts[i, j, :, 1].astype(int).tolist() for j, s in enumerate(ids)}} for i, d in enumerate(labels)}}
    cfg = {"hours": list(range(24)), "eval_days": labels, "public_days": labels, "scored_days": labels,
           "equal_daily_total": True, "enforce_docks": True, "min_total": 1, "score_formula": official.GSMMC_FORMULA}
    alloc = {d: {s: int(x) for s, x in zip(ids, row)} for d, row in zip(labels, allocation)}
    errors = official.validate_alloc(alloc, truth, cfg)
    if errors:
        raise AssertionError(errors)
    measured = official.score_submission(alloc, None, truth, cfg)["all"]["S_avg"]
    own = realized_score(counts[..., 1] - counts[..., 0], allocation, data.capacity)["score"]
    assert abs(measured - own) < 0.01, (measured, own)
    return measured


def validate(data_dir, public_dir, output, baseline):
    spec = importlib.util.spec_from_file_location("official_score_module", data_dir / "score.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    a, b = forecast.read_data(data_dir, "A"), forecast.read_data(data_dir, "B")
    frame = pd.read_csv(output / forecast.ALLOCATION_NAME, encoding="utf-8-sig")
    assert list(frame) == ["station_id"] + [f"D{i:02d}" for i in range(1, 31)]
    assert frame.station_id.tolist() == b.city.stations.tolist()
    values = frame.iloc[:, 1:].to_numpy(float).T
    assert np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
    assert (values >= 0).all() and (values <= b.capacity).all()
    assert len(set(values.sum(1))) == 1 and values[0].sum() >= 1
    forecasts = pd.read_csv(output / forecast.FORECAST_NAME)
    expected = pd.MultiIndex.from_product([b.city.dates.strftime("%Y-%m-%d"), range(24), b.city.stations], names=["date", "hour", "station_id"])
    indexed = forecast.base.unique_index(forecasts, expected.names)
    assert len(indexed) == len(expected) == 114480
    assert not len(indexed.index.difference(expected)) and np.isfinite(indexed.pred_net_flow).all()
    days = np.arange(23, 30)
    actual = a.city.net_flow[days].numpy()
    report = {"B_submission": {"stations": len(frame), "days": 30, "N": int(values[0].sum()), "constraints": "PASS"},
              "A_official_scores": {}, "public_day_scores_are_in_sample": True}
    for name in ("A_검증배치.csv", "A_최종검증배치.csv", "A_공개일활용_검증배치.csv"):
        allocation = pd.read_csv(output / name, index_col="station_id").loc[a.city.stations].to_numpy().T
        report["A_official_scores"][name] = official_score(a, days, allocation, a.counts[days].numpy(), official)
    final_report = json.loads((output / "public_transfer_validation.json").read_text(encoding="utf-8"))
    assert abs(report["A_official_scores"]["A_공개일활용_검증배치.csv"] - final_report["actual_score"]["score"]) < 0.01
    public_days, public_truth, _ = transfer.read_public(public_dir, b)
    # The official recurrence depends on net flow only. These equivalent nonnegative
    # count pairs retain exactly the observed net flow for an independent scorer check.
    counts = np.stack([np.maximum(-public_truth, 0), np.maximum(public_truth, 0)], axis=-1)
    report["B_public_actual_score"] = official_score(b, public_days, values[public_days], counts, official)
    # A two-day empirical baseline uses exactly the same revealed station-day labels.
    raw_inc = transfer.increments(a.city.net_flow[public_days].numpy(), a.capacity).mean(0)
    costs = inventory.cost_table(np.broadcast_to(raw_inc, (len(days),) + raw_inc.shape), a.capacity)
    empirical = np.empty(actual.shape[:2], dtype=np.int64)
    for fold in final_report["folds"]:
        sites = a.city.stations.get_indexer(fold["test_station_ids"])
        empirical[:, sites], _ = optimize(costs[:, sites], a.capacity[sites])
    report["A_two_public_day_empirical_baseline"] = official_score(a, days, empirical, a.counts[days].numpy(), official)
    if baseline is not None:
        keys = pd.MultiIndex.from_product([a.city.dates[days].strftime("%Y-%m-%d"), a.city.stations, range(24)], names=["date", "station_id", "hour"])
        old = pd.read_csv(baseline).set_index(keys.names).loc[keys]
        pred = old.pred_net_flow.to_numpy().reshape(actual.shape)
        old_costs = forecast.scenario_costs(pred[:, :, None, :], np.ones(pred.shape[:2] + (1,)), a.capacity)
        old_alloc = np.empty(actual.shape[:2], dtype=np.int64)
        for fold in final_report["folds"]:
            sites = a.city.stations.get_indexer(fold["test_station_ids"])
            old_alloc[:, sites], _ = optimize(old_costs[:, sites], a.capacity[sites])
        report["A_old_GNN_LSTM_cold_start_score"] = official_score(a, days, old_alloc, a.counts[days].numpy(), official)
        report["comparison_note"] = "Old GNN+LSTM has zero target-station labels; public-transfer model has two labelled days. This is not an architecture-only comparison."
    (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1])
    parser.add_argument("--public-dir", type=Path, default=ROOT.parents[1] / "open_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    validate(args.data_dir, args.public_dir, args.output_dir, args.baseline)
