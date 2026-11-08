"""Neural mean forecasts, empirical daily uncertainty, and exact integer allocation.

python bike_neural_final.py train
python bike_neural_final.py predict
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for name in (".ml-runtime", ".decision-runtime"):
    if (ROOT / name).is_dir():
        sys.path.insert(0, str(ROOT / name))

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import bike_net_flow as base
from neural_allocation import scenario_costs, optimize, realized_score

SEED = 202606
SEEDS = (202606, 202607)
ARCHITECTURES = ("graph_mlp", "graph_tcn")
RISKS = (0.0, 0.5, 1.0)
FORECAST_NAME = "B지역_시간별_순유입_예측.csv"
ALLOCATION_NAME = "최종제출_배치표_B지역.csv"


@dataclass
class Data:
    city: base.City
    counts: torch.Tensor | None
    capacity: np.ndarray
    similarity: np.ndarray
    day_weather: np.ndarray


def read_data(folder, region):
    path = folder / ("learn_data" if region == "A" else "input_data")
    city = base.read_city(path, region == "A")
    capacity = pd.read_csv(path / "spots.csv", encoding="utf-8-sig", float_precision="round_trip").set_index("station_id").loc[city.stations, "capacity"].to_numpy(np.int64)
    counts = None
    if region == "A":
        frame = pd.read_csv(path / "demand.csv", encoding="utf-8-sig", float_precision="round_trip")
        frame["date"] = pd.to_datetime(frame.date)
        index = pd.MultiIndex.from_product([city.dates, city.stations, range(24)], names=["date", "station_id", "hour"])
        counts = base.tensor(frame.set_index(index.names).loc[index, ["rentals", "returns"]].to_numpy().reshape(len(city.dates), len(city.stations), 24, 2))
    weather = city.temporal[:, :, -4:].numpy()
    daily = np.column_stack([weather[:, :, 0].mean(1), weather[:, :, 0].max(1),
                             np.log1p(np.expm1(weather[:, :, 2]).sum(1)), weather[:, :, 1].mean(1), weather[:, :, 3].mean(1)])
    extra = base.tensor(daily)[:, None, :].expand(-1, 24, -1)
    city.temporal = torch.cat([city.temporal, extra], dim=-1)
    similarity = torch.cat([city.static, city.adjacency @ city.static], dim=-1).numpy().astype(float)
    return Data(city, counts, capacity, similarity, daily)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                 nn.Dropout(0.1), nn.Linear(width, width))

    def forward(self, x):
        return x + 0.5 * self.net(x)


class ForecastNet(nn.Module):
    def __init__(self, ns, nt, architecture):
        super().__init__()
        self.architecture = architecture
        self.register_buffer("s_mean", torch.zeros(ns))
        self.register_buffer("s_scale", torch.ones(ns))
        self.register_buffer("t_mean", torch.zeros(nt))
        self.register_buffer("t_scale", torch.ones(nt))
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_scale", torch.tensor(1.0))
        self.graph_self = nn.Linear(ns, 24)
        self.graph_neighbors = nn.Linear(ns, 24, bias=False)
        self.join = nn.Sequential(nn.Linear(24 + ns + nt, 48), nn.SiLU())
        self.blocks = nn.Sequential(ResidualBlock(48), ResidualBlock(48))
        self.convs = nn.ModuleList([nn.Conv1d(48, 48, 3, padding=d, dilation=d) for d in (1, 2)]) if architecture == "graph_tcn" else nn.ModuleList()
        self.head = nn.Linear(48, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def fit_scales(self, data, days, sites):
        city = data.city
        for x, mean, scale in [(city.static[sites], self.s_mean, self.s_scale),
                               (city.temporal[days].flatten(0, 1), self.t_mean, self.t_scale)]:
            mean.copy_(x.mean(0))
            scale.copy_(x.std(0, unbiased=False).clamp_min(0.2))
        y = city.net_flow[days][:, sites]
        self.y_mean.copy_(y.mean())
        self.y_scale.copy_(y.std(unbiased=False).clamp_min(1.0))
        with torch.no_grad():
            self.head.bias[1:].copy_(data.counts[days][:, sites].mean((0, 1, 2)).clamp_min(0.1).log())

    def forward(self, data, days, sites):
        city = data.city
        static = ((city.static - self.s_mean) / self.s_scale).clamp(-5, 5)
        graph = F.silu(self.graph_self(static) + self.graph_neighbors(city.adjacency @ static))
        station = torch.cat([static[sites], graph[sites]], dim=-1)[:, None, :].expand(-1, 24, -1)
        time = ((city.temporal[days] - self.t_mean) / self.t_scale).clamp(-5, 5)
        x = self.blocks(self.join(torch.cat([station, time], dim=-1)))
        for conv in self.convs:
            x = x + 0.5 * F.silu(conv(x.transpose(1, 2))).transpose(1, 2)
        out = self.head(x)
        return out[:, :, 0] * self.y_scale + self.y_mean, out[:, :, 1:].clamp(-8, 8)


def loss(pred, log_counts, actual, counts, scale):
    error = (pred - actual) / scale
    hourly = error.square().mean()
    # A path with persistent errors harms inventory. sqrt(h) keeps its scale comparable.
    hours = torch.arange(1, 25, device=error.device, dtype=error.dtype).sqrt()
    cumulative = (error.cumsum(-1) / hours).square().mean()
    rates = log_counts.exp()
    poisson = (rates - counts * log_counts).mean()
    consistency = ((pred - (rates[:, :, 1] - rates[:, :, 0])) / scale).square().mean()
    return hourly + 0.05 * cumulative + 0.05 * poisson + 0.02 * consistency


@torch.no_grad()
def predict_one(model, data, days, sites):
    model.eval()
    pairs = base.examples(days, sites)
    values = [model(data, batch[:, 0], batch[:, 1])[0] for batch in pairs.split(256)]
    return torch.cat(values).reshape(len(days), len(sites), 24).numpy().astype(float)


def fit(data, days, sites, architecture, seed, epochs=40, dev=None):
    torch.manual_seed(seed)
    model = ForecastNet(data.city.static.shape[-1], data.city.temporal.shape[-1], architecture)
    model.fit_scales(data, days, sites)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    pairs = base.examples(days, sites)
    best, selected, state = float("inf"), 1, None
    for epoch in range(1, epochs + 1):
        model.train()
        for index in torch.randperm(len(pairs)).split(128):
            d, s = pairs[index].unbind(1)
            prediction, log_counts = model(data, d, s)
            objective = loss(prediction, log_counts, data.city.net_flow[d, s], data.counts[d, s], model.y_scale)
            optimizer.zero_grad()
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        if dev is not None:
            vd, vs = dev
            error = predict_one(model, data, vd, vs) - data.city.net_flow[vd][:, vs].numpy()
            criterion = float(np.mean(error ** 2))
            if criterion < best - 0.001:
                best, selected = criterion, epoch
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch - selected >= 6:
                break
    if dev is not None:
        model.load_state_dict(state)
    return model, selected if dev is not None else epochs


def ensemble(models, data, days, sites):
    return np.mean([predict_one(model, data, days, sites) for model in models], axis=0)


def make_reference(data, days, sites):
    return {"static": data.similarity[sites], "weather": data.day_weather[days],
            "off": data.city.off_day[days], "capacity": data.capacity[sites],
            "net": data.city.net_flow[days][:, sites].numpy().astype(float)}


def analogs(ref, data, days, sites):
    delta = (data.similarity[sites, None] - ref["static"][None]) / np.maximum(ref["static"].std(0), 0.25)
    delta[:, :, 0] *= 2
    distance = np.mean(delta ** 2, axis=-1)
    k = min(32, len(ref["static"]))
    near = np.argsort(distance, axis=1, kind="stable")[:, :k]
    chosen_distance = np.take_along_axis(distance, near, axis=1)
    site_weights = np.exp(-chosen_distance / (2 * np.maximum(np.median(chosen_distance, axis=1, keepdims=True), 1e-6)))
    site_weights /= site_weights.sum(1, keepdims=True)
    scale = np.clip(np.sqrt(data.capacity[sites, None] / ref["capacity"][near]), 0.5, 2)
    wscale = np.maximum(ref["weather"].std(0), [1, 1, 0.5, 5, 1])
    paths, weights = [], []
    for day in days:
        matched = np.flatnonzero(ref["off"] == data.city.off_day[day])
        if not len(matched):
            matched = np.arange(len(ref["off"]))
        wd = np.mean(((ref["weather"][matched] - data.day_weather[day]) / wscale) ** 2, axis=1)
        best_days = np.resize(matched[np.argsort(wd, kind="stable")[:3]], 3)
        wd = np.mean(((ref["weather"][best_days] - data.day_weather[day]) / wscale) ** 2, axis=1)
        day_weights = np.exp(-wd / 2)
        day_weights /= np.array([(best_days == d).sum() for d in best_days])
        day_weights /= day_weights.sum()
        p = ref["net"][best_days[:, None, None], near[None], :].transpose(1, 0, 2, 3)
        p = (p * scale[:, None, :, None]).reshape(len(sites), 3 * k, 24)
        w = (site_weights[:, None, :] * day_weights[None, :, None]).reshape(len(sites), 3 * k)
        paths.append(p)
        weights.append(w)
    return np.array(paths), np.array(weights)


def allocation(mean, paths, weights, risk, capacities):
    if risk == 0:
        scenarios, mass = mean[:, :, None, :], np.ones(mean.shape[:2] + (1,))
    else:
        center = (paths * weights[:, :, :, None]).sum(2)
        scenarios = mean[:, :, None, :] + risk * (paths - center[:, :, None, :])
        mass = weights
    return optimize(scenario_costs(scenarios, mass, capacities), capacities)


def metrics(actual, predicted):
    error = predicted - actual
    return {"station_hours": int(error.size), "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean()),
            "daily_cumulative_MAE": float(np.abs(error.cumsum(-1)).mean())}


def select(data, sites, days):
    past, validation = days[:-7], days[-7:]
    groups = np.array_split(sites, 3)
    records = {}
    for i, dev in enumerate(groups):
        core = np.concatenate(groups[:i] + groups[i+1:])
        paths, weights = analogs(make_reference(data, past, core), data, validation, dev)
        actual = data.city.net_flow[validation][:, dev].numpy()
        for architecture in ARCHITECTURES:
            models, epochs = [], []
            for seed in SEEDS:
                model, count = fit(data, past, core, architecture, seed, dev=(validation, dev))
                models.append(model)
                epochs.append(count)
            mean = ensemble(models, data, validation, dev)
            for risk in RISKS:
                allocated, _ = allocation(mean, paths, weights, risk, data.capacity[dev])
                result = realized_score(actual, allocated, data.capacity[dev])
                records.setdefault((architecture, risk), []).append({"epochs": epochs, "allocation": result,
                                                                     "forecast": metrics(actual, mean)})
        print(f"  Inner fold {i+1}/3 done", flush=True)
    options = []
    for (architecture, risk), folds in records.items():
        options.append({"architecture": architecture, "risk": risk,
                        "epochs": [int(np.median([f["epochs"][j] for f in folds])) for j in range(len(SEEDS))],
                        "selection_score": sum(f["allocation"]["score"] for f in folds), "inner_folds": folds})
    chosen = min(options, key=lambda x: x["selection_score"])
    return {k: chosen[k] for k in ("architecture", "risk", "epochs")}, options


def train_models(data, days, sites, config):
    return [fit(data, days, sites, config["architecture"], seed, epochs=epochs)[0]
            for seed, epochs in zip(SEEDS, config["epochs"])]


def save_csv(data, days, predictions, file, actual=None):
    index = pd.MultiIndex.from_product([data.city.dates[days], data.city.stations, range(24)], names=["date", "station_id", "hour"])
    frame = pd.DataFrame({"pred_net_flow": predictions.ravel()}, index=index).reset_index()
    if actual is not None:
        frame["actual_net_flow"] = actual.ravel()
    frame = frame[["date", "hour", "station_id"] + [x for x in frame if x not in ("date", "hour", "station_id")]]
    frame.sort_values(["date", "hour", "station_id"]).to_csv(file, index=False, encoding="utf-8-sig", float_format="%.6f")


def train(data, output):
    city = data.city
    days, sites = np.arange(len(city.dates)), np.arange(len(city.stations))
    past, future = days[:-7], days[-7:]
    order = np.random.default_rng(SEED).permutation(len(sites))
    groups = np.split(order, [i * (len(sites) // 5) for i in range(1, 5)])
    all_pred = np.empty((7, len(sites), 24))
    allocated = np.empty((7, len(sites)), dtype=np.int64)
    report = {"seed": SEED, "seeds": list(SEEDS), "outer_test_dates": city.dates[future].strftime("%Y-%m-%d").tolist(),
              "validation": "5 outer station folds; train days 1-23, test days 24-30. 3 inner station folds: fit days 1-16, choose days 17-23.", "folds": []}
    for i, test in enumerate(groups):
        sources = np.concatenate(groups[i+1:] + groups[:i])
        print(f"Outer fold {i+1}/5", flush=True)
        config, options = select(data, sources, past)
        models = train_models(data, past, sources, config)
        mean = ensemble(models, data, future, test)
        paths, weights = analogs(make_reference(data, past, sources), data, future, test)
        alloc, expected = allocation(mean, paths, weights, config["risk"], data.capacity[test])
        all_pred[:, test] = mean
        allocated[:, test] = alloc
        entry = {"test_station_ids": city.stations[test].tolist(), "config": config, "inner_search": options,
                 "forecast": metrics(city.net_flow[future][:, test].numpy(), mean),
                 "actual_allocation": realized_score(city.net_flow[future][:, test].numpy(), alloc, data.capacity[test])}
        report["folds"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k not in ("inner_search", "test_station_ids")}), flush=True)
        (output / "validation_progress.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    actual = city.net_flow[future].numpy()
    report["forecast"] = metrics(actual, all_pred)
    report["actual_allocation"] = realized_score(actual, allocated, data.capacity)
    # Also report common-fleet pooled OOF allocation, comparable with the previous mean model.
    point, _ = allocation(all_pred, None, None, 0, data.capacity)
    report["pooled_point_allocation"] = realized_score(actual, point, data.capacity)
    report["baselines"] = {"one_each": realized_score(actual, np.ones(allocated.shape, dtype=np.int64), data.capacity),
                           "half_capacity": realized_score(actual, np.broadcast_to(data.capacity // 2, allocated.shape), data.capacity),
                           "zero_forecast": metrics(actual, np.zeros_like(actual))}
    save_csv(data, future, all_pred, output / "A_검증예측.csv", actual)
    pd.DataFrame(allocated.T, index=city.stations, columns=[f"D{d+1:02d}" for d in future]).rename_axis("station_id").to_csv(output / "A_검증배치.csv", encoding="utf-8-sig")
    print("Selecting final configuration on days 1-23 only", flush=True)
    config, options = select(data, order, past)
    models = train_models(data, days, sites, config)
    checkpoint = {"model_type": "neural_daily_scenarios_v1", "config": config, "ns": city.static.shape[-1],
                  "nt": city.temporal.shape[-1], "states": [m.state_dict() for m in models], "seeds": list(SEEDS),
                  "target": "returns - rentals", "loss": "symmetric MSE + 0.05 cumulative MSE + 0.05 Poisson + 0.02 consistency MSE"}
    torch.save(checkpoint, output / "neural_ensemble.pt")
    np.savez_compressed(output / "reference.npz", **make_reference(data, days, sites))
    report["final_config"], report["final_search"] = config, options
    (output / "A_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"final_config": config, "forecast": report["forecast"], "actual_allocation": report["actual_allocation"], "pooled_point_allocation": report["pooled_point_allocation"]}), flush=True)


def export(data, output):
    saved = torch.load(output / "neural_ensemble.pt", map_location="cpu", weights_only=True)
    if saved.get("model_type") != "neural_daily_scenarios_v1" or saved.get("target") != "returns - rentals":
        raise ValueError("Incompatible neural model")
    models = []
    for state in saved["states"]:
        model = ForecastNet(saved["ns"], saved["nt"], saved["config"]["architecture"])
        model.load_state_dict(state)
        models.append(model)
    with np.load(output / "reference.npz", allow_pickle=False) as archive:
        ref = {k: archive[k] for k in archive.files}
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    mean = ensemble(models, data, days, sites)
    if not np.isfinite(mean).all():
        raise ValueError("Non-finite forecasts")
    paths, weights = analogs(ref, data, days, sites)
    alloc, estimated = allocation(mean, paths, weights, saved["config"]["risk"], data.capacity)
    save_csv(data, days, mean, output / FORECAST_NAME)
    pd.DataFrame(alloc.T, index=data.city.stations, columns=[f"D{d+1:02d}" for d in days]).rename_axis("station_id").to_csv(output / ALLOCATION_NAME, encoding="utf-8-sig")
    summary = {"model_config": saved["config"], "stations": len(sites), "days": len(days), "station_hours": int(mean.size),
               "N": int(alloc[0].sum()), "estimated_scenario_score": estimated,
               "mean_pred_net_flow": float(mean.mean()), "negative_fraction": float((mean < 0).mean()),
               "note": "B actual demand and actual score are unknown."}
    (output / "B_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"], nargs="?", default="train")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "train":
        train(read_data(args.data_dir, "A"), args.output_dir)
    export(read_data(args.data_dir, "B"), args.output_dir)
