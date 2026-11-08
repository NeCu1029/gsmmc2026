"""Learn the expected marginal penalty of each additional midnight bicycle."""
import argparse
import json
from pathlib import Path
import bike_neural_final as forecast
import numpy as np
import pandas as pd
import torch
from torch import nn
from neural_allocation import scenario_costs, optimize, realized_score

ROOT = Path(__file__).resolve().parent
SEEDS = (202616, 202617)


def day_features(data):
    t = data.city.temporal
    return torch.cat([t[:, 0, 8:18], t[:, :, 18:22].flatten(1), t[:, 0, 22:]], dim=1)


class InventoryNet(nn.Module):
    def __init__(self, ns, nt):
        super().__init__()
        self.register_buffer("s_mean", torch.zeros(ns))
        self.register_buffer("s_scale", torch.ones(ns))
        self.register_buffer("t_mean", torch.zeros(nt))
        self.register_buffer("t_scale", torch.ones(nt))
        self.station = nn.Sequential(nn.Linear(ns, 24), nn.SiLU())
        self.weather = nn.Sequential(nn.Linear(nt, 24), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(48, 48), nn.SiLU(), nn.Dropout(0.1), nn.Linear(48, 32), nn.SiLU())
        self.value = nn.Sequential(nn.Linear(35, 32), nn.SiLU(), nn.Linear(32, 1))

    def fit_scales(self, data, days, sites):
        for x, mean, scale in [(forecast.base.tensor(data.similarity[sites]), self.s_mean, self.s_scale),
                               (day_features(data)[days], self.t_mean, self.t_scale)]:
            mean.copy_(x.mean(0))
            scale.copy_(x.std(0, unbiased=False).clamp_min(0.2))

    def forward(self, data, days, sites):
        site_indices = sites.detach().cpu().numpy()
        s = forecast.base.tensor(data.similarity[site_indices])
        s = ((s - self.s_mean) / self.s_scale).clamp(-5, 5)
        t = ((day_features(data)[days] - self.t_mean) / self.t_scale).clamp(-5, 5)
        context = self.context(torch.cat([self.station(s), self.weather(t)], dim=1))
        cap = forecast.base.tensor(data.capacity[site_indices])[:, None]
        x = torch.arange(int(data.capacity.max()), dtype=torch.float32)[None].expand(len(days), -1)
        stock = torch.stack([x / cap, torch.log1p(x) / 3, (x >= torch.floor(cap / 2)).float()], dim=-1)
        embedding = context[:, None, :].expand(-1, x.shape[1], -1)
        return self.value(torch.cat([embedding, stock], dim=-1)).squeeze(-1) * 20


@torch.no_grad()
def predict(model, data, days, sites):
    model.eval()
    pairs = forecast.base.examples(days, sites)
    result = [model(data, p[:, 0], p[:, 1]).numpy() for p in pairs.split(256)]
    return np.concatenate(result).reshape(len(days), len(sites), -1).astype(float)


def cost_table(increments, capacities):
    costs = np.concatenate([np.zeros(increments.shape[:2] + (1,)), increments.cumsum(-1)], axis=-1)
    costs = np.where(np.arange(costs.shape[-1])[None, None, :] <= capacities[None, :, None], costs, np.inf)
    return costs


def fit(data, target, days, sites, seed, epochs=40, validation=None):
    torch.manual_seed(seed)
    model = InventoryNet(data.similarity.shape[-1], day_features(data).shape[-1])
    model.fit_scales(data, days, sites)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    pairs = forecast.base.examples(days, sites)
    target = forecast.base.tensor(target)
    level = torch.arange(target.shape[-1])[None]
    best, count, state = float("inf"), 1, None
    for epoch in range(1, epochs + 1):
        model.train()
        for order in torch.randperm(len(pairs)).split(128):
            d, s = pairs[order].unbind(1)
            predicted = model(data, d, s)
            mask = level < torch.as_tensor(data.capacity[s.numpy()])[:, None]
            loss = ((predicted - target[d, s])[mask] / 20).square().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        if validation is not None:
            vd, vs = validation
            error = predict(model, data, vd, vs) - target[vd][:, vs].numpy()
            mask = np.broadcast_to(np.arange(error.shape[-1])[None, None, :] < data.capacity[vs][None, :, None], error.shape)
            value = float(np.mean(error[mask] ** 2))
            if value < best - 0.01:
                best, count = value, epoch
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch - count >= 6:
                break
    if validation is not None:
        model.load_state_dict(state)
    return model, count if validation is not None else epochs


def ensemble(models, data, days, sites):
    return np.mean([predict(m, data, days, sites) for m in models], axis=0)


def select(data, target, sites, days):
    past, validation = days[:-7], days[-7:]
    groups = np.array_split(sites, 3)
    records = []
    for i, dev in enumerate(groups):
        core = np.concatenate(groups[:i] + groups[i+1:])
        models, epochs = [], []
        for seed in SEEDS:
            model, epoch = fit(data, target, past, core, seed, validation=(validation, dev))
            models.append(model)
            epochs.append(epoch)
        increments = ensemble(models, data, validation, dev)
        allocation, _ = optimize(cost_table(increments, data.capacity[dev]), data.capacity[dev])
        records.append({"epochs": epochs, "actual_score": realized_score(data.city.net_flow[validation][:, dev].numpy(), allocation, data.capacity[dev])})
        print(f"  Inventory inner fold {i+1}/3 done", flush=True)
    return {"epochs": [int(np.median([r["epochs"][j] for r in records])) for j in range(len(SEEDS))],
            "selection_score": sum(r["actual_score"]["score"] for r in records), "inner_folds": records}


def train(data, output):
    city = data.city
    days, sites = np.arange(len(city.dates)), np.arange(len(city.stations))
    past, future = days[:-7], days[-7:]
    full_cost = scenario_costs(city.net_flow.numpy().astype(float)[:, :, None, :], np.ones(city.net_flow.shape[:2] + (1,)), data.capacity)
    # Cost at stock zero is independent of the choice. Learn only C(x+1)-C(x).
    finite = np.where(np.isfinite(full_cost), full_cost, 0)
    increments = np.diff(finite, axis=-1)
    order = np.random.default_rng(forecast.SEED).permutation(len(sites))
    groups = np.split(order, [i * (len(sites) // 5) for i in range(1, 5)])
    forecast_report = json.loads((output / "A_validation.json").read_text(encoding="utf-8"))
    forecast_alloc = pd.read_csv(output / "A_검증배치.csv", index_col="station_id").loc[city.stations].to_numpy().T
    chosen_alloc = np.empty_like(forecast_alloc)
    inventory_alloc = np.empty_like(forecast_alloc)
    records = []
    for i, test in enumerate(groups):
        sources = np.concatenate(groups[i+1:] + groups[:i])
        print(f"Inventory outer fold {i+1}/5", flush=True)
        config = select(data, increments, sources, past)
        models = [fit(data, increments, past, sources, seed, epoch)[0] for seed, epoch in zip(SEEDS, config["epochs"])]
        predicted = ensemble(models, data, future, test)
        allocation, _ = optimize(cost_table(predicted, data.capacity[test]), data.capacity[test])
        previous_score = min(x["selection_score"] for x in forecast_report["folds"][i]["inner_search"])
        selected = "inventory_net" if config["selection_score"] < previous_score else "forecast_scenarios"
        inventory_alloc[:, test] = allocation
        chosen_alloc[:, test] = allocation if selected == "inventory_net" else forecast_alloc[:, test]
        records.append({"test_station_ids": city.stations[test].tolist(), "selected": selected, "inventory_config": config,
                        "forecast_selection_score": previous_score,
                        "actual_inventory_score": realized_score(city.net_flow[future][:, test].numpy(), allocation, data.capacity[test])})
        print(json.dumps({"selected": selected, "score": records[-1]["actual_inventory_score"]}), flush=True)
        (output / "inventory_progress.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    final = select(data, increments, order, past)
    old_score = min(x["selection_score"] for x in forecast_report["final_search"])
    selected = "inventory_net" if final["selection_score"] < old_score else "forecast_scenarios"
    models = [fit(data, increments, days, sites, seed, epoch)[0] for seed, epoch in zip(SEEDS, final["epochs"])]
    torch.save({"model_type": "marginal_inventory_v1", "ns": data.similarity.shape[-1], "nt": day_features(data).shape[-1],
                "states": [m.state_dict() for m in models], "config": final, "selected": selected}, output / "inventory_ensemble.pt")
    actual = city.net_flow[future].numpy()
    report = {"selected": selected, "final_inventory_config": final, "forecast_selection_score": old_score, "folds": records,
              "selected_actual_score": realized_score(actual, chosen_alloc, data.capacity),
              "inventory_only_actual_score": realized_score(actual, inventory_alloc, data.capacity),
              "forecast_scenarios_actual_score": forecast_report["actual_allocation"]}
    (output / "inventory_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(chosen_alloc.T, index=city.stations, columns=[f"D{d+1:02d}" for d in future]).rename_axis("station_id").to_csv(output / "A_최종검증배치.csv", encoding="utf-8-sig")
    print(json.dumps({k: v for k, v in report.items() if k not in ("folds", "final_inventory_config")}), flush=True)


def export(data, output):
    saved = torch.load(output / "inventory_ensemble.pt", map_location="cpu", weights_only=True)
    if saved.get("model_type") != "marginal_inventory_v1":
        raise ValueError("Incorrect inventory model")
    if saved["selected"] == "forecast_scenarios":
        forecast.export(data, output)
    else:
        models = []
        for state in saved["states"]:
            model = InventoryNet(saved["ns"], saved["nt"])
            model.load_state_dict(state)
            models.append(model)
        days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
        allocation, relative = optimize(cost_table(ensemble(models, data, days, sites), data.capacity), data.capacity)
        pd.DataFrame(allocation.T, index=data.city.stations, columns=[f"D{d+1:02d}" for d in days]).rename_axis("station_id").to_csv(output / forecast.ALLOCATION_NAME, encoding="utf-8-sig")
        report = json.loads((output / "B_summary.json").read_text(encoding="utf-8"))
        report.pop("estimated_scenario_score", None)
        report.update({"allocation_method": "inventory_net", "N": int(allocation[0].sum()),
                       "relative_objective": relative, "note": "Relative objective omits C(0); it is NOT an estimate of actual B score."})
        (output / "B_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Final allocation method: {saved['selected']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"], nargs="?", default="train")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.action == "train":
        train(forecast.read_data(args.data_dir, "A"), args.output_dir)
    export(forecast.read_data(args.data_dir, "B"), args.output_dir)
