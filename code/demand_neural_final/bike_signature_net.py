"""Public-day signatures as neural inputs, with explicit held-out station validation."""
import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import bike_neural_final as base
import bike_inventory_net as inventory
import bike_public_transfer as transfer
import numpy as np
import pandas as pd
import torch
from torch import nn
from neural_allocation import optimize, realized_score

ROOT = Path(__file__).resolve().parent
SEEDS = (202626, 202627)
BLENDS = (0.0, 0.5, 1.0)
MODEL = "signature_inventory.pt"
SUBMISSION = "최종제출_배치표_B지역_개선2.csv"


@dataclass
class Inputs:
    source: base.Data
    signature: torch.Tensor
    anchor: torch.Tensor
    spread: torch.Tensor


def make_inputs(data, public_net):
    # All target-region observations enter through these explicitly public paths.
    caps = data.capacity[:, None]
    mean, std = public_net.mean(0), public_net.std(0)
    profile = np.concatenate([np.arcsinh(mean / caps), np.log1p(std / caps),
                              np.arcsinh(mean.cumsum(-1) / caps)], axis=-1)
    signature = np.concatenate([data.similarity, profile], axis=-1)
    inc = transfer.increments(public_net, data.capacity)
    return Inputs(data, base.base.tensor(signature), base.base.tensor(inc.mean(0)), base.base.tensor(inc.std(0)))


class SignatureNet(nn.Module):
    def __init__(self, ns, nt):
        super().__init__()
        for prefix, n in (("s", ns), ("t", nt)):
            self.register_buffer(prefix + "_mean", torch.zeros(n))
            self.register_buffer(prefix + "_scale", torch.ones(n))
        self.station = nn.Sequential(nn.Linear(ns, 32), nn.SiLU(), nn.Dropout(0.1))
        self.weather = nn.Sequential(nn.Linear(nt, 16), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(48, 48), nn.SiLU(), nn.Dropout(0.1), nn.Linear(48, 32), nn.SiLU())
        self.value = nn.Sequential(nn.Linear(37, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def fit_scales(self, inputs, days, sites):
        for x, prefix in ((inputs.signature[sites], "s"), (inventory.day_features(inputs.source)[days], "t")):
            getattr(self, prefix + "_mean").copy_(x.mean(0))
            getattr(self, prefix + "_scale").copy_(x.std(0, unbiased=False).clamp_min(0.2))

    def forward(self, inputs, days, sites):
        s = ((inputs.signature[sites] - self.s_mean) / self.s_scale).clamp(-5, 5)
        t = ((inventory.day_features(inputs.source)[days] - self.t_mean) / self.t_scale).clamp(-5, 5)
        context = self.context(torch.cat([self.station(s), self.weather(t)], dim=-1))
        anchor, spread = inputs.anchor[sites], inputs.spread[sites]
        cap = base.base.tensor(inputs.source.capacity[sites.numpy()])[:, None]
        x = torch.arange(anchor.shape[-1], dtype=torch.float32)[None].expand(len(days), -1)
        stock = torch.stack([x / cap, torch.log1p(x) / 3, (x >= torch.floor(cap / 2)).float(),
                             torch.asinh(anchor / 20), torch.log1p(spread / 20)], dim=-1)
        context = context[:, None].expand(-1, x.shape[1], -1)
        return anchor + 20 * self.value(torch.cat([context, stock], dim=-1)).squeeze(-1)


@torch.no_grad()
def predict(model, inputs, days, sites):
    model.eval()
    pairs = base.base.examples(days, sites)
    arrays = [model(inputs, b[:, 0], b[:, 1]).numpy() for b in pairs.split(256)]
    return np.concatenate(arrays).reshape(len(days), len(sites), -1).astype(float)


def ensemble(models, inputs, days, sites):
    return np.mean([predict(m, inputs, days, sites) for m in models], axis=0)


def fit(inputs, target, days, sites, seed, epochs=45, validation=None):
    torch.manual_seed(seed)
    model = SignatureNet(inputs.signature.shape[-1], inventory.day_features(inputs.source).shape[-1])
    model.fit_scales(inputs, days, sites)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    pairs = base.base.examples(days, sites)
    target = base.base.tensor(target)
    level = torch.arange(target.shape[-1])[None]
    best, selected, state = float("inf"), 1, None
    for epoch in range(1, epochs + 1):
        model.train()
        for order in torch.randperm(len(pairs)).split(128):
            d, s = pairs[order].unbind(1)
            result = model(inputs, d, s)
            mask = level < torch.as_tensor(inputs.source.capacity[s.numpy()])[:, None]
            loss = ((result - target[d, s])[mask] / 20).square().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        if validation is not None:
            vd, vs = validation
            err = predict(model, inputs, vd, vs) - target[vd][:, vs].numpy()
            mask = np.broadcast_to(np.arange(err.shape[-1])[None, None] < inputs.source.capacity[vs][None, :, None], err.shape)
            value = float(np.mean(err[mask] ** 2))
            if value < best - 0.01:
                best, selected = value, epoch
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch - selected >= 8:
                break
    if validation is not None:
        model.load_state_dict(state)
    return model, selected if validation is not None else epochs


def blend(pred, anchor, off, weekday, holiday):
    weights = np.where(off, holiday, weekday)[:, None, None]
    return anchor[None] + weights * (pred - anchor[None])


def select(inputs, target, sites, days, public_days):
    validation = days[-7:]
    past = np.setdiff1d(days[:-7], public_days)
    if not set(public_days).issubset(days[:-7]):
        raise ValueError("Public days must precede inner validation")
    groups = np.array_split(sites, 3)
    candidates = list(itertools.product(BLENDS, repeat=2))
    scores, counts = {c: [] for c in candidates}, []
    data = inputs.source
    for i, dev in enumerate(groups):
        core = np.concatenate(groups[:i] + groups[i+1:])
        fitted = [fit(inputs, target, past, core, seed, validation=(validation, dev)) for seed in SEEDS]
        counts.append([e for _, e in fitted])
        pred = ensemble([m for m, _ in fitted], inputs, validation, dev)
        for c in candidates:
            mixed = blend(pred, inputs.anchor[dev].numpy(), data.city.off_day[validation], *c)
            allocation, _ = optimize(inventory.cost_table(mixed, data.capacity[dev]), data.capacity[dev])
            scores[c].append(realized_score(data.city.net_flow[validation][:, dev].numpy(), allocation, data.capacity[dev])["score"])
        print(f"  signature inner {i+1}/3 done", flush=True)
    search = [{"weekday_blend": c[0], "holiday_blend": c[1], "selection_score": sum(v)} for c, v in scores.items()]
    chosen = min(search, key=lambda x: x["selection_score"])
    return {**chosen, "epochs": np.median(counts, axis=0).astype(int).tolist()}, search


def train(data, public_days, output):
    inputs = make_inputs(data, data.city.net_flow[public_days].numpy())
    target = transfer.increments(data.city.net_flow.numpy(), data.capacity)
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    past, future = days[:-7], days[-7:]
    train_days = np.setdiff1d(past, public_days)
    order = np.random.default_rng(base.SEED).permutation(len(sites))
    groups = np.split(order, [i * (len(sites) // 5) for i in range(1, 5)])
    allocations = np.empty((len(future), len(sites)), dtype=np.int64)
    out_of_fold = np.empty((len(future), len(sites), int(data.capacity.max())))
    records = []
    for i, test in enumerate(groups):
        print(f"Signature outer {i+1}/5", flush=True)
        sources = np.concatenate(groups[i+1:] + groups[:i])
        config, search = select(inputs, target, sources, past, public_days)
        models = [fit(inputs, target, train_days, sources, seed, epoch)[0] for seed, epoch in zip(SEEDS, config["epochs"])]
        raw = ensemble(models, inputs, future, test)
        mixed = blend(raw, inputs.anchor[test].numpy(), data.city.off_day[future], config["weekday_blend"], config["holiday_blend"])
        out_of_fold[:, test] = mixed
        allocations[:, test], _ = optimize(inventory.cost_table(mixed, data.capacity[test]), data.capacity[test])
        actual = realized_score(data.city.net_flow[future][:, test].numpy(), allocations[:, test], data.capacity[test])
        records.append({"test_station_ids": data.city.stations[test].tolist(), "config": config, "search": search, "actual_score": actual})
        print(json.dumps({"config": config, "actual_score": actual}), flush=True)
        (output / "validation_progress.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    config, search = select(inputs, target, order, past, public_days)
    models = [fit(inputs, target, np.setdiff1d(days, public_days), sites, seed, epoch)[0] for seed, epoch in zip(SEEDS, config["epochs"])]
    torch.save({"model_type": "public_signature_v2", "ns": inputs.signature.shape[-1], "nt": inventory.day_features(data).shape[-1],
                "states": [m.state_dict() for m in models], "seeds": list(SEEDS), "config": config, "public_days": public_days.tolist()}, output / MODEL)
    report = {"protocol": "5 held-out station folds, fit other stations on D01-D23 except public D09/D10. Target stations provide only D09/D10 signatures. Evaluate D24-D30. 3 inner folds select epochs and blends on D17-D23.",
              "public_dates": data.city.dates[public_days].strftime("%Y-%m-%d").tolist(), "folds": records, "final_config": config, "final_search": search,
              "actual_score": realized_score(data.city.net_flow[future].numpy(), allocations, data.capacity)}
    (output / "A_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output / 'A_signature_costs.npz', increments=out_of_fold)
    pd.DataFrame(allocations.T, index=data.city.stations, columns=[f"D{d+1:02d}" for d in future]).rename_axis("station_id").to_csv(output / "A_검증배치.csv", encoding="utf-8-sig")
    print(json.dumps({"final_config": config, "actual_score": report["actual_score"]}), flush=True)


def export(data, public_dir, output):
    public_days, truth, provenance = transfer.read_public(public_dir, data)
    saved = torch.load(output / MODEL, map_location="cpu", weights_only=True)
    if saved.get("model_type") != "public_signature_v2" or saved["public_days"] != public_days.tolist():
        raise ValueError("Checkpoint/public-day protocol mismatch")
    inputs = make_inputs(data, truth)
    models = []
    for state in saved["states"]:
        model = SignatureNet(saved["ns"], saved["nt"])
        model.load_state_dict(state)
        models.append(model)
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    config = saved["config"]
    raw = ensemble(models, inputs, days, sites)
    pred = blend(raw, inputs.anchor.numpy(), data.city.off_day, config["weekday_blend"], config["holiday_blend"])
    pred[public_days] = transfer.increments(truth, data.capacity)
    cost = inventory.cost_table(pred, data.capacity)
    allocation, relative = optimize(cost, data.capacity)
    pd.DataFrame(allocation.T, index=data.city.stations, columns=[f"D{d+1:02d}" for d in days]).rename_axis("station_id").to_csv(output / SUBMISSION, encoding="utf-8-sig")
    np.savez_compressed(output / "B_costs.npz", increments=pred, ids=data.city.stations.to_numpy(str), capacity=data.capacity)
    result = {"N": int(allocation[0].sum()), "stations": len(sites), "days": len(days), "model_type": saved["model_type"], "config": config,
              "public_dates": data.city.dates[public_days].strftime("%Y-%m-%d").tolist(), "public_files": provenance,
              "public_actual_score": realized_score(truth, allocation[public_days], data.capacity), "relative_objective": relative,
              "note": "Public scores are in-sample; full-month B score unknown. Relative objective excludes C(0), so is not predicted competition score."}
    (output / "B_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"], nargs="?", default="predict")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1])
    parser.add_argument("--public-dir", type=Path, default=ROOT.parents[1] / "open_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_v2")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_b = base.read_data(args.data_dir, "B")
    if args.action == "train":
        public_days, _, _ = transfer.read_public(args.public_dir, data_b)
        train(base.read_data(args.data_dir, "A"), public_days, args.output_dir)
    export(data_b, args.public_dir, args.output_dir)


if __name__ == "__main__":
    main()
