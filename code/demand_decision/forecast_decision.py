"""Cold-start bicycle demand and scenario-based allocation.

python code/demand_decision/forecast_decision.py train
python code/demand_decision/forecast_decision.py predict

Only the supplied learn_data/input_data CSV files are used. The official daily objective is
20*N + sum(6*shortage**2 + 5*overflow + excess_above_half_capacity).
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if (ROOT / ".decision-runtime").is_dir():
    sys.path.insert(0, str(ROOT / ".decision-runtime"))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from numba import njit

SEED = 202606
DEPTHS = (4, 6)
POLICIES = ("point", "analog", "blend", "centered")
ALPHA = {"analog": 0.0, "blend": 0.5, "centered": 1.0}
ACCESS = ["nearest_bus_distance_m", "bus_stop_count_500m",
          "nearest_subway_distance_m", "subway_station_count_500m",
          "subway_station_count_1000m", "school_name_flag", "park_name_flag",
          "commercial_name_flag"]
WEATHER = ["temperature_c", "humidity_pct", "precipitation_mm", "wind_speed_kmh"]
FORECAST = "B지역_시간별_순유입_예측.csv"
ALLOCATION = "B지역_추천배치표.csv"


@dataclass
class City:
    ids: np.ndarray
    dates: np.ndarray
    static: np.ndarray
    temporal: np.ndarray
    day_weather: np.ndarray
    off_day: np.ndarray
    capacity: np.ndarray
    net: np.ndarray | None


@dataclass
class Reference:
    static: np.ndarray
    day_weather: np.ndarray
    off_day: np.ndarray
    capacity: np.ndarray
    net: np.ndarray


def unique_index(frame, keys):
    result = frame.set_index(keys)
    if not result.index.is_unique:
        raise ValueError(f"Duplicate keys: {keys}")
    return result


def read_city(path, labelled):
    filenames = {"01_시간별수요": "demand.csv", "02_대여소": "spots.csv",
                 "03_거리": "dist.csv", "04_05_접근환경": "nearby.csv",
                 "06_07_달력기상": "calclim.csv"}
    def read(name):
        return pd.read_csv(path / filenames[name], encoding="utf-8-sig",
                           float_precision="round_trip")
    stations = unique_index(read("02_대여소"), "station_id").sort_index()
    ids = stations.index
    access = unique_index(read("04_05_접근환경"), "station_id").loc[ids]
    distance = read("03_거리").pivot(index="from_station_id", columns="to_station_id",
                                   values="distance_km").loc[ids, ids].to_numpy(float, copy=True)
    capacity = stations.capacity.to_numpy(np.int64)
    if (len(ids) <= 6 or (capacity <= 0).any()
            or not np.isfinite(distance).all() or (distance < 0).any()):
        raise ValueError("Invalid station capacities or distance matrix")
    if not np.allclose(distance, distance.T) or not np.allclose(np.diag(distance), 0):
        raise ValueError("Distance matrix must be symmetric with zero diagonal")
    np.fill_diagonal(distance, np.inf)
    near = np.argsort(distance, axis=1, kind="stable")[:, :6]
    near_distance = np.take_along_axis(distance, near, axis=1)
    bandwidth = max(np.median(near_distance[:, -1]), 1e-6)
    xy = stations[["x_km", "y_km"]].to_numpy()
    radius = np.linalg.norm(xy - xy.mean(0), axis=1)
    basic = np.column_stack([np.log(capacity), np.log1p(access[ACCESS[:5]]),
                              access[ACCESS[5:]]])
    geometry = np.column_stack([radius / max(np.median(radius), 1e-6),
                                near_distance[:, 0] / bandwidth,
                                near_distance.mean(1) / bandwidth])
    static = np.column_stack([basic, geometry, basic[near].mean(axis=1)])
    calendar = read("06_07_달력기상")
    calendar["date"] = pd.to_datetime(calendar.date)
    dates = np.sort(calendar.date.unique())
    times = pd.MultiIndex.from_product([dates, range(24)], names=["date", "hour"])
    calendar = unique_index(calendar, ["date", "hour"]).loc[times]
    weather = calendar[WEATHER].to_numpy(float).reshape(len(dates), 24, 4)
    daily = np.column_stack([weather[:, :, 0].mean(1), weather[:, :, 0].max(1),
                             np.log1p(weather[:, :, 2].sum(1)),
                             weather[:, :, 1].mean(1), weather[:, :, 3].mean(1)])
    weekday = pd.DatetimeIndex(dates).day_of_week.to_numpy()
    off = (calendar.is_weekend.astype(bool) | calendar.is_holiday.astype(bool)).to_numpy()
    off_day = off.reshape(-1, 24)[:, 0]
    h = np.tile(np.arange(24), len(dates))
    angle = h * (2 * np.pi / 24)
    rain = np.log1p(weather[:, :, 2])
    rain_lag = np.cumsum(rain, axis=1)
    rain_lag[:, 3:] -= np.cumsum(rain, axis=1)[:, :-3]
    temporal = np.column_stack([h, np.sin(angle), np.cos(angle),
                                np.sin(2 * angle), np.cos(2 * angle),
                                np.repeat(weekday, 24), calendar.is_weekend,
                                calendar.is_holiday, off, calendar[WEATHER],
                                rain_lag.ravel(), np.repeat(daily, 24, axis=0)])
    temporal = temporal.reshape(len(dates), 24, -1)
    net = None
    if labelled:
        observed = read("01_시간별수요")
        observed["date"] = pd.to_datetime(observed.date)
        keys = ["date", "station_id", "hour"]
        expected = pd.MultiIndex.from_product([dates, ids, range(24)], names=keys)
        labels = unique_index(observed, keys)
        if len(labels) != len(expected):
            raise ValueError("Unexpected number of observed station-hours")
        counts = labels.loc[expected, ["rentals", "returns"]].to_numpy(float)
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Missing or negative counts")
        net = (counts[:, 1] - counts[:, 0]).reshape(len(dates), len(ids), 24)
    for data in (static, temporal, daily):
        if not np.isfinite(data).all():
            raise ValueError("Non-finite features")
    return City(ids.to_numpy(str), dates.astype("datetime64[D]").astype(str),
                static, temporal, daily, off_day, capacity, net)


def features(city, days, sites):
    n_days, n_sites = len(days), len(sites)
    station = np.broadcast_to(city.static[sites][None, :, None, :],
                              (n_days, n_sites, 24, city.static.shape[-1]))
    time = np.broadcast_to(city.temporal[days][:, None, :, :],
                           (n_days, n_sites, 24, city.temporal.shape[-1]))
    return np.concatenate([station, time], axis=-1).reshape(n_days * n_sites * 24, -1).astype(np.float32)


def fit_mean(city, days, sites, depth, iterations=400, validation=None):
    model = CatBoostRegressor(iterations=int(iterations), depth=int(depth),
                              learning_rate=0.05, l2_leaf_reg=20, loss_function="RMSE",
                              random_seed=SEED, thread_count=4, verbose=False,
                              allow_writing_files=False)
    kwargs = {}
    if validation is not None:
        vd, vs = validation
        kwargs = {"eval_set": (features(city, vd, vs), city.net[vd][:, vs].ravel()),
                  "early_stopping_rounds": 35, "use_best_model": True}
    model.fit(features(city, days, sites), city.net[days][:, sites].ravel(), **kwargs)
    return model


def predict_mean(model, city, days, sites):
    return model.predict(features(city, days, sites)).reshape(len(days), len(sites), 24)


def reference(city, days, sites):
    return Reference(city.static[sites], city.day_weather[days], city.off_day[days],
                     city.capacity[sites], city.net[days][:, sites])


def analog_scenarios(ref, city, days, sites):
    """Whole-day observed paths preserve simultaneous and consecutive-hour variation."""
    scale = np.maximum(ref.static.std(0), 0.25)
    differences = (city.static[sites, None] - ref.static[None]) / scale
    differences[:, :, 0] *= 2  # capacity is important for the scale of demand
    distance = np.mean(differences ** 2, axis=-1)
    k = min(32, len(ref.static))
    neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k]
    selected_distance = np.take_along_axis(distance, neighbors, axis=1)
    local_bandwidth = np.maximum(np.median(selected_distance, axis=1, keepdims=True), 1e-6)
    site_weights = np.exp(-selected_distance / (2 * local_bandwidth))
    site_weights /= site_weights.sum(1, keepdims=True)
    capacity_ratio = np.sqrt(city.capacity[sites, None] / ref.capacity[neighbors])
    capacity_ratio = np.clip(capacity_ratio, 0.5, 2.0)
    weather_scale = np.maximum(ref.day_weather.std(0), [1, 1, 0.5, 5, 1])
    scenarios, weights = [], []
    for day in days:
        matching = np.flatnonzero(ref.off_day == city.off_day[day])
        if len(matching) == 0:
            matching = np.arange(len(ref.off_day))
        weather_distance = np.mean(((ref.day_weather[matching] - city.day_weather[day])
                                    / weather_scale) ** 2, axis=1)
        rank = np.argsort(weather_distance, kind="stable")[:3]
        selected_days = matching[rank]
        # Duplicate slots split the original day's mass if fewer than three days exist.
        selected_days = np.resize(selected_days, 3)
        day_dist = np.mean(((ref.day_weather[selected_days] - city.day_weather[day])
                           / weather_scale) ** 2, axis=1)
        day_weights = np.exp(-day_dist / 2)
        day_weights /= np.array([(selected_days == d).sum() for d in selected_days])
        day_weights /= day_weights.sum()
        paths = ref.net[selected_days[:, None, None], neighbors[None, :, :], :]
        paths = paths.transpose(1, 0, 2, 3) * capacity_ratio[:, None, :, None]
        joint = site_weights[:, None, :] * day_weights[None, :, None]
        scenarios.append(paths.reshape(len(sites), 3 * k, 24))
        weights.append(joint.reshape(len(sites), 3 * k))
    return np.array(scenarios), np.array(weights)


def policy_scenarios(paths, weights, mean, policy):
    if policy == "point":
        return mean[:, :, None, :].copy(), np.ones(mean.shape[:2] + (1,))
    center = (paths * weights[:, :, :, None]).sum(axis=2)
    shift = ALPHA[policy] * (mean - center)
    return paths + shift[:, :, None, :], weights


@njit(cache=True)
def scenario_costs(paths, weights, caps):
    days, sites, scenarios, hours = paths.shape
    costs = np.full((days, sites, caps.max() + 1), np.inf)
    for d in range(days):
        for i in range(sites):
            cap = caps[i]
            for x in range(cap + 1):
                total = 0.0
                for s in range(scenarios):
                    b = float(x)
                    penalty = 0.0
                    for h in range(hours):
                        z = b + paths[d, i, s, h]
                        u, o = max(-z, 0.0), max(z - cap, 0.0)
                        b = min(float(cap), max(0.0, z))
                        penalty += 6 * u * u + 5 * o + max(b - cap // 2, 0.0)
                    total += weights[d, i, s] * penalty
                costs[d, i, x] = total
    return costs


# Exact total-fleet DP adapted from the team's code/3_DemandToOut.py.
@njit(cache=True)
def dp_all_totals(cost, caps):
    total = int(caps.sum())
    prev = np.full(total + 1, np.inf)
    prev[0] = 0.0
    max_before = 0
    for i in range(len(caps)):
        nxt = np.full(total + 1, np.inf)
        for n in range(max_before + 1):
            for x in range(caps[i] + 1):
                value = prev[n] + cost[i, x]
                if value < nxt[n + x]:
                    nxt[n + x] = value
        prev = nxt
        max_before += caps[i]
    return prev


@njit(cache=True)
def dp_reconstruct(cost, caps, total):
    prev = np.full(total + 1, np.inf)
    prev[0] = 0.0
    chosen = np.full((len(caps), total + 1), -1, dtype=np.int64)
    max_before = 0
    for i in range(len(caps)):
        nxt = np.full(total + 1, np.inf)
        for n in range(min(max_before, total) + 1):
            for x in range(min(caps[i], total - n) + 1):
                value = prev[n] + cost[i, x]
                if value < nxt[n + x]:
                    nxt[n + x] = value
                    chosen[i, n + x] = x
        prev = nxt
        max_before += caps[i]
    result = np.zeros(len(caps), dtype=np.int64)
    remaining = total
    for i in range(len(caps) - 1, -1, -1):
        result[i] = chosen[i, remaining]
        remaining -= result[i]
    return result


def optimize(costs, caps):
    totals = np.stack([dp_all_totals(day, caps) for day in costs])
    scores = 20 * np.arange(len(totals[0])) + totals.mean(0)
    scores[0] = np.inf
    fleet = int(np.argmin(scores))
    allocation = np.stack([dp_reconstruct(day, caps, fleet) for day in costs])
    if (allocation < 0).any() or (allocation > caps).any() or not np.all(allocation.sum(1) == fleet):
        raise ValueError("Invalid allocation from DP")
    obtained = costs[np.arange(len(costs))[:, None], np.arange(len(caps)), allocation].sum(1)
    if not np.allclose(obtained, totals[:, fleet]):
        raise ValueError("Allocation does not reproduce DP optimum")
    return allocation, float(scores[fleet])


def realized_score(truth, allocation, caps):
    b = allocation.astype(float).copy()
    q = np.zeros(len(truth))
    o = np.zeros(len(truth))
    e = np.zeros(len(truth))
    for hour in range(24):
        z = b + truth[:, :, hour]
        q += (np.maximum(-z, 0) ** 2).sum(1)
        o += np.maximum(z - caps, 0).sum(1)
        b = np.clip(z, 0, caps)
        e += np.maximum(b - caps // 2, 0).sum(1)
    n = allocation.sum(1)
    if not np.all(n == n[0]) or n[0] < 1:
        raise ValueError("Fleet must be positive and fixed over days")
    return {"score": float((20 * n + 6 * q + 5 * o + e).mean()),
            "N": int(n[0]), "Q_daily": float(q.mean()), "O_daily": float(o.mean()),
            "E_daily": float(e.mean())}


def net_metrics(truth, pred):
    error = pred - truth
    return {"station_hours": int(error.size), "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "over_MAE_all_rows": float(np.maximum(error, 0).mean()),
            "under_MAE_all_rows": float(np.maximum(-error, 0).mean()),
            "bias": float(error.mean())}


def select_config(city, sites, days):
    train_days, dev_days = days[:-7], days[-7:]
    groups = np.array_split(sites, 3)
    collected = {}
    for group, dev in enumerate(groups):
        core = np.concatenate(groups[:group] + groups[group+1:])
        ref = reference(city, train_days, core)
        paths, weights = analog_scenarios(ref, city, dev_days, dev)
        caps = city.capacity[dev]
        truth = city.net[dev_days][:, dev]
        for depth in DEPTHS:
            model = fit_mean(city, train_days, core, depth, validation=(dev_days, dev))
            mean = predict_mean(model, city, dev_days, dev)
            for policy in POLICIES:
                # Analog policy is independent of tree depth.
                if policy == "analog" and depth != DEPTHS[0]:
                    continue
                scenarios, mass = policy_scenarios(paths, weights, mean, policy)
                allocation, _ = optimize(scenario_costs(scenarios, mass, caps), caps)
                collected.setdefault((depth, policy), []).append({
                    "iterations": int(model.tree_count_),
                    "validation": realized_score(truth, allocation, caps)})
    choices = []
    for (depth, policy), records in collected.items():
        choices.append({"depth": depth, "policy": policy,
                        "iterations": int(np.median([r["iterations"] for r in records])),
                        "validation": {key: sum(r["validation"][key] for r in records)
                                       for key in records[0]["validation"]},
                        "inner_folds": records})
    best = min(choices, key=lambda item: item["validation"]["score"])
    return {k: best[k] for k in ("depth", "iterations", "policy")}, choices


def forecast_policy(ref, city, days, sites, mean, policy):
    paths, weights = analog_scenarios(ref, city, days, sites)
    scenarios, mass = policy_scenarios(paths, weights, mean, policy)
    prediction = (scenarios * mass[:, :, :, None]).sum(axis=2)
    allocation, expected_score = optimize(scenario_costs(scenarios, mass, city.capacity[sites]),
                                          city.capacity[sites])
    return prediction, allocation, expected_score


def train(city, output):
    total_days, total_sites = len(city.dates), len(city.ids)
    past, future = np.arange(total_days - 7), np.arange(total_days - 7, total_days)
    order = np.random.default_rng(SEED).permutation(total_sites)
    groups = np.split(order, [i * (total_sites // 5) for i in range(1, 5)])
    all_predictions = {k: np.empty((7, total_sites, 24)) for k in ("selected", "catboost_point", "zero")}
    all_allocations = {k: np.empty((7, total_sites), dtype=np.int64)
                       for k in ("selected", "catboost_point", "one_each", "half_capacity")}
    reports = []
    for fold, test in enumerate(groups):
        sources = np.concatenate(groups[fold+1:] + groups[:fold])
        config, search = select_config(city, sources, past)
        print(f"Fold {fold + 1}/5 selected {config}", flush=True)
        point_config = min((c for c in search if c["policy"] == "point"),
                           key=lambda c: c["validation"]["score"])
        point_model = fit_mean(city, past, sources, point_config["depth"], point_config["iterations"])
        point_mean = predict_mean(point_model, city, future, test)
        mean = point_mean
        if config["policy"] != "analog" and any(config[k] != point_config[k] for k in ("depth", "iterations")):
            model = fit_mean(city, past, sources, config["depth"], config["iterations"])
            mean = predict_mean(model, city, future, test)
        ref = reference(city, past, sources)
        pred, allocation, _ = forecast_policy(ref, city, future, test, mean, config["policy"])
        point_allocation, _ = optimize(scenario_costs(point_mean[:, :, None, :],
                                                     np.ones(point_mean.shape[:2] + (1,)),
                                                     city.capacity[test]), city.capacity[test])
        all_predictions["selected"][:, test] = pred
        all_predictions["catboost_point"][:, test] = point_mean
        all_predictions["zero"][:, test] = 0
        all_allocations["selected"][:, test] = allocation
        all_allocations["catboost_point"][:, test] = point_allocation
        all_allocations["one_each"][:, test] = 1
        all_allocations["half_capacity"][:, test] = city.capacity[test] // 2
        scores = {k: realized_score(city.net[future][:, test], value[:, test], city.capacity[test])
                  for k, value in all_allocations.items()}
        reports.append({"test_station_ids": city.ids[test].tolist(), "config": config,
                        "point_config": {k: point_config[k] for k in ("depth", "iterations")},
                        "inner_search": search, "scores": scores})
        print(f"Fold {fold + 1} actual scores {scores}", flush=True)
    truth = city.net[future]
    report = {"seed": SEED, "train_dates": city.dates[past].tolist(),
              "test_dates": city.dates[future].tolist(),
              "validation_design": "5 outer station folds, 3 inner station folds; inner train days 1-16, selection days 17-23, outer test days 24-30",
              "allocation_scores": {k: realized_score(truth, alloc, city.capacity)
                                    for k, alloc in all_allocations.items()},
              "forecast_metrics": {k: net_metrics(truth, pred) for k, pred in all_predictions.items()},
              "folds": reports}
    print(json.dumps(report["allocation_scores"]), flush=True)
    print(json.dumps(report["forecast_metrics"]), flush=True)
    # Final selection uses days 1-23 only; outer test outcomes do not select parameters.
    config, search = select_config(city, order, past)
    config.update({"seed": SEED, "target": "returns - rentals", "neighbors": 32,
                   "weather_days": 3, "capacity_exponent": 0.5,
                   "model_type": "conditional_daily_scenarios"})
    ref = reference(city, np.arange(total_days), np.arange(total_sites))
    if config["policy"] != "analog":
        model = fit_mean(city, np.arange(total_days), np.arange(total_sites),
                         config["depth"], config["iterations"])
        model.save_model(str(output / "net_mean.cbm"))
    np.savez_compressed(output / "reference.npz", static=ref.static, day_weather=ref.day_weather,
                        off_day=ref.off_day, capacity=ref.capacity, net=ref.net)
    report["final_config"] = config
    report["final_inner_search"] = search
    (output / "model.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "A_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    idx = pd.MultiIndex.from_product([city.dates[future], city.ids, range(24)],
                                     names=["date", "station_id", "hour"])
    audit = pd.DataFrame({"actual_net": truth.ravel(),
                          "selected_net": all_predictions["selected"].ravel(),
                          "catboost_net": all_predictions["catboost_point"].ravel()}, index=idx)
    audit.to_csv(output / "A_검증예측.csv", encoding="utf-8-sig")
    pd.DataFrame(all_allocations["selected"].T, index=city.ids,
                 columns=[f"D{i+1:02d}" for i in future]).rename_axis("station_id").to_csv(
                     output / "A_검증배치.csv", encoding="utf-8-sig")
    print(f"Final config {config}", flush=True)


def export(city, output):
    config = json.loads((output / "model.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "conditional_daily_scenarios":
        raise ValueError("Expected a conditional daily scenario model")
    with np.load(output / "reference.npz", allow_pickle=False) as data:
        ref = Reference(**{key: data[key] for key in Reference.__annotations__})
    days, sites = np.arange(len(city.dates)), np.arange(len(city.ids))
    mean = np.zeros((len(days), len(sites), 24))
    if config["policy"] != "analog":
        model = CatBoostRegressor()
        model.load_model(str(output / "net_mean.cbm"))
        mean = predict_mean(model, city, days, sites)
    pred, allocation, expected = forecast_policy(ref, city, days, sites, mean, config["policy"])
    index = pd.MultiIndex.from_product([city.dates, city.ids, range(24)],
                                       names=["date", "station_id", "hour"])
    frame = pd.DataFrame({"pred_net_flow": pred.ravel()}, index=index).reset_index()
    frame = frame[["date", "hour", "station_id", "pred_net_flow"]].sort_values(["date", "hour", "station_id"])
    if not np.isfinite(pred).all():
        raise ValueError("Invalid forecast")
    frame.to_csv(output / FORECAST, index=False, float_format="%.4f", encoding="utf-8-sig")
    pd.DataFrame(allocation.T, index=city.ids,
                 columns=[f"D{i+1:02d}" for i in days]).rename_axis("station_id").to_csv(
                     output / ALLOCATION, encoding="utf-8-sig")
    summary = {"stations": len(sites), "days": len(days), "station_hours": int(pred.size),
               "N": int(allocation[0].sum()), "estimated_scenario_score": expected,
               "note": "Estimated scenario score only. B actual score is unknown.",
               "model_config": config}
    (output / "B_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {pred.size} station-hours, N={summary['N']}, policy={config['policy']}", flush=True)


def find_input(folder, pattern):
    paths = [p for p in folder.glob(pattern) if not p.name.startswith("~$")]
    if len(paths) != 1:
        raise ValueError(f"Expected one input file matching {pattern}")
    return paths[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"], nargs="?", default="train")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.action == "train":
        train(read_city(args.data_dir / "learn_data", True), args.output_dir)
    export(read_city(args.data_dir / "input_data", False), args.output_dir)
