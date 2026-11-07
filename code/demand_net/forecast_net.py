"""반납 - 대여를 하나의 실수로 직접 예측하는 GNN + LSTM 모델.

재현 환경: torch 2.15.0 (CPU), numpy 2.3.5, pandas 3.0.1, openpyxl 3.1.5.
저장소 루트에서 learn_data와 input_data의 CSV를 읽는다.
python code/demand_net/forecast_net.py train
python code/demand_net/forecast_net.py predict

y = 실제 반납 - 실제 대여, e = 순유입 예측값 - y
기본 손실은 mean(e^2)인 대칭 제곱오차다 (over_weight=1).
양수 출력은 재고 증가, 음수 출력은 재고 감소다.
over_weight>1은 평균보다 낮은 예측을 유도하는 비교 실험용 설정이다.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 이 프로젝트에 설치한 Windows용 CPU 런타임도 사용할 수 있다.
if find_spec("torch") is None and sys.version_info[:2] == (3, 12) and os.name == "nt":
    local_runtime = ROOT / ".ml-runtime"
    if local_runtime.is_dir():
        sys.path.insert(0, str(local_runtime))

import numpy as np
import pandas as pd
import torch
from torch import nn


SEED = 202606
K = 6
TARGET = "returns - rentals"
ACCESS = ["nearest_bus_distance_m", "bus_stop_count_500m",
          "nearest_subway_distance_m", "subway_station_count_500m",
          "subway_station_count_1000m", "school_name_flag", "park_name_flag",
          "commercial_name_flag"]
WEATHER = ["temperature_c", "humidity_pct", "precipitation_mm", "wind_speed_kmh"]
CSV_NAME = "B지역_시간별_순유입_예측.csv"
MODEL_NAME = "bike_net_flow.pt"
DEFAULT_OVER_WEIGHT = 1.0


@dataclass
class City:
    stations: pd.Index
    dates: pd.DatetimeIndex
    static: torch.Tensor
    adjacency: torch.Tensor
    temporal: torch.Tensor        # day, hour, feature
    off_day: np.ndarray
    net_flow: torch.Tensor | None  # day, station, hour. returns - rentals


def tensor(x):
    return torch.from_numpy(np.array(x, dtype=np.float32, order="C", copy=True))


def unique_index(frame, keys):
    indexed = frame.set_index(keys)
    if not indexed.index.is_unique:
        raise ValueError(f"중복 키가 있습니다: {keys}")
    return indexed


def location_inputs(stations, dist, access):
    # 지역마다 원점·방향·축척이 다르다. 상대 거리와 중심으로부터의 거리만 사용한다.
    n = len(stations)
    if n <= K or not np.allclose(dist, dist.T) or not np.allclose(np.diag(dist), 0):
        raise ValueError("대여소 거리 행렬 확인 필요")
    if not np.isfinite(dist).all() or (dist < 0).any():
        raise ValueError("거리 결측 또는 음수")
    edges = dist.copy()
    np.fill_diagonal(edges, np.inf)
    closest = np.argsort(edges, axis=1, kind="stable")[:, :K]
    lengths = np.take_along_axis(edges, closest, axis=1)
    typical_radius = max(np.median(lengths[:, -1]), 1e-6)
    graph = np.zeros_like(dist)
    np.put_along_axis(graph, closest, np.exp(-lengths / typical_radius), axis=1)
    graph = np.maximum(graph, graph.T)
    graph /= graph.sum(axis=1, keepdims=True)
    xy = stations[["x_km", "y_km"]].to_numpy()
    radius = np.linalg.norm(xy - xy.mean(axis=0), axis=1)
    location = np.column_stack([radius / max(np.median(radius), 1e-6),
                                lengths[:, 0] / typical_radius,
                                lengths.mean(axis=1) / typical_radius])
    attributes = np.column_stack([np.log(stations.capacity), np.log1p(access[ACCESS[:5]]),
                                  access[ACCESS[5:]], location])
    return tensor(attributes), tensor(graph)


def read_city(folder, labelled):
    def read(filename):
        return pd.read_csv(folder / filename, encoding="utf-8-sig", float_precision="round_trip")

    stations = unique_index(read("spots.csv"), "station_id").sort_index()
    ids = stations.index
    access = unique_index(read("nearby.csv"), "station_id").loc[ids]
    pairs = read("dist.csv").pivot(index="from_station_id", columns="to_station_id",
                               values="distance_km").loc[ids, ids]
    if not (stations.capacity > 0).all():
        raise ValueError("수용량은 양수여야 합니다")
    static, adjacency = location_inputs(stations, pairs.to_numpy(), access)
    cal = read("calclim.csv")
    cal["date"] = pd.to_datetime(cal.date)
    dates = pd.DatetimeIndex(sorted(cal.date.unique()))
    times = pd.MultiIndex.from_product([dates, range(24)], names=["date", "hour"])
    cal = unique_index(cal, ["date", "hour"]).loc[times]
    h = cal.index.get_level_values("hour").to_numpy()
    weekday = cal.index.get_level_values("date").dayofweek.to_numpy()
    provided_dow = cal.day_of_week.map(dict(zip("월화수목금토일", range(7))))
    if not np.array_equal(provided_dow, weekday):
        raise ValueError("달력 요일 불일치")
    off = (cal.is_weekend.astype(bool) | cal.is_holiday.astype(bool)).to_numpy()
    angle = h * (2 * np.pi / 24)
    clock = [f(k * angle) for k in range(1, 5) for f in (np.sin, np.cos)]
    weather = cal[WEATHER].copy()
    weather["precipitation_mm"] = np.log1p(weather.precipitation_mm)
    temporal = np.column_stack([*clock, np.eye(7)[weekday], cal.is_weekend,
                                cal.is_holiday, off, weather])
    temporal = tensor(temporal.reshape(len(dates), 24, -1))
    net_flow = None
    if labelled:
        observations = read("demand.csv")
        observations["date"] = pd.to_datetime(observations.date)
        keys = ["date", "station_id", "hour"]
        expected = pd.MultiIndex.from_product([dates, ids, range(24)], names=keys)
        y = unique_index(observations, keys)
        if len(y) != len(expected):
            raise ValueError("수요 행 수가 날짜 × 대여소 × 24와 다릅니다")
        values = y.loc[expected, ["rentals", "returns"]].to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("수요에 누락 또는 음수가 있습니다")
        counts = tensor(values.reshape(len(dates), len(ids), 24, 2))
        net_flow = counts[..., 1] - counts[..., 0]
    if not torch.isfinite(static).all() or not torch.isfinite(temporal).all():
        raise ValueError("대여소·달력·기상 입력에 결측 또는 무한값이 있습니다")
    return City(ids, dates, static, adjacency, temporal,
                off.reshape(-1, 24)[:, 0], net_flow)


class GraphLSTM(nn.Module):
    def __init__(self, ns, nt):
        super().__init__()
        self.register_buffer("s_mean", torch.zeros(ns))
        self.register_buffer("s_scale", torch.ones(ns))
        self.register_buffer("t_mean", torch.zeros(nt))
        self.register_buffer("t_scale", torch.ones(nt))
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_scale", torch.tensor(1.0))
        self.g1 = nn.Linear(2 * ns, 16)
        self.g2 = nn.Linear(32, 12)
        self.station = nn.Sequential(nn.Linear(ns + 12, 16), nn.Tanh())
        self.init_state = nn.Linear(16, 48)
        self.lstm = nn.LSTM(nt + 16, 24, batch_first=True)
        self.dropout = nn.Dropout(0.12)
        self.head = nn.Linear(24 + 16, 1)

    def fit_scales(self, city, days, sites):
        for x, mean, scale in [(city.static[sites], self.s_mean, self.s_scale),
                               (city.temporal[days].flatten(0, 1), self.t_mean, self.t_scale)]:
            mean.copy_(x.mean(0))
            scale.copy_(x.std(0, unbiased=False).clamp_min(0.2))
        target = city.net_flow[days][:, sites]
        self.y_mean.copy_(target.mean())
        self.y_scale.copy_(target.std(unbiased=False).clamp_min(1.0))

    def forward(self, city, days, sites):
        x = ((city.static - self.s_mean) / self.s_scale).clamp(-5, 5)
        graph = city.adjacency
        z = torch.tanh(self.g1(torch.cat((x, graph @ x), dim=-1)))
        z = torch.tanh(self.g2(torch.cat((z, graph @ z), dim=-1)))
        station = self.station(torch.cat((x, z), dim=-1))[sites]
        station = self.dropout(station)
        h, c = self.init_state(station).tanh().chunk(2, dim=-1)
        repeat = station[:, None].expand(-1, 24, -1)
        time = ((city.temporal[days] - self.t_mean) / self.t_scale).clamp(-5, 5)
        out, _ = self.lstm(torch.cat((time, repeat), -1),
                           (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous()))
        # 선형 출력으로 양수·음수·0을 모두 허용한다. y의 통계는 학습 표본에서만 구한다.
        standardized_net = self.head(torch.cat((self.dropout(out), repeat), -1)).squeeze(-1)
        return standardized_net * self.y_scale + self.y_mean


def examples(days, sites):
    return torch.cartesian_prod(torch.as_tensor(days), torch.as_tensor(sites))


@torch.no_grad()
def predict(model, city, days, sites):
    model.eval()
    batches = examples(days, sites).split(256)
    predictions = [model(city, pairs[:, 0], pairs[:, 1]) for pairs in batches]
    return torch.cat(predictions).reshape(len(days), len(sites), 24)


def objective(prediction, truth, over_weight):
    error = prediction - truth
    over = error.clamp_min(0)
    under = (-error).clamp_min(0)
    return (over_weight * over.square() + under.square()).mean()


def fit(city, days, sites, epochs, dev_sites=None, *, over_weight=DEFAULT_OVER_WEIGHT):
    torch.manual_seed(SEED)
    model = GraphLSTM(city.static.shape[-1], city.temporal.shape[-1])
    model.fit_scales(city, days, sites)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.003)
    pairs = examples(days, sites)
    best_epoch, best_loss = 0, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(pairs)).split(128):
            d, s = pairs[indices].unbind(1)
            predicted_net = model(city, d, s)
            loss = objective(predicted_net, city.net_flow[d, s], over_weight) / model.y_scale.square()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        if dev_sites is not None:
            predicted_net = predict(model, city, days, dev_sites)
            truth = city.net_flow[days][:, dev_sites]
            criterion = objective(predicted_net, truth, over_weight).item()
            if criterion < best_loss - 0.001:
                best_epoch, best_loss = epoch, criterion
            if epoch - best_epoch >= 10:
                break
        if epoch % 5 == 0:
            print(f"epoch {epoch}/{epochs}", flush=True)
    return model, best_epoch


def metrics(truth, predictions, over_weight):
    net_error = (predictions - truth).flatten().double()
    over = net_error.clamp_min(0)
    under = (-net_error).clamp_min(0)
    return {"station_hours": len(net_error),
            "net_MSE": net_error.square().mean().item(),
            "net_RMSE": net_error.square().mean().sqrt().item(),
            "mean_observed_net": truth.double().mean().item(),
            "mean_predicted_net": predictions.double().mean().item(),
            "net_MAE": net_error.abs().mean().item(),
            "net_bias": net_error.mean().item(),
            "net_over_MAE_all_rows": over.mean().item(),
            "net_under_MAE_all_rows": under.mean().item(),
            "net_over_RMSE_all_rows": over.square().mean().sqrt().item(),
            "net_overprediction_rate": (net_error > 0).double().mean().item(),
            "asymmetric_net_MSE": (over_weight * over.square() + under.square()).mean().item()}


def validate_fold(a, historical, future, training_pool, test, over_weight):
    dev, core = np.split(training_pool, [len(a.stations) // 6])
    sources = np.sort(training_pool)
    _, epochs = fit(a, historical, core, 55, dev, over_weight=over_weight)
    model, _ = fit(a, historical, sources, epochs, over_weight=over_weight)
    forecast = predict(model, a, future, test)
    observed = a.net_flow[future][:, test]
    # 요일유형별·시간별 실측 평균. 검증 정답을 전혀 쓰지 않는 두 기준선.
    hourly_baseline = []
    for day in future:
        match = historical[a.off_day[historical] == a.off_day[day]]
        reference = a.net_flow[match][:, sources].mean((0, 1))
        hourly_baseline.append(reference.expand(len(test), -1))
    results = {"direct_net_gnn_lstm": forecast,
               "hour_daytype_baseline": torch.stack(hourly_baseline),
               "zero_net_baseline": torch.zeros_like(forecast)}
    fold = {"selected_epochs": epochs, "fit_stations": len(sources),
            "test_station_ids": a.stations[test].tolist(),
            "epoch_selection_station_ids": a.stations[dev].tolist(),
            "metrics": {k: metrics(observed, v, over_weight) for k, v in results.items()}}
    return observed, results, fold


def train(a, folder, over_weight):
    nd, ns = len(a.dates), len(a.stations)
    historical, future = np.arange(nd - 7), np.arange(nd - 7, nd)
    permutation = np.random.default_rng(SEED).permutation(ns)
    # 모든 대여소가 외부 검증에 정확히 한 번 등장. 내부 검증으로만 epoch 선택.
    groups = np.split(permutation, [i * (ns // 5) for i in range(1, 5)])
    observations, reports = [], []
    predictions = {}
    for i, test in enumerate(groups):
        print(f"A station fold {i + 1}/5", flush=True)
        pool = np.concatenate(groups[i+1:] + groups[:i])
        observed, results, fold = validate_fold(a, historical, future, pool, test, over_weight)
        observations.append(observed)
        reports.append(fold)
        for name, pred in results.items():
            predictions.setdefault(name, []).append(pred)
        print(fold["metrics"]["direct_net_gnn_lstm"], flush=True)
    observed = torch.cat(observations, dim=1)
    epochs = int(np.median([fold["selected_epochs"] for fold in reports]))
    loss_config = {"over_weight": over_weight, "net_definition": TARGET,
                   "net_error_definition": "predicted_net - actual_net",
                   "objective": "mse" if over_weight == 1 else "asymmetric_mse",
                   "forecast_role": "mean_estimate" if over_weight == 1 else "conservative_estimate"}
    report = {"target": TARGET, "seed": SEED, "final_epochs": epochs,
              "loss_config": loss_config,
              "train_dates": [str(a.dates[historical[i]].date()) for i in (0, -1)],
              "test_dates": [str(a.dates[future[i]].date()) for i in (0, -1)],
              "pooled": {k: metrics(observed, torch.cat(v, dim=1), over_weight)
                         for k, v in predictions.items()}, "folds": reports}
    print(json.dumps(report["pooled"]), flush=True)
    print(f"Training on all A stations and dates, {epochs} epochs", flush=True)
    full, _ = fit(a, np.arange(nd), np.arange(ns), epochs, over_weight=over_weight)
    checkpoint = folder / MODEL_NAME
    torch.save({"state": full.state_dict(), "ns": a.static.shape[-1],
                "nt": a.temporal.shape[-1], "k": K, "loss_config": loss_config,
                "seed": SEED, "epochs": epochs, "target": TARGET,
                "model_type": "direct_net_gnn_lstm"}, checkpoint.with_suffix(".tmp"))
    os.replace(checkpoint.with_suffix(".tmp"), checkpoint)
    (folder / "A_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    # Holdout predictions allow independent checking of both bias and allocation costs.
    station_order = [sid for fold in reports for sid in fold["test_station_ids"]]
    index = pd.MultiIndex.from_product([a.dates[future], station_order, range(24)],
                                       names=["date", "station_id", "hour"])
    audit = pd.DataFrame({"actual_net_flow": observed.flatten().numpy(),
                          "pred_net_flow": torch.cat(predictions["direct_net_gnn_lstm"], dim=1).flatten().numpy()},
                         index=index).reset_index()
    audit = audit[["date", "hour", "station_id", "actual_net_flow", "pred_net_flow"]]
    audit.sort_values(["date", "hour", "station_id"]).to_csv(
        folder / "A_검증예측.csv", index=False, encoding="utf-8-sig")


def validate_checkpoint(saved, expected_over_weight):
    """Prevent an old asymmetric checkpoint from silently serving a mean forecast."""
    if saved.get("model_type") != "direct_net_gnn_lstm" or saved.get("target") != TARGET:
        raise ValueError("직접 순유입 예측용 모델 파일이 필요합니다")
    if saved["k"] != K:
        raise ValueError("학습 모델과 그래프의 이웃 수가 다릅니다")
    saved_weight = saved.get("loss_config", {}).get("over_weight")
    if saved_weight != expected_over_weight:
        raise ValueError(f"모델의 over_weight={saved_weight}, 요청한 값={expected_over_weight}. "
                         "현재 설정으로 train을 다시 실행하거나 같은 설정의 모델을 사용하세요. "
                         "기존 비대칭 모델 재현은 --over-weight 3을 명시해야 합니다.")


def export_b(b, folder, expected_over_weight=DEFAULT_OVER_WEIGHT):
    saved = torch.load(folder / MODEL_NAME, map_location="cpu", weights_only=True)
    validate_checkpoint(saved, expected_over_weight)
    model = GraphLSTM(saved["ns"], saved["nt"])
    model.load_state_dict(saved["state"])
    net = predict(model, b, np.arange(len(b.dates)), np.arange(len(b.stations)))
    idx = pd.MultiIndex.from_product([b.dates, b.stations, range(24)],
                                     names=["date", "station_id", "hour"])
    out = pd.DataFrame(net.flatten().numpy(), idx,
                       columns=["pred_net_flow"]).reset_index()
    out = out[["date", "hour", "station_id", "pred_net_flow"]]
    out = out.sort_values(["date", "hour", "station_id"])
    if not torch.isfinite(net).all():
        raise ValueError("잘못된 예측값")
    path = folder / CSV_NAME
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.4f")
    os.replace(tmp, path)
    values = net.flatten().double()
    summary = {"target": TARGET, "loss_config": saved["loss_config"],
               "station_hours": len(out), "stations": len(b.stations), "days": len(b.dates),
               "mean_pred_net_flow": values.mean().item(),
               "mean_abs_pred_net_flow": values.abs().mean().item(),
               "negative_fraction": (values < 0).double().mean().item(),
               "sum_pred_net_flow": values.sum().item(),
               "note": "Forecast distribution only; B ground truth and actual score are unavailable."}
    (folder / "B_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"Exported {len(out)} station-hours to {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"], nargs="?", default="train")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1],
                        help="learn_data와 input_data를 포함한 저장소 루트")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--over-weight", type=float, default=DEFAULT_OVER_WEIGHT,
                        help="기본 1은 평균 순유입용 대칭 MSE. 1 초과는 과소예측을 유도하는 비교 실험용")
    args = parser.parse_args()
    if not np.isfinite(args.over_weight) or args.over_weight < 1:
        parser.error("--over-weight는 유한한 1 이상의 수여야 합니다")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.action == "train":
        a = read_city(args.data_dir / "learn_data", True)
        train(a, args.output_dir, args.over_weight)
    b = read_city(args.data_dir / "input_data", False)
    export_b(b, args.output_dir, args.over_weight)
