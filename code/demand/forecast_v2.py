"""공유자전거 시간별 수요 모델 v2. 저장소의 learn_data, input_data CSV를 사용한다.

재현 환경: Python 3.12, torch 2.15.0 (CPU), numpy 2.3.5, pandas 3.0.1.
python code/demand/forecast_v2.py train     # A 검증, 전체 A 재학습, B 예측
python code/demand/forecast_v2.py predict   # 저장한 모델로 B CSV만 다시 생성
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
TARGETS = ["rentals", "returns"]
ACCESS = ["nearest_bus_distance_m", "bus_stop_count_500m",
          "nearest_subway_distance_m", "subway_station_count_500m",
          "subway_station_count_1000m", "school_name_flag", "park_name_flag",
          "commercial_name_flag"]
WEATHER = ["temperature_c", "humidity_pct", "precipitation_mm", "wind_speed_kmh"]
CSV_NAME = "B지역_시간별_대여반납_예측.csv"
MODEL_NAME = "bike_demand.pt"


@dataclass
class City:
    stations: pd.Index
    dates: pd.DatetimeIndex
    static: torch.Tensor
    adjacency: torch.Tensor
    temporal: torch.Tensor        # day, hour, feature
    capacity: torch.Tensor
    off_day: np.ndarray
    counts: torch.Tensor | None   # day, station, hour, [rentals, returns]


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
    counts = None
    if labelled:
        observations = read("demand.csv")
        observations["date"] = pd.to_datetime(observations.date)
        keys = ["date", "station_id", "hour"]
        expected = pd.MultiIndex.from_product([dates, ids, range(24)], names=keys)
        y = unique_index(observations, keys)
        if len(y) != len(expected):
            raise ValueError("수요 행 수가 날짜 × 대여소 × 24와 다릅니다")
        values = y.loc[expected, TARGETS].to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("수요에 누락 또는 음수가 있습니다")
        counts = tensor(values.reshape(len(dates), len(ids), 24, 2))
    if not torch.isfinite(static).all() or not torch.isfinite(temporal).all():
        raise ValueError("대여소·달력·기상 입력에 결측 또는 무한값이 있습니다")
    return City(ids, dates, static, adjacency, temporal, tensor(stations.capacity),
                off.reshape(-1, 24)[:, 0], counts)


class GraphLSTM(nn.Module):
    def __init__(self, ns, nt):
        super().__init__()
        self.register_buffer("s_mean", torch.zeros(ns))
        self.register_buffer("s_scale", torch.ones(ns))
        self.register_buffer("t_mean", torch.zeros(nt))
        self.register_buffer("t_scale", torch.ones(nt))
        self.g1 = nn.Linear(2 * ns, 16)
        self.g2 = nn.Linear(32, 12)
        self.station = nn.Sequential(nn.Linear(ns + 12, 16), nn.Tanh())
        self.init_state = nn.Linear(16, 48)
        self.lstm = nn.LSTM(nt + 16, 24, batch_first=True)
        self.dropout = nn.Dropout(0.12)
        self.head = nn.Linear(24 + 16, 2)

    def fit_scales(self, city, days, sites):
        for x, mean, scale in [(city.static[sites], self.s_mean, self.s_scale),
                               (city.temporal[days].flatten(0, 1), self.t_mean, self.t_scale)]:
            mean.copy_(x.mean(0))
            scale.copy_(x.std(0, unbiased=False).clamp_min(0.2))

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
        # 수용량은 초기 수요 규모에 대한 단서. 시간당 회전 수요에 상한은 두지 않는다.
        offset = 0.65 * (city.capacity[sites] / 15).log() + 1.0
        log_rate = self.head(torch.cat((self.dropout(out), repeat), -1))
        return (log_rate + offset[:, None, None]).clamp(-9, 8)


def examples(days, sites):
    return torch.cartesian_prod(torch.as_tensor(days), torch.as_tensor(sites))


@torch.no_grad()
def predict(model, city, days, sites):
    model.eval()
    batches = examples(days, sites).split(256)
    rates = [model(city, pairs[:, 0], pairs[:, 1]).exp() for pairs in batches]
    return torch.cat(rates).reshape(len(days), len(sites), 24, 2)


def fit(city, days, sites, epochs, dev_sites=None):
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
            log_rate = model(city, d, s)
            loss = (log_rate.exp() - city.counts[d, s] * log_rate).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        if dev_sites is not None:
            rates = predict(model, city, days, dev_sites)
            truth = city.counts[days][:, dev_sites]
            criterion = (rates - truth * rates.log()).mean().item()
            if criterion < best_loss - 0.001:
                best_epoch, best_loss = epoch, criterion
            if epoch - best_epoch >= 10:
                break
        if epoch % 5 == 0:
            print(f"epoch {epoch}/{epochs}", flush=True)
    return model, best_epoch


def metrics(truth, rates):
    diff = (rates - truth).reshape(-1, 2).double()
    return {"station_hours": len(diff), "MAE": diff.abs().mean(0).tolist(),
            "RMSE": diff.square().mean(0).sqrt().tolist(),
            "mean_observed": truth.reshape(-1, 2).double().mean(0).tolist(),
            "net_MAE": (diff[:, 0] - diff[:, 1]).abs().mean().item()}


def validate_fold(a, historical, future, training_pool, test):
    dev, core = np.split(training_pool, [len(a.stations) // 6])
    sources = np.sort(training_pool)
    _, epochs = fit(a, historical, core, 55, dev)
    model, _ = fit(a, historical, sources, epochs)
    forecast = predict(model, a, future, test)
    observed = a.counts[future][:, test]
    # 요일유형별·시간별 실측 평균. 검증 정답을 전혀 쓰지 않는 두 기준선.
    rates = []
    capacity_rates = []
    for day in future:
        match = historical[a.off_day[historical] == a.off_day[day]]
        reference = a.counts[match][:, sources].mean((0, 1))
        rates.append(reference.expand(len(test), -1, -1))
        capacity_rates.append(reference * (a.capacity[test] / a.capacity[sources].mean())[:, None, None])
    results = {"gnn_lstm": forecast, "hour_daytype_baseline": torch.stack(rates),
               "capacity_hour_daytype_baseline": torch.stack(capacity_rates)}
    fold = {"selected_epochs": epochs, "fit_stations": len(sources),
            "test_station_ids": a.stations[test].tolist(),
            "epoch_selection_station_ids": a.stations[dev].tolist(),
            "metrics": {k: metrics(observed, v) for k, v in results.items()}}
    return observed, results, fold


def train(a, folder):
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
        observed, results, fold = validate_fold(a, historical, future, pool, test)
        observations.append(observed)
        reports.append(fold)
        for name, pred in results.items():
            predictions.setdefault(name, []).append(pred)
        print(fold["metrics"]["gnn_lstm"], flush=True)
    observed = torch.cat(observations, dim=1)
    epochs = int(np.median([fold["selected_epochs"] for fold in reports]))
    report = {"target_order": TARGETS, "seed": SEED, "final_epochs": epochs,
              "train_dates": [str(a.dates[historical[i]].date()) for i in (0, -1)],
              "test_dates": [str(a.dates[future[i]].date()) for i in (0, -1)],
              "pooled": {k: metrics(observed, torch.cat(v, dim=1))
                         for k, v in predictions.items()}, "folds": reports}
    print(json.dumps(report["pooled"]), flush=True)
    print(f"Training on all A stations and dates, {epochs} epochs", flush=True)
    full, _ = fit(a, np.arange(nd), np.arange(ns), epochs)
    checkpoint = folder / MODEL_NAME
    torch.save({"state": full.state_dict(), "ns": a.static.shape[-1],
                "nt": a.temporal.shape[-1], "k": K}, checkpoint.with_suffix(".tmp"))
    os.replace(checkpoint.with_suffix(".tmp"), checkpoint)
    (folder / "A_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")


def export_b(b, folder):
    saved = torch.load(folder / MODEL_NAME, map_location="cpu", weights_only=True)
    if saved["k"] != K:
        raise ValueError("학습 모델과 그래프의 이웃 수가 다릅니다")
    model = GraphLSTM(saved["ns"], saved["nt"])
    model.load_state_dict(saved["state"])
    rates = predict(model, b, np.arange(len(b.dates)), np.arange(len(b.stations)))
    idx = pd.MultiIndex.from_product([b.dates, b.stations, range(24)],
                                     names=["date", "station_id", "hour"])
    out = pd.DataFrame(rates.reshape(-1, 2).numpy(), idx,
                       columns=["pred_rentals", "pred_returns"]).reset_index()
    out = out[["date", "hour", "station_id", "pred_rentals", "pred_returns"]]
    out = out.sort_values(["date", "hour", "station_id"])
    if not torch.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("잘못된 예측값")
    path = folder / CSV_NAME
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.4f")
    os.replace(tmp, path)
    print(f"Exported {len(out)} station-hours to {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"], nargs="?", default="train")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1],
                        help="learn_data와 input_data를 포함한 저장소 루트")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "수요예측_결과")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.action == "train":
        a = read_city(args.data_dir / "learn_data", True)
        train(a, args.output_dir)
    b = read_city(args.data_dir / "input_data", False)
    export_b(b, args.output_dir)
