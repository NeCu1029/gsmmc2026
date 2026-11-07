#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""대여인기 규칙 모델 (팀 메모 그대로).

1. 매일 같은 배치를 쓴다.
2. 대여인기 = 0시부터 12시까지 누적 순유입(반납 - 대여)이 -floor(capacity/2) 이하 (출근 기준 = 평일).
3. 로지스틱 회귀로 대여인기를 학습한다. 입력 = 대여소 간 거리 중앙값 + 접근성.
4. 대여인기 대여소에는 거치대 수(max), 나머지는 0대.

저장소 루트에서 실행:  python code/demand_rule/rule_model.py
결과: code/demand_rule/results/B_배치표.csv (제출 형식), B_대여인기_예측.csv

공개일 사용:  python code/demand_rule/rule_model.py --reveal open_data/B지역_공개일_D10_2026-06-10.csv
  로지스틱 대신 공개일 B 실측으로 같은 기준(0~12시 누적 순유입 <= -floor(capacity/2))을 적용해
  대여인기를 정하고, 같은 배치를 모든 날에 쓴다.
  결과: code/demand_rule/results/B_배치표_공개일.csv, B_대여인기_공개일.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NOON = 12  # 0시 ~ 11시 (12시 전까지) 누적
OUT = Path("code/demand_rule/results")


def station_features(folder):
    spots = pd.read_csv(f"{folder}/spots.csv").set_index("station_id").sort_index()
    ids = spots.index
    dist = (pd.read_csv(f"{folder}/dist.csv")
            .pivot(index="from_station_id", columns="to_station_id", values="distance_km")
            .loc[ids, ids].to_numpy(float, copy=True))
    np.fill_diagonal(dist, np.nan)
    access = pd.read_csv(f"{folder}/nearby.csv").set_index("station_id").loc[ids]
    X = access.select_dtypes("number").copy()          # 접근성 (버스·지하철·시설 표지)
    X.insert(0, "distance_median_km", np.nanmedian(dist, axis=1))  # 다른 대여소까지 거리 중앙값
    return spots, X


def rental_popular_labels(spots):
    demand = pd.read_csv("learn_data/demand.csv")
    cal = pd.read_csv("learn_data/calclim.csv").groupby("date")[["is_weekend", "is_holiday"]].max()
    weekdays = cal.index[(cal.is_weekend == 0) & (cal.is_holiday == 0)]
    morning = demand[demand.date.isin(weekdays) & (demand.hour < NOON)]
    net = (morning.returns - morning.rentals).groupby([morning.date, morning.station_id]).sum()
    net = net.groupby("station_id").mean().loc[spots.index]   # 평일 평균 0~12시 누적 순유입
    return (net <= -(spots.capacity // 2)).astype(int), net


def write_table(spots_b, x, name):
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv("submit_temp.csv", encoding="utf-8-sig")
    lookup = dict(zip(spots_b.index, x))
    for d in range(1, 31):
        table[f"D{d:02d}"] = table.station_id.map(lookup).astype(int)   # 매일 같은 배치
    table.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    return OUT / name


def from_reveal(path):
    spots_b = pd.read_csv("input_data/spots.csv").set_index("station_id").sort_index()
    day = pd.read_csv(path, encoding="utf-8-sig").rename(columns={"시": "hour"})
    morning = day[day.hour < NOON]
    net = (morning.returns - morning.rentals).groupby(morning.station_id).sum()
    net = net.reindex(spots_b.index, fill_value=0)
    popular = (net <= -(spots_b.capacity // 2)).to_numpy()
    x = np.where(popular, spots_b.capacity.to_numpy(), 0)
    if x.sum() < 1:
        x[np.argmax(spots_b.capacity.to_numpy())] = 1
    out = write_table(spots_b, x, "B_배치표_공개일.csv")
    pd.DataFrame({"station_id": spots_b.index, "capacity": spots_b.capacity,
                  "net_0_12h": net.to_numpy(), "rental_popular": popular, "x": x}).to_csv(
        OUT / "B_대여인기_공개일.csv", index=False, encoding="utf-8-sig")
    print(f"공개일 {Path(path).name} 기준 B 대여인기 {int(popular.sum())}/{len(popular)}곳 "
          f"→ N = {int(x.sum())} (매일 동일)")
    print(f"저장: {out}")


def main():
    spots_a, Xa = station_features("learn_data")
    spots_b, Xb = station_features("input_data")
    y, net = rental_popular_labels(spots_a)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(Xa, y)
    popular = model.predict(Xb[Xa.columns]).astype(bool)

    x = np.where(popular, spots_b.capacity.to_numpy(), 0)
    if x.sum() < 1:                      # 대회 규칙: N >= 1
        x[np.argmax(spots_b.capacity.to_numpy())] = 1
    N = int(x.sum())

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv("submit_temp.csv", encoding="utf-8-sig")
    lookup = dict(zip(spots_b.index, x))
    for d in range(1, 31):
        table[f"D{d:02d}"] = table.station_id.map(lookup).astype(int)   # 매일 같은 배치
    table.to_csv(OUT / "B_배치표.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"station_id": spots_b.index, "capacity": spots_b.capacity,
                  "p_rental_popular": model.predict_proba(Xb[Xa.columns])[:, 1].round(4),
                  "rental_popular": popular, "x": x}).to_csv(
        OUT / "B_대여인기_예측.csv", index=False, encoding="utf-8-sig")

    coef = model[-1].coef_[0]
    print(f"A 대여인기 {int(y.sum())}/{len(y)}곳 (학습 정확도 {model.score(Xa, y):.3f})")
    print("계수(표준화):", ", ".join(f"{c}={v:+.2f}" for c, v in zip(Xa.columns, coef)))
    print(f"B 대여인기 {int(popular.sum())}/{len(popular)}곳 → N = {N} (매일 동일)")
    print(f"저장: {OUT / 'B_배치표.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reveal", help="공개일 CSV (날짜,요일,시,station_id,rentals,returns)")
    args = parser.parse_args()
    from_reveal(args.reveal) if args.reveal else main()
