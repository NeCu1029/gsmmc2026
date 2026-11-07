#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""대여인기·반납인기 규칙 모델 (팀 메모).

1. 매일 같은 배치를 쓴다.
2. 출근 기준(평일) 0시부터 12시까지 누적 순유입(반납 - 대여)을 net, e = floor(capacity/2) 라 할 때
     대여인기: net <= -e
     반납인기: net >= +e   (대여인기보다 판정 우선도 낮음: 둘 다면 대여인기)
     비인기: 나머지
3. 로지스틱 회귀로 두 인기를 각각 학습한다. 입력 = 대여소 간 거리 중앙값 + 접근성.
4. 배치: 대여인기 = 거치대 수(max), 반납인기 = 0대, 비인기 = 1대.

저장소 루트에서 실행:  python code/demand_rule/rule_model.py
결과: code/demand_rule/results/B_배치표.csv (제출 형식), B_대여인기_예측.csv

공개일 사용:  python code/demand_rule/rule_model.py --reveal open_data/B지역_공개일_D10_2026-06-10.csv
  로지스틱 대신 공개일 B 실측의 net으로 같은 기준을 적용하고, 같은 배치를 모든 날에 쓴다.
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
# 대여인기는 거치대 수(max)
X_RETURN = 0       # 반납인기
X_OTHER = 1        # 비인기


def classify(net, capacity):
    """net(0~12시 누적 순유입)과 거치대 수로 대여인기·반납인기를 판정한다."""
    e = np.asarray(capacity) // 2
    rental = np.asarray(net) <= -e
    ret = (np.asarray(net) >= e) & ~rental          # 대여인기가 우선
    return rental, ret


def allocate(rental, ret, capacity):
    capacity = np.asarray(capacity)
    ret = ret & ~rental                              # 대여인기가 우선
    x = np.full(len(capacity), X_OTHER, dtype=np.int64)
    x[ret] = X_RETURN
    x[rental] = capacity[rental]
    x = np.minimum(x, capacity)
    if x.sum() < 1:                                  # 대회 규칙: N >= 1
        x[np.argmax(capacity)] = 1
    return x


def label_name(rental, ret):
    return np.where(rental, "대여인기", np.where(ret, "반납인기", "비인기"))


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


def morning_labels(spots):
    demand = pd.read_csv("learn_data/demand.csv")
    cal = pd.read_csv("learn_data/calclim.csv").groupby("date")[["is_weekend", "is_holiday"]].max()
    weekdays = cal.index[(cal.is_weekend == 0) & (cal.is_holiday == 0)]
    morning = demand[demand.date.isin(weekdays) & (demand.hour < NOON)]
    net = (morning.returns - morning.rentals).groupby([morning.date, morning.station_id]).sum()
    net = net.groupby("station_id").mean().loc[spots.index]   # 평일 평균 0~12시 누적 순유입
    rental, ret = classify(net.to_numpy(), spots.capacity.to_numpy())
    return rental.astype(int), ret.astype(int), net


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
    cap = spots_b.capacity.to_numpy()
    rental, ret = classify(net.to_numpy(), cap)
    x = allocate(rental, ret, cap)
    out = write_table(spots_b, x, "B_배치표_공개일.csv")
    pd.DataFrame({"station_id": spots_b.index, "capacity": cap, "net_0_12h": net.to_numpy(),
                  "class": label_name(rental, ret), "x": x}).to_csv(
        OUT / "B_대여인기_공개일.csv", index=False, encoding="utf-8-sig")
    other = ~rental & ~ret
    print(f"공개일 {Path(path).name} 기준 B: 대여인기 {int(rental.sum())}곳(max), "
          f"반납인기 {int(ret.sum())}곳(0대), 비인기 {int(other.sum())}곳(1대) → N = {int(x.sum())} (매일 동일)")
    print(f"저장: {out}")


def main():
    spots_a, Xa = station_features("learn_data")
    spots_b, Xb = station_features("input_data")
    y_rental, y_return, net = morning_labels(spots_a)
    Xb = Xb[Xa.columns]
    cap = spots_b.capacity.to_numpy()

    models, prob = {}, {}
    for name, y in (("rental", y_rental), ("return", y_return)):
        models[name] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xa, y)
        prob[name] = models[name].predict_proba(Xb)[:, 1]
    rental = prob["rental"] >= 0.5
    ret = (prob["return"] >= 0.5) & ~rental            # 대여인기가 우선
    x = allocate(rental, ret, cap)

    out = write_table(spots_b, x, "B_배치표.csv")
    pd.DataFrame({"station_id": spots_b.index, "capacity": cap,
                  "p_rental_popular": prob["rental"].round(4), "p_return_popular": prob["return"].round(4),
                  "class": label_name(rental, ret), "x": x}).to_csv(
        OUT / "B_대여인기_예측.csv", index=False, encoding="utf-8-sig")

    for name, y in (("rental", y_rental), ("return", y_return)):
        m = models[name]
        print(f"A {'대여' if name == 'rental' else '반납'}인기 {int(y.sum())}/{len(y)}곳 "
              f"(학습 정확도 {m.score(Xa, y):.3f}) 계수: "
              + ", ".join(f"{c}={v:+.2f}" for c, v in zip(Xa.columns, m[-1].coef_[0])))
    other = ~rental & ~ret
    print(f"B: 대여인기 {int(rental.sum())}곳(max), 반납인기 {int(ret.sum())}곳(0대), "
          f"비인기 {int(other.sum())}곳(1대) → N = {int(x.sum())} (매일 동일)")
    print(f"저장: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reveal", help="공개일 CSV (날짜,요일,시,station_id,rentals,returns)")
    args = parser.parse_args()
    from_reveal(args.reveal) if args.reveal else main()
