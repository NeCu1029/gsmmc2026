#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B 지역 시간별 예측 수요로 날짜별 자정 최적 배치 CSV 생성.

실행: python make_result.py

모든 입력/출력 경로는 스크립트 위치가 아니라 실행 작업 디렉터리(CWD) 기준이다.
  예측:    code/demand_asymmetric/수요예측_비대칭_결과/B지역_시간별_대여반납_예측.csv
  제출 틀: submit_temp.csv
  수용량:  input_data/spots.csv
  결과:    result.csv

예측 열: date,hour,station_id,pred_rentals,pred_returns
제출 틀: station_id,D01,...,D30
수용량:   station_id,x_km,y_km,capacity

평가 벌점: 20*N + mean_d sum_i sum_t (6*u^2 + 5*o + e)
매일 같은 N을 사용하고 각 대여소의 초기 배치는 0~capacity 사이 정수.
실수형 pred_rentals/pred_returns는 반올림하지 않고 그대로 투입한다.

필요 패키지: numpy, numba  (pip install numpy numba)
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np
from numba import njit

DAYS = 30
HOURS = 24
# Path는 실행 위치(CWD)를 기준으로 해석한다. __file__을 기준으로 하지 않는다.
PRED_PATH = Path("code/demand_asymmetric/수요예측_비대칭_결과/B지역_시간별_대여반납_예측.csv")
TEMPLATE_PATH = Path("submit_temp.csv")
STATIONS_PATH = Path("input_data/spots.csv")
OUTPUT_PATH = Path("result.csv")


def read_dict_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"비어 있는 CSV: {path}")
        return [s.strip() for s in reader.fieldnames], list(reader)


def load_template(path: Path) -> tuple[list[str], list[str]]:
    fields, rows = read_dict_rows(path)
    expected = ["station_id"] + [f"D{d:02d}" for d in range(1, DAYS + 1)]
    if any(col not in fields for col in expected):
        raise ValueError(f"{path}: 필요한 열은 {expected}입니다.")
    ids = [r["station_id"].strip() for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("제출 틀의 대여소 코드가 비어 있거나 중복됩니다.")
    if len(ids) != 159:
        print(f"[참고] 제출 틀의 대여소 수는 {len(ids)}개입니다 (대회 B지역은 159개).")
    return fields, ids


def load_hourly(path: Path, stations: list[str]) -> tuple[np.ndarray, np.ndarray]:
    fields, rows = read_dict_rows(path)
    required = {"date", "hour", "station_id", "pred_rentals", "pred_returns"}
    if not required.issubset(set(fields)):
        raise ValueError(f"예측 CSV에 다음 열이 필요합니다: {sorted(required)}")

    m = len(stations)
    station_index = {sid: i for i, sid in enumerate(stations)}
    rentals = np.zeros((DAYS, m, HOURS), dtype=np.float64)
    returns = np.zeros_like(rentals)
    seen = np.zeros((DAYS, m, HOURS), dtype=np.bool_)

    for line_no, r in enumerate(rows, start=2):
        try:
            dt = date.fromisoformat(r["date"].strip())
            if (dt.year, dt.month) != (2026, 6):
                raise ValueError("2026년 6월 날짜가 아닙니다")
            d = dt.day - 1
            h = int(r["hour"])
            sid = r["station_id"].strip()
            i = station_index[sid]
            rent = float(r["pred_rentals"])
            ret = float(r["pred_returns"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"예측 CSV {line_no}행 데이터 오류: {exc}") from exc
        if not (0 <= h < HOURS):
            raise ValueError(f"예측 CSV {line_no}행: hour는 0~23이어야 합니다.")
        if seen[d, i, h]:
            raise ValueError(f"예측 CSV {line_no}행: ({dt}, {sid}, {h}) 중복")
        if not (np.isfinite(rent) and np.isfinite(ret) and rent >= 0 and ret >= 0):
            raise ValueError(f"예측 CSV {line_no}행: 대여·반납 예측값은 유한한 비음수여야 합니다.")
        rentals[d, i, h] = rent
        returns[d, i, h] = ret
        seen[d, i, h] = True

    if not seen.all():
        missing = np.argwhere(~seen)
        first = missing[0]
        raise ValueError(
            f"예측 CSV에 {len(missing)}개 시간 기록이 없습니다. "
            f"첫 누락: 6월 {int(first[0])+1}일, {stations[int(first[1])]}, {int(first[2])}시"
        )
    return rentals, returns


def load_capacities(path: Path, stations: list[str]) -> np.ndarray:
    fields, rows = read_dict_rows(path)
    if "station_id" not in fields or "capacity" not in fields:
        raise ValueError(f"{path}: station_id, capacity 열이 필요합니다.")
    lookup: dict[str, int] = {}
    for line_no, row in enumerate(rows, start=2):
        sid = row["station_id"].strip()
        try:
            raw = float(row["capacity"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"수용량 CSV {line_no}행 {sid}: 수용량 숫자가 필요합니다.") from exc
        if not np.isfinite(raw) or raw < 0 or not raw.is_integer():
            raise ValueError(f"수용량 CSV {line_no}행 {sid}: 수용량은 0 이상의 정수여야 합니다.")
        if sid in lookup:
            raise ValueError(f"수용량 CSV의 중복 대여소 코드: {sid}")
        lookup[sid] = int(raw)
    missing = set(stations) - set(lookup)
    if missing:
        raise ValueError(f"수용량 CSV에 {len(missing)}개 대여소가 빠졌습니다: {sorted(missing)[:8]}")
    caps = np.asarray([lookup[sid] for sid in stations], dtype=np.int32)
    if int(caps.sum()) < 1:
        raise ValueError("총 거치대 수가 1보다 작아서 유효한 배치를 만들 수 없습니다.")
    return caps


@njit(cache=True)
def compute_costs(rentals: np.ndarray, returns: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """cost[d, i, x] = 24시간 벌점 (x는 정수, 시간별 수요는 실수 그대로)."""
    days, stations, _ = rentals.shape
    max_c = int(np.max(caps))
    cost = np.full((days, stations, max_c + 1), np.inf, dtype=np.float64)
    for d in range(days):
        for i in range(stations):
            c = caps[i]
            p = c // 2
            for x in range(c + 1):
                b = float(x)
                total = 0.0
                for t in range(HOURS):
                    z = b + returns[d, i, t] - rentals[d, i, t]
                    u = max(0.0, -z)
                    o = max(0.0, z - c)
                    b = min(float(c), max(0.0, z))
                    e = max(0.0, b - p)
                    total += 6.0 * u * u + 5.0 * o + e
                cost[d, i, x] = total
    return cost


@njit(cache=True)
def dp_all_totals(cost_one_day: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """하루 최소 벌점 F_d[N]을 모든 가능한 N에 대해 정확히 계산한다."""
    c_total = int(np.sum(caps))
    prev = np.full(c_total + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    max_before = 0
    for i in range(len(caps)):
        c = int(caps[i])
        nxt = np.full(c_total + 1, np.inf, dtype=np.float64)
        for n in range(max_before + 1):
            v = prev[n]
            if not np.isfinite(v):
                continue
            for x in range(c + 1):
                score = v + cost_one_day[i, x]
                if score < nxt[n + x]:
                    nxt[n + x] = score
        prev = nxt
        max_before += c
    return prev


@njit(cache=True)
def dp_reconstruct(cost_one_day: np.ndarray, caps: np.ndarray, n_target: int) -> np.ndarray:
    """N대를 정확히 쓰는 하루 최적 배치를 역추적한다."""
    station_count = len(caps)
    prev = np.full(n_target + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    chosen = np.full((station_count, n_target + 1), -1, dtype=np.int32)
    max_before = 0
    for i in range(station_count):
        c = min(int(caps[i]), n_target)
        nxt = np.full(n_target + 1, np.inf, dtype=np.float64)
        for n in range(min(max_before, n_target) + 1):
            v = prev[n]
            if not np.isfinite(v):
                continue
            for x in range(min(c, n_target - n) + 1):
                score = v + cost_one_day[i, x]
                if score < nxt[n + x]:
                    nxt[n + x] = score
                    chosen[i, n + x] = x
        prev = nxt
        max_before += int(caps[i])

    if not np.isfinite(prev[n_target]):
        raise ValueError("이 총 자전거 수에 대한 유효 배치가 없습니다.")
    alloc = np.zeros(station_count, dtype=np.int32)
    remaining = n_target
    for i in range(station_count - 1, -1, -1):
        x = chosen[i, remaining]
        if x < 0:
            raise ValueError("DP 경로 역추적 실패")
        alloc[i] = x
        remaining -= x
    if remaining != 0:
        raise ValueError("DP 역추적 결과가 총 자전거 수와 일치하지 않습니다.")
    return alloc


def optimize(rentals: np.ndarray, returns: np.ndarray, caps: np.ndarray):
    print("[1/3] 대여소·날짜별 배치 수에 따른 벌점표 계산...")
    cost = compute_costs(rentals, returns, caps)
    total_capacity = int(caps.sum())
    print(f"[2/3] 날짜별 동적 계획법 실행 (가능한 총 자전거 수: 1~{total_capacity})...")
    best_per_day = np.empty((DAYS, total_capacity + 1), dtype=np.float64)
    for d in range(DAYS):
        best_per_day[d] = dp_all_totals(cost[d], caps)
        print(f"  {d+1:02d}/{DAYS}일 계산 완료", flush=True)

    scores = 20.0 * np.arange(total_capacity + 1) + best_per_day.mean(axis=0)
    scores[0] = np.inf  # 전체 N=0은 제출 불가
    n_star = int(np.argmin(scores))

    print(f"[3/3] 최적 총 자전거 수 N={n_star}로 날짜별 배치 역추적...")
    allocations = np.empty((DAYS, len(caps)), dtype=np.int32)
    for d in range(DAYS):
        allocations[d] = dp_reconstruct(cost[d], caps, n_star)

    assert np.all(allocations.sum(axis=1) == n_star)
    assert np.all((allocations >= 0) & (allocations <= caps[None, :]))
    costs_reconstructed = np.array([
        sum(cost[d, i, int(allocations[d, i])] for i in range(len(caps)))
        for d in range(DAYS)
    ])
    if not np.allclose(costs_reconstructed, best_per_day[:, n_star], rtol=1e-9, atol=1e-6):
        raise RuntimeError("최적 DP 결과와 역추적 배치의 벌점이 다릅니다.")
    score = 20 * n_star + costs_reconstructed.mean()
    return allocations, n_star, float(score)


def write_result(path: Path, template: Path, stations: list[str], alloc: np.ndarray):
    fields, rows = read_dict_rows(template)
    lookup = {sid: i for i, sid in enumerate(stations)}
    for row in rows:
        i = lookup[row["station_id"].strip()]
        for d in range(DAYS):
            row[f"D{d+1:02d}"] = str(int(alloc[d, i]))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    for label, path in (
        ("예측", PRED_PATH),
        ("제출 틀", TEMPLATE_PATH),
        ("대여소 수용량", STATIONS_PATH),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} 파일을 찾을 수 없습니다: {path.resolve()}\n"
                "경로는 make_result.py의 위치가 아닌 명령어 실행 위치를 기준으로 합니다."
            )

    _, stations = load_template(TEMPLATE_PATH)
    caps = load_capacities(STATIONS_PATH, stations)
    rentals, returns = load_hourly(PRED_PATH, stations)
    alloc, n_star, score = optimize(rentals, returns, caps)
    write_result(OUTPUT_PATH, TEMPLATE_PATH, stations, alloc)
    print(f"저장 완료: {OUTPUT_PATH.resolve()}")
    print(f"총 자전거 수 N={n_star} (매일 동일), 예측 평가 벌점={score:,.4f}")


if __name__ == "__main__":
    main()
