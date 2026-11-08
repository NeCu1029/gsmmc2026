"""공개일 보정 개선안 평가 + 배치 생성 (기존 파일은 수정하지 않는다).

기존 bike_public_transfer.py 의 학습된 모델(public_inventory.pt)을 그대로 쓰고,
공개일 보정 방식만 바꿔 본다. B 에서 할 수 있는 정직한 검증은 '공개일 하나만 보고 보정 →
다른 공개일 실측으로 채점'(leave-one-reveal-out) 뿐이므로 그것으로 후보를 비교한다.

  python code/demand_neural_final/public_transfer_v2.py evaluate --output-dir <모델 폴더>
  python code/demand_neural_final/public_transfer_v2.py export --output-dir <모델 폴더> --weekday-weight 1.0 --shrink 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import bike_inventory_net as inventory
import bike_neural_final as forecast
from bike_public_transfer import increments, read_public
from neural_allocation import dp_all_totals, dp_reconstruct, optimize, realized_score

ROOT = Path(__file__).resolve().parent


def load_models(output):
    saved = torch.load(output / "public_inventory.pt", map_location="cpu", weights_only=True)
    models = []
    for state in saved["states"]:
        m = inventory.InventoryNet(saved["ns"], saved["nt"])
        m.load_state_dict(state)
        models.append(m)
    return models, saved["config"]


def calibrate(pred, pub_days, pub_inc, off, weekday_weight, offday_weight, shrink, mode="residual"):
    """pred: (30, S, Xmax) 예측 한계비용. pub_inc: (P, S, Xmax) 공개일 실측 한계비용.

    residual: 예측 + w * 평균(실측-예측)                (기존 방식, shrink=0)
    copy    : 같은 유형 날은 공개일 실측 한계비용 평균 + 예측의 날짜별 변화
    shrink  : 대여소별 잔차를 공개일 수 / (공개일 수 + shrink) 만큼만 반영 (잡음 줄이기)
    """
    result = pred.copy()
    for kind, weight in ((False, weekday_weight), (True, offday_weight)):
        matched = off[pub_days] == kind
        if not matched.any():
            matched = np.ones(len(pub_days), dtype=bool)
        k = matched.sum()
        residual = (pub_inc[matched] - pred[pub_days][matched]).mean(0)
        factor = weight * k / (k + shrink)
        result[off == kind] += factor * residual
    return result


def allocate(costs_inc, caps, fixed_n=None, off=None, offday_cap=False, lam=20.0):
    """공통 N DP. offday_cap=True 이면 주말·공휴일에는 대여소당 floor(c/2) 이하로만 둔다
    (남는 자전거를 넓게 분산해 적정 잔여량 초과 E 를 피한다). lam 은 자전거 1대 비용(공식 20)."""
    if off is None or not offday_cap:
        day_caps = [caps] * len(costs_inc)
    else:
        day_caps = [caps // 2 if off[d] else caps for d in range(len(costs_inc))]
    tables = [inventory.cost_table(costs_inc[d][None], c)[0] for d, c in enumerate(day_caps)]
    if fixed_n is None:
        size = int(caps.sum()) + 1
        totals = np.full((len(tables), size), np.inf)
        for d, (t, c) in enumerate(zip(tables, day_caps)):
            v = dp_all_totals(t, c)
            totals[d, :len(v)] = v
        score = lam * np.arange(size) + totals.mean(0)
        score[0] = np.inf
        fixed_n = int(np.argmin(score))
    alloc = np.stack([dp_reconstruct(t, c, fixed_n) for t, c in zip(tables, day_caps)])
    if (alloc < 0).any() or (alloc > caps).any() or not np.all(alloc.sum(1) == fixed_n):
        raise ValueError("Invalid allocation")
    return alloc


def loro(data, pred, days_pub, inc_pub, truth_pub, caps, off, **kw):
    """공개일 하나를 숨기고 나머지로 보정 → 숨긴 날 실측으로 채점."""
    out = []
    for h in range(len(days_pub)):
        keep = [j for j in range(len(days_pub)) if j != h]
        cal = calibrate(pred, days_pub[keep], inc_pub[keep], off, kw["w_day"], kw["w_off"], kw["shrink"])
        cal[days_pub[keep]] = inc_pub[keep]                     # 보이는 공개일은 정확한 비용
        alloc = allocate(cal, caps, kw.get("fixed_n"))
        r = realized_score(truth_pub[h][None], alloc[days_pub[h]][None], caps)
        out.append((int(alloc[0].sum()), r["score"]))
    return out


def evaluate(args):
    torch.set_num_threads(2)
    data = forecast.read_data(args.data_dir, "B")
    days_pub, truth, _ = read_public(args.public_dir, data)
    caps = data.capacity
    models, config = load_models(args.output_dir)
    alldays, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    pred = inventory.ensemble(models, data, alldays, sites)
    inc_pub = increments(truth, caps)
    off = data.city.off_day
    names = [data.city.dates[d].strftime("%m/%d") for d in days_pub]
    print(f"기존 설정: {config}   공개일: {names}")

    def show(label, res):
        s = " | ".join(f"{names[1 - i]}→{names[i]}: N={n} S_d={sc:,.0f}" for i, (n, sc) in enumerate(res))
        print(f"{label:34s} {s} | 합 {sum(sc for _, sc in res):,.0f}")

    base = dict(w_day=config["weekday_weight"], w_off=config["offday_weight"], shrink=0.0)
    show("기존 (w평일=%.1f, w휴일=%.1f)" % (base["w_day"], base["w_off"]), loro(data, pred, days_pub, inc_pub, truth, caps, off, **base))
    show("보정 없음 (신경망만)", loro(data, pred, days_pub, inc_pub, truth, caps, off, w_day=0.0, w_off=0.0, shrink=0.0))
    for w in (0.5, 0.75, 1.25, 1.5):
        show(f"평일 보정 비율 {w}", loro(data, pred, days_pub, inc_pub, truth, caps, off, **{**base, "w_day": w}))
    for sh in (0.5, 1.0):
        show(f"잔차 축소 shrink={sh}", loro(data, pred, days_pub, inc_pub, truth, caps, off, **{**base, "shrink": sh}))
    # N 민감도: 기존 보정으로 N 만 강제
    n0 = loro(data, pred, days_pub, inc_pub, truth, caps, off, **base)[0][0]
    for dn in (-80, -40, +40, +80):
        show(f"N 강제 {dn:+d}", loro(data, pred, days_pub, inc_pub, truth, caps, off, **{**base, "fixed_n": n0 + dn}))


def export(args):
    torch.set_num_threads(2)
    data = forecast.read_data(args.data_dir, "B")
    days_pub, truth, prov = read_public(args.public_dir, data)
    caps = data.capacity
    models, config = load_models(args.output_dir)
    alldays, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    pred = inventory.ensemble(models, data, alldays, sites)
    inc_pub = increments(truth, caps)
    w_day = config["weekday_weight"] if args.weekday_weight is None else args.weekday_weight
    w_off = config["offday_weight"] if args.offday_weight is None else args.offday_weight
    cal = calibrate(pred, days_pub, inc_pub, data.city.off_day, w_day, w_off, args.shrink)
    cal[days_pub] = inc_pub
    alloc = allocate(cal, caps, args.fixed_n, data.city.off_day, args.offday_cap, args.lam)
    name = args.name or "개선_배치표_B지역.csv"
    pd.DataFrame(alloc.T, index=data.city.stations, columns=[f"D{d + 1:02d}" for d in alldays]) \
        .rename_axis("station_id").to_csv(args.output_dir / name, encoding="utf-8-sig")
    pub_score = realized_score(truth, alloc[days_pub], caps)
    print(json.dumps({"N": int(alloc[0].sum()), "lam": args.lam, "offday_cap": args.offday_cap,
                      "weekday_weight": w_day, "offday_weight": w_off, "shrink": args.shrink,
                      "public_days": [data.city.dates[d].strftime("%Y-%m-%d") for d in days_pub],
                      "public_actual_score_in_sample": pub_score, "file": str(args.output_dir / name)}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["evaluate", "export"])
    p.add_argument("--data-dir", type=Path, default=ROOT.parents[1])
    p.add_argument("--public-dir", type=Path, default=ROOT.parents[1] / "open_data")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--weekday-weight", type=float, default=None)
    p.add_argument("--offday-weight", type=float, default=None)
    p.add_argument("--shrink", type=float, default=0.0)
    p.add_argument("--fixed-n", type=int, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--offday-cap", action="store_true", help="주말·공휴일은 대여소당 floor(c/2) 이하로 분산")
    p.add_argument("--lam", type=float, default=20.0, help="N 선택에 쓸 자전거 1대 비용 (공식 20)")
    a = p.parse_args()
    evaluate(a) if a.action == "evaluate" else export(a)
