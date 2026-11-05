"""공유자전거 배치 채점기 — 서버와 같은 채점 코드(설명문만 뺐다).

평가함수(최종 문항 10/29): S = 20N + (6Q + 5O + E) / 30, 낮을수록 좋다.
매일 00시 배치 x 로 시작해 매 정시 z = b + a - r (b 직전 재고, a 반납, r 대여)
  u = max(-z, 0) 부족량 · o = max(z - c, 0) 수용량 초과량 · b' = min(c, max(0, z)) 다음 재고
  e = max(b' - p, 0) 적정 잔여량 초과량, p = 거치대 수 c 의 절반(소수점 버림)
N 총 자전거 수(1대 이상) · Q = u 제곱의 합 · O = o 의 합 · E = e 의 합 (30일 · 모든 대여소 · 24시간).
부족량 합 U = u 의 합은 점수에 안 들어가는 참고값이다.
라이브러리로 불러 쓸 때는 formula=GSMMC_FORMULA 를 넘겨라 — 코어 기본값(PENALTY_DEFAULTS)은 옛 계수다.
  score_day(..., formula=GSMMC_FORMULA) · score_submission 은 config 의 "score_formula" 에 GSMMC_FORMULA 를 넣는다.
  GSMMC_FORMULA(λ 20, w_u 0, γ 6, w_o 5, w_e 1, p_ratio 0.5)는 파일 끝 _cli 바로 위에 있고, CLI 도 이것을 쓴다.
사용: python score.py --hourly A_01_시간별수요.csv --stations A_02_대여소.csv --alloc A_배치표.csv
"""
from __future__ import annotations
import csv
import io
import math
from typing import Any
HOURS_DEFAULT = [7, 8, 9, 10]
DEFAULT_R = 0.25
DEFAULT_MAE_FLOOR = 0.9
DEFAULT_EXACT_RATE_MAX = 0.6
DEFAULT_S_CEILING = 80.0
DEFAULT_BAND_FLOOR = 5
PENALTY_DEFAULTS = {'lambda': 20.0, 'w_u': 10.0, 'gamma': 5.0, 'w_o': 5.0, 'w_e': 1.0, 'p_ratio': 0.5}

def station_need(rent: list[float], ret: list[float]) -> int:
    cum = 0.0
    peak = 0.0
    for d, r in zip(rent, ret):
        cum += float(d) - float(r)
        if cum > peak:
            peak = cum
    return int(round(peak))

def simulate_station(x: int, rent: list[float], ret: list[float]) -> dict[str, Any]:
    inv = float(x)
    traj = [inv]
    unmet_by_hour: list[int] = []
    served = 0.0
    idle = 0.0
    for d, r in zip(rent, ret):
        avail = inv + float(r)
        s = min(float(d), avail)
        unmet_by_hour.append(int(round(float(d) - s)))
        served += s
        inv = avail - s
        idle += inv
        traj.append(inv)
    return {'served': int(round(served)), 'unmet': int(sum(unmet_by_hour)), 'unmet_by_hour': unmet_by_hour, 'traj': [int(round(v)) for v in traj], 'idle_bike_hours': int(round(idle))}

def apply_rebalance(inv: dict[str, float], target: dict[str, float], max_move: int) -> tuple[dict[str, float], int]:
    inv = dict(inv)
    total = sum(inv.values())
    tsum = sum((float(v) for v in target.values()))
    if max_move <= 0 or tsum <= 0 or total <= 0:
        return (inv, 0)
    scale = total / tsum
    tgt = {s: float(target.get(s, 0.0)) * scale for s in inv}
    moved = 0
    while moved < max_move:
        surplus = {s: inv[s] - tgt[s] for s in inv}
        src = max(surplus, key=lambda s: surplus[s])
        dst = min(surplus, key=lambda s: surplus[s])
        m = int(math.floor(min(surplus[src], -surplus[dst], max_move - moved)))
        if m < 1:
            break
        inv[src] -= m
        inv[dst] += m
        moved += m
    return (inv, moved)

def simulate_day(alloc: dict[str, int], day: dict[str, Any], hours: list[int] | None=None, rebalance: dict[str, Any] | None=None) -> dict[str, Any]:
    rent: dict[str, list[float]] = day['rent']
    ret: dict[str, list[float]] = day['ret']
    sids = list(rent.keys())
    n_t = len(next(iter(rent.values()))) if sids else 0
    hours = list(hours) if hours else list(range(7, 7 + n_t))
    inv = {s: float(int(alloc.get(s, 0) or 0)) for s in sids}
    traj = {s: [int(inv[s])] for s in sids}
    unmet_by_hour = {s: [] for s in sids}
    served = {s: 0.0 for s in sids}
    idle = {s: 0.0 for s in sids}
    moved = 0
    reb_hour = rebalance.get('hour') if rebalance else None
    reb_target = rebalance.get('target') if rebalance else None
    for t in range(n_t):
        if reb_target and reb_hour is not None and (t < len(hours)) and (hours[t] == reb_hour):
            inv, moved = apply_rebalance(inv, reb_target, int(rebalance.get('max_move', 0) or 0))
        for s in sids:
            r_t = float(ret.get(s, [0.0] * n_t)[t]) if t < len(ret.get(s, [])) else 0.0
            d_t = float(rent[s][t])
            avail = inv[s] + r_t
            sv = min(d_t, avail)
            unmet_by_hour[s].append(int(round(d_t - sv)))
            served[s] += sv
            inv[s] = avail - sv
            idle[s] += inv[s]
            traj[s].append(int(round(inv[s])))
    per: dict[str, Any] = {}
    for s in sids:
        per[s] = {'served': int(round(served[s])), 'unmet': int(sum(unmet_by_hour[s])), 'unmet_by_hour': unmet_by_hour[s], 'traj': traj[s], 'idle_bike_hours': int(round(idle[s]))}
    return {'stations': per, 'moved': moved, 'hours': hours}

def penalty_params(formula: dict[str, Any] | None) -> dict[str, float]:
    out = dict(PENALTY_DEFAULTS)
    for k in PENALTY_DEFAULTS:
        if formula and formula.get(k) is not None:
            out[k] = float(formula[k])
    return out

def simulate_station_penalty(x: int, rent: list[float], ret: list[float], cap: float, p: float) -> dict[str, Any]:
    b = float(x)
    traj = [int(round(b))]
    u_h: list[int] = []
    o_h: list[int] = []
    e_sum = 0.0
    idle = 0.0
    for d, r in zip(rent, ret):
        z = b + float(r) - float(d)
        u_h.append(int(round(max(-z, 0.0))))
        o_h.append(int(round(max(z - cap, 0.0))))
        b = min(cap, max(0.0, z))
        e_sum += max(b - p, 0.0)
        idle += b
        traj.append(int(round(b)))
    return {'U': int(sum(u_h)), 'Q': int(sum((u * u for u in u_h))), 'O': int(sum(o_h)), 'E': int(round(e_sum)), 'unmet_by_hour': u_h, 'over_by_hour': o_h, 'traj': traj, 'idle_bike_hours': int(round(idle))}

def _score_day_penalty(alloc: dict[str, int], day: dict[str, Any], formula: dict[str, Any] | None, docks: dict[str, float] | None, *, detail: bool, hours: list[int] | None) -> dict[str, Any]:
    if not docks:
        raise ValueError('penalty_gamma 점수식에는 대여소별 거치대 수(truth stations[].docks)가 필요합니다.')
    pp = penalty_params(formula)
    rent: dict[str, list[float]] = day['rent']
    ret: dict[str, list[float]] = day['ret']
    n_t = len(next(iter(rent.values()))) if rent else 0
    hours = list(hours) if hours else list(range(n_t))
    tot = {'N': 0, 'U': 0, 'Q': 0, 'O': 0, 'E': 0}
    demand = 0
    unmet_station_hours = 0
    idle = 0
    max_unmet = 0
    max_unmet_station = None
    per_station: dict[str, Any] = {}
    for sid, rent_v in rent.items():
        if sid not in docks:
            raise ValueError(f'penalty_gamma: 대여소 {sid} 의 거치대 수가 없습니다.')
        cap = float(docks[sid])
        p = float(math.floor(cap * pp['p_ratio'] + 1e-09))
        ret_v = ret.get(sid, [0.0] * len(rent_v))
        x = int(alloc.get(sid, 0) or 0)
        s = simulate_station_penalty(x, rent_v, ret_v, cap, p)
        dem = int(round(sum((float(v) for v in rent_v))))
        tot['N'] += x
        for k in ('U', 'Q', 'O', 'E'):
            tot[k] += s[k]
        demand += dem
        unmet_station_hours += sum((1 for u in s['unmet_by_hour'] if u > 0))
        idle += s['idle_bike_hours']
        if s['U'] > max_unmet:
            max_unmet = s['U']
            max_unmet_station = sid
        if detail:
            per_station[sid] = {'alloc': x, 'docks': int(cap) if cap.is_integer() else cap, 'p': int(p), 'U': s['U'], 'Q': s['Q'], 'O': s['O'], 'E': s['E'], 'unmet': s['U'], 'unmet_by_hour': s['unmet_by_hour'], 'over_by_hour': s['over_by_hour'], 'traj': s['traj'], 'served': dem - s['U'], 'demand': dem, 'need': station_need(rent_v, ret_v)}
    parts = {'N': round(pp['lambda'] * tot['N'], 3), 'U': round(pp['w_u'] * tot['U'], 3), 'Q': round(pp['gamma'] * tot['Q'], 3), 'O': round(pp['w_o'] * tot['O'], 3), 'E': round(pp['w_e'] * tot['E'], 3)}
    s_val = sum(parts.values())
    out: dict[str, Any] = {'bikes': tot['N'], 'unmet': tot['U'], 'served': demand - tot['U'], 'demand': demand, 'need_sum': None, 'cost': round(s_val, 3), 'zero_cost': None, 'oracle_cost': None, 'S': round(s_val, 2), 'unmet_rate': round(100.0 * tot['U'] / demand, 3) if demand else 0.0, 'unmet_station_hours': unmet_station_hours, 'idle_bike_hours': idle, 'max_unmet': max_unmet, 'max_unmet_station': max_unmet_station, 'moved': 0, 'N': tot['N'], 'U': tot['U'], 'Q': tot['Q'], 'O': tot['O'], 'E': tot['E'], 'parts': parts}
    if detail:
        out['per_station'] = per_station
        out['hours'] = hours
    return out

def score_s(cost: float, zero_cost: float, oracle_cost: float) -> float:
    denom = zero_cost - oracle_cost
    if denom <= 1e-09:
        return 100.0 if cost <= oracle_cost + 1e-09 else 0.0
    return 100.0 * (zero_cost - cost) / denom

def score_from_unmet_rate(unmet: float, demand: float, formula: dict[str, Any]) -> float:
    u_full = float(formula.get('u_full', 1.5))
    u_zero = float(formula.get('u_zero', 3.5))
    u = 100.0 * float(unmet) / float(demand) if demand else 0.0
    if u_zero - u_full <= 1e-09:
        return 100.0 if u <= u_full else 0.0
    return 100.0 * max(0.0, min(1.0, (u_zero - u) / (u_zero - u_full)))

def score_from_reference(unmet: float, reference: dict[str, Any] | None, formula: dict[str, Any]) -> float:
    base_key = str(formula.get('baseline') or 'cap_prop')
    orc_key = str(formula.get('oracle') or 'oracle')
    if not isinstance(reference, dict):
        raise ValueError('reference_day_norm 점수식에는 truth.reference[day] 가 필요합니다.')
    try:
        u_b = float(reference[base_key]['unmet'])
        u_o = float(reference[orc_key]['unmet'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"reference_day_norm 점수식에는 truth.reference[day][{base_key!r}]['unmet'] 와 [{orc_key!r}]['unmet'] 이 필요합니다.") from exc
    denom = u_b - u_o
    if denom <= 1e-09:
        return 100.0 if float(unmet) <= u_o + 1e-09 else 0.0
    return 100.0 * max(0.0, min(1.0, (u_b - float(unmet)) / denom))

def band_points(s: float | None, bands: list[dict[str, Any]] | None, floor: float=DEFAULT_BAND_FLOOR) -> float | None:
    if s is None or not bands:
        return None
    if any(('max_s' in b for b in bands)):
        for b in sorted(bands, key=lambda b: float(b.get('max_s', math.inf))):
            if float(s) <= float(b.get('max_s', math.inf)) + 1e-09:
                return b.get('points')
        return floor
    ordered = sorted(bands, key=lambda b: float(b.get('min_s', 0)), reverse=True)
    for b in ordered:
        if float(s) >= float(b.get('min_s', 0)) - 1e-09:
            return b.get('points')
    return floor

def _formula_kind(formula: dict[str, Any] | None) -> str:
    return (formula or {}).get('kind') or 'cost_s'

def score_day(alloc: dict[str, int], day: dict[str, Any], r: float, *, detail: bool=False, hours: list[int] | None=None, rebalance: dict[str, Any] | None=None, formula: dict[str, Any] | None=None, reference: dict[str, Any] | None=None, docks: dict[str, float] | None=None) -> dict[str, Any]:
    if _formula_kind(formula) == 'penalty_gamma':
        if rebalance and rebalance.get('target'):
            raise ValueError('penalty_gamma 점수식은 낮 재배치를 지원하지 않습니다.')
        return _score_day_penalty(alloc, day, formula, docks, detail=detail, hours=hours)
    rent: dict[str, list[float]] = day['rent']
    ret: dict[str, list[float]] = day['ret']
    sim = simulate_day(alloc, day, hours, rebalance)
    per = sim['stations']
    bikes = 0
    unmet = 0
    served = 0
    demand = 0
    need_sum = 0
    unmet_station_hours = 0
    idle = 0
    max_unmet = 0
    max_unmet_station = None
    per_station: dict[str, Any] = {}
    for sid, rent_v in rent.items():
        ret_v = ret.get(sid, [0.0] * len(rent_v))
        x = int(alloc.get(sid, 0) or 0)
        need = station_need(rent_v, ret_v)
        s = per[sid]
        dem = int(round(sum((float(v) for v in rent_v))))
        bikes += x
        unmet += s['unmet']
        served += s['served']
        demand += dem
        need_sum += need
        unmet_station_hours += sum((1 for u in s['unmet_by_hour'] if u > 0))
        idle += s['idle_bike_hours']
        if s['unmet'] > max_unmet:
            max_unmet = s['unmet']
            max_unmet_station = sid
        if detail:
            per_station[sid] = {'alloc': x, 'need': need, 'unmet': s['unmet'], 'unmet_by_hour': s['unmet_by_hour'], 'traj': s['traj'], 'served': s['served'], 'demand': dem}
    cost = unmet + r * bikes
    zero_cost = float(need_sum)
    oracle_cost = r * need_sum
    unmet_rate = round(100.0 * unmet / demand, 3) if demand else 0.0
    kind = _formula_kind(formula)
    if kind == 'unmet_rate_clip':
        s_val = score_from_unmet_rate(unmet, demand, formula or {})
    elif kind == 'reference_day_norm':
        s_val = score_from_reference(unmet, reference, formula or {})
    else:
        s_val = score_s(cost, zero_cost, oracle_cost)
    out: dict[str, Any] = {'bikes': bikes, 'unmet': unmet, 'served': served, 'demand': demand, 'need_sum': need_sum, 'cost': round(cost, 3), 'zero_cost': round(zero_cost, 3), 'oracle_cost': round(oracle_cost, 3), 'S': round(s_val, 2), 'unmet_rate': unmet_rate, 'unmet_station_hours': unmet_station_hours, 'idle_bike_hours': idle, 'max_unmet': max_unmet, 'max_unmet_station': max_unmet_station, 'moved': sim['moved']}
    if detail:
        out['per_station'] = per_station
        out['hours'] = sim['hours']
    return out

def score_pred(pred: dict[str, list[float]] | None, day: dict[str, Any]) -> dict[str, Any] | None:
    if not pred:
        return None
    rent: dict[str, list[float]] = day['rent']
    unc: dict[str, list[int]] = day.get('uncensored', {})
    n = 0
    abs_sum = 0.0
    exact = 0
    abs_zero = 0.0
    for sid, actual in rent.items():
        p = pred.get(sid)
        if p is None:
            continue
        flags = unc.get(sid) or [1] * len(actual)
        for t, (a, f) in enumerate(zip(actual, flags)):
            if not f or t >= len(p):
                continue
            try:
                pv = float(p[t])
            except (TypeError, ValueError):
                continue
            if math.isnan(pv):
                continue
            n += 1
            abs_sum += abs(float(a) - pv)
            abs_zero += abs(float(a))
            if abs(float(a) - pv) < 1e-09:
                exact += 1
    if n == 0:
        return None
    return {'mae': round(abs_sum / n, 4), 'mae_zero': round(abs_zero / n, 4), 'exact_rate': round(exact / n, 4), 'n': n}

def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None

def _resolve_days(truth: dict[str, Any], config: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    defaults = truth.get('config_defaults', {})
    days = truth['days']
    eval_days = [d for d in config.get('eval_days') or defaults.get('eval_days') or list(days.keys()) if d in days]
    public_days = [d for d in config.get('public_days') or defaults.get('public_days') or eval_days[:1] if d in eval_days]
    scored_days = [d for d in config.get('scored_days') or defaults.get('scored_days') or eval_days if d in eval_days]
    return (eval_days, public_days, scored_days)

def score_submission(alloc_by_day: dict[str, dict[str, int]], pred_by_day: dict[str, dict[str, list[float]]] | None, truth: dict[str, Any], config: dict[str, Any] | None=None, rebalance_by_day: dict[str, dict[str, int]] | None=None) -> dict[str, Any]:
    config = config or {}
    defaults = truth.get('config_defaults', {})
    r = float(config.get('r', defaults.get('r', DEFAULT_R)))
    formula = config.get('score_formula', defaults.get('score_formula'))
    hours = list(config.get('hours') or truth.get('hours') or HOURS_DEFAULT)
    reb_cfg = config.get('rebalance', defaults.get('rebalance'))
    bands = config.get('score_bands', defaults.get('score_bands')) or None
    band_floor = config.get('band_points_floor', defaults.get('band_points_floor', DEFAULT_BAND_FLOOR))
    reference_all: dict[str, Any] = truth.get('reference') or {}
    days: dict[str, Any] = truth['days']
    eval_days, public_days, scored_days = _resolve_days(truth, config)
    allow_partial = bool(config.get('allow_partial_days', defaults.get('allow_partial_days', False)))
    submitted = [d for d in eval_days if d in alloc_by_day] if allow_partial else list(eval_days)
    private_days = [d for d in scored_days if d not in public_days]
    is_penalty = _formula_kind(formula) == 'penalty_gamma'
    docks = stations_docks(truth) if is_penalty else None
    per_day: dict[str, Any] = {}
    for d in submitted:
        day = days[d]
        ref_d = reference_all.get(d)
        res = score_day(alloc_by_day.get(d, {}), day, r, hours=hours, formula=formula, reference=ref_d, docks=docks)
        res['pred'] = score_pred((pred_by_day or {}).get(d), day)
        res['rebal'] = None
        tgt = (rebalance_by_day or {}).get(d)
        if reb_cfg and tgt:
            reb = score_day(alloc_by_day.get(d, {}), day, r, hours=hours, formula=formula, rebalance={'hour': reb_cfg.get('hour', 13), 'max_move': reb_cfg.get('max_move', 100), 'target': tgt}, reference=ref_d)
            res['rebal'] = {'unmet': reb['unmet'], 'unmet_rate': reb['unmet_rate'], 'S': reb['S'], 'moved': reb['moved'], 'cost': reb['cost']}
        per_day[d] = res

    def agg(keys: list[str]) -> dict[str, Any]:
        keys = [k for k in keys if k in per_day]
        rows = [per_day[k] for k in keys]
        unmet_sum = sum((x['unmet'] for x in rows)) if rows else None
        demand_sum = sum((x['demand'] for x in rows)) if rows else None
        if rows and _formula_kind(formula) == 'unmet_rate_clip':
            s_avg = round(score_from_unmet_rate(unmet_sum, demand_sum, formula), 2)
        else:
            s_avg = _mean([x['S'] for x in rows])
        reb_rows = [x['rebal'] for x in rows if x.get('rebal')]
        out = {'days': keys, 'S_avg': s_avg, 'cost_sum': round(sum((x['cost'] for x in rows)), 3) if rows else None, 'unmet_sum': unmet_sum, 'demand_sum': demand_sum, 'unmet_rate': (round(100.0 * unmet_sum / demand_sum, 3) if demand_sum else 0.0) if rows else None, 'bikes_avg': _mean([float(x['bikes']) for x in rows]), 'mae_avg': _mean([x['pred']['mae'] for x in rows if x['pred']]), 'exact_rate_avg': _mean([x['pred']['exact_rate'] for x in rows if x['pred']]), 'rebal_S': None, 'rebal_unmet_sum': None, 'rebal_unmet_rate': None, 'moved_sum': None}
        if reb_rows and len(reb_rows) == len(rows):
            ru = sum((x['unmet'] for x in reb_rows))
            out['rebal_unmet_sum'] = ru
            out['rebal_unmet_rate'] = round(100.0 * ru / demand_sum, 3) if demand_sum else 0.0
            out['moved_sum'] = sum((x['moved'] for x in reb_rows))
            if _formula_kind(formula) == 'unmet_rate_clip':
                out['rebal_S'] = round(score_from_unmet_rate(ru, demand_sum, formula), 2)
            else:
                out['rebal_S'] = _mean([x['S'] for x in reb_rows])
        if is_penalty:
            out['N_avg'] = out['bikes_avg']
            for k in ('U', 'Q', 'O', 'E'):
                out[k + '_sum'] = sum((x[k] for x in rows)) if rows else None
            out['parts_avg'] = {k: round(sum((x['parts'][k] for x in rows)) / len(rows), 3) for k in ('N', 'U', 'Q', 'O', 'E')} if rows else None
        if bands:
            out['band_points'] = band_points(s_avg, bands, band_floor)
        return out
    result = {'r': r, 'formula': _formula_kind(formula), 'hours': hours, 'days': per_day, 'public_days': public_days, 'private_days': private_days, 'scored_days': scored_days, 'submitted_days': submitted, 'public': agg(public_days), 'private': agg(private_days), 'all': agg(scored_days)}
    if is_penalty:
        result['lower_is_better'] = True
        result['penalty'] = penalty_params(formula)
    result['flags'] = leak_flags(result, config)
    return result

def leak_flags(result: dict[str, Any], config: dict[str, Any] | None=None) -> list[str]:
    config = config or {}
    floor = float(config.get('mae_floor', DEFAULT_MAE_FLOOR))
    exact_max = float(config.get('exact_rate_max', DEFAULT_EXACT_RATE_MAX))
    s_ceiling = float(config.get('s_ceiling', DEFAULT_S_CEILING))
    flags: list[str] = []
    maes = [v['pred']['mae'] for v in result['days'].values() if v.get('pred')]
    if maes and min(maes) < floor:
        flags.append('mae_below_floor')
    exacts = [v['pred']['exact_rate'] for v in result['days'].values() if v.get('pred')]
    if exacts and max(exacts) > exact_max:
        flags.append('exact_rate_high')
    s_all = result['all'].get('S_avg')
    if result.get('lower_is_better'):
        s_floor = config.get('s_floor')
        if s_floor is not None and s_all is not None and (s_all < float(s_floor)) and (len(result['all']['days']) >= 2):
            flags.append('s_below_floor')
        return flags
    if s_all is not None and s_all > s_ceiling and (len(result['all']['days']) >= 2):
        flags.append('s_above_ceiling')
    return flags

def replay_day(alloc: dict[str, int], day: dict[str, Any], r: float, *, hours: list[int] | None=None, rebalance: dict[str, Any] | None=None, formula: dict[str, Any] | None=None, reference: dict[str, Any] | None=None, docks: dict[str, float] | None=None) -> dict[str, Any]:
    res = score_day(alloc, day, r, detail=True, hours=hours, rebalance=rebalance, formula=formula, reference=reference, docks=docks)
    per = res.pop('per_station')
    losses = sorted(per.items(), key=lambda kv: (-kv[1]['unmet'], kv[0]))
    res['stations'] = per
    res['top_losses'] = [{'id': sid, 'unmet': v['unmet'], 'alloc': v['alloc'], 'need': v['need']} for sid, v in losses[:10] if v['unmet'] > 0]
    res['oracle_alloc'] = {sid: v['need'] for sid, v in per.items()}
    return res

def stations_docks(truth: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for st in truth.get('stations') or []:
        try:
            dk = float(st.get('docks'))
        except (TypeError, ValueError):
            continue
        if dk > 0:
            out[st['id']] = dk
    return out

def day_hourly_csv(truth: dict[str, Any], day_code: str) -> str:
    day = truth['days'][day_code]
    hours = list(truth.get('hours') or range(len(next(iter(day['rent'].values())))))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['날짜', '요일', '시', 'station_id', 'rentals', 'returns'])
    for k, h in enumerate(hours):
        for st in truth['stations']:
            sid = st['id']
            w.writerow([day.get('date', day_code), day.get('label', ''), h, sid, int(round(float(day['rent'][sid][k]))), int(round(float(day['ret'][sid][k])))])
    return buf.getvalue()

def day_totals_csv(truth: dict[str, Any], day_code: str) -> str:
    day = truth['days'][day_code]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['날짜', '요일', 'station_id', 'rentals', 'returns'])
    for st in truth['stations']:
        sid = st['id']
        w.writerow([day.get('date', day_code), day.get('label', ''), sid, int(round(sum((float(v) for v in day['rent'][sid])))), int(round(sum((float(v) for v in day['ret'][sid]))))])
    return buf.getvalue()

def validate_alloc(alloc_by_day: dict[str, dict[str, int]], truth: dict[str, Any], config: dict[str, Any] | None=None) -> list[str]:
    config = config or {}
    defaults = truth.get('config_defaults', {})
    station_ids = [s['id'] for s in truth['stations']]
    eval_days, _, _ = _resolve_days(truth, config)
    min_total = config.get('min_total', defaults.get('min_total'))
    max_total = config.get('max_total', defaults.get('max_total'))
    fixed_total = config.get('fixed_total', defaults.get('fixed_total'))
    allow_partial = bool(config.get('allow_partial_days', defaults.get('allow_partial_days', False)))
    equal_daily = bool(config.get('equal_daily_total', defaults.get('equal_daily_total', False)))
    enforce_docks = bool(config.get('enforce_docks', defaults.get('enforce_docks', False)))
    docks: dict[str, int | float] = {}
    if enforce_docks:
        for s in truth['stations']:
            try:
                dk = float(s.get('docks'))
            except (TypeError, ValueError):
                continue
            if dk > 0:
                docks[s['id']] = int(dk) if dk.is_integer() else dk
    day_totals: list[tuple[str, int]] = []
    too_few: list[tuple[str, int]] = []
    errors: list[str] = []
    check_days = [d for d in eval_days if d in alloc_by_day] if allow_partial else eval_days
    if allow_partial and (not check_days):
        errors.append(f"평가일({', '.join(eval_days)}) 중 적어도 하루의 배치가 있어야 합니다.")
        return errors
    unknown_days = [d for d in alloc_by_day if d not in eval_days]
    if unknown_days:
        errors.append(f"평가일이 아닌 날짜 {', '.join(sorted(unknown_days)[:5])} 가 있습니다. 평가일은 {', '.join(eval_days)} 입니다.")
    for d in check_days:
        col = alloc_by_day.get(d)
        if col is None:
            errors.append(f'{d} 열이 없습니다. 평가일 {len(eval_days)}일 모두 배치 수량이 필요합니다.')
            continue
        missing = [s for s in station_ids if s not in col]
        extra = [s for s in col if s not in station_ids]
        if missing:
            errors.append(f"{d}: 대여소 {len(missing)}곳의 배치가 비어 있습니다 (예: {', '.join(missing[:5])}).")
        if extra:
            errors.append(f"{d}: 모르는 대여소 {', '.join(sorted(extra)[:5])} 가 있습니다.")
        bad = [s for s, v in col.items() if not isinstance(v, int) or isinstance(v, bool) or v < 0]
        if bad:
            errors.append(f"{d}: 배치 수량은 0 이상의 정수여야 합니다 (예: {', '.join(sorted(bad)[:5])}).")
            continue
        total = sum(col.values())
        if fixed_total is not None and total != int(fixed_total):
            errors.append(f'{d}: 합계가 {total}대입니다. 정확히 {fixed_total}대여야 합니다.')
        elif max_total is not None and total > int(max_total):
            errors.append(f'{d}: 합계가 {total}대로 보유 상한 {max_total}대를 넘습니다.')
        elif min_total is not None and total < int(min_total):
            too_few.append((d, total))
        day_totals.append((d, total))
        if enforce_docks:
            over = [s for s in station_ids if s in docks and s in col and (col[s] > docks[s])]
            if over:
                ex = ', '.join((f'{s} {col[s]}>{docks[s]}' for s in over[:5]))
                errors.append(f'{d}: 거치대 수를 넘는 대여소 {len(over)}곳 (예: {ex}).')
    kinds = {t for _, t in day_totals}
    if equal_daily and len(kinds) > 1:
        listed = ', '.join((f'{d} {t}' for d, t in day_totals[:8])) + (', …' if len(day_totals) > 8 else '')
        errors.append(f'매일 합계가 같아야 합니다(= 총 자전거 수). 날짜별 합계: {listed} (서로 다른 합계 {len(kinds)}가지).')
    if too_few:
        listed = ', '.join((f'{d} {t}' for d, t in too_few[:8])) + (', …' if len(too_few) > 8 else '')
        errors.append(f'총 자전거 수(하루 배치 합계)는 {int(min_total)}대 이상이어야 합니다. 합계가 모자란 날 {len(too_few)}일: {listed}')
    return errors

def validate_rebalance(rebalance_by_day: dict[str, dict[str, int]], truth: dict[str, Any], config: dict[str, Any] | None=None) -> list[str]:
    config = config or {}
    station_ids = [s['id'] for s in truth['stations']]
    eval_days, _, _ = _resolve_days(truth, config)
    errors: list[str] = []
    for d, col in rebalance_by_day.items():
        if d not in eval_days:
            errors.append(f'재배치표: 평가일이 아닌 날짜 {d} 가 있습니다.')
            continue
        missing = [s for s in station_ids if s not in col]
        if missing:
            errors.append(f"재배치표 {d}: 대여소 {len(missing)}곳이 비어 있습니다 (예: {', '.join(missing[:5])}).")
        bad = [s for s, v in col.items() if not isinstance(v, int) or isinstance(v, bool) or v < 0]
        if bad:
            errors.append(f"재배치표 {d}: 목표 대수는 0 이상의 정수여야 합니다 (예: {', '.join(sorted(bad)[:5])}).")
        if sum((int(v) for v in col.values() if isinstance(v, int))) <= 0:
            errors.append(f'재배치표 {d}: 목표 대수가 전부 0 입니다.')
    return errors

def _norm_header(h: str) -> str:
    return (h or '').strip().lstrip('\ufeff').lower()
_STATION_HEADERS = {'대여소', '대여소번호', '대여소 id', '대여소id', 'station', 'station_id', 'id'}
_DAY_HEADERS = {'날짜', 'day', 'date', '평가일', '일자'}
_HOUR_HEADERS = {'시각', '시간', 'hour', 'h', '시'}
_PRED_HEADERS = {'예측대여', '예측', 'pred', 'prediction', 'rent_pred', '예상대여'}
_COUNT_HEADERS = {'배치대수', '배치', '대수', 'count', 'alloc', 'x', 'bikes'}
_TARGET_HEADERS = {'목표대수', '목표', 'target', '목표재고'}

def normalize_day_code(raw: str) -> str:
    s = (raw or '').strip().upper().replace(' ', '')
    for p in ('DAY', 'D'):
        if s.startswith(p):
            s = s[len(p):]
            break
    if s.isdigit():
        return f'D{int(s):02d}'
    return (raw or '').strip().upper()

def _parse_long_csv(rows: list[list[str]], header: list[str], days: list[str], count_headers: set[str], what: str) -> dict[str, dict[str, int]]:
    try:
        ci = next((i for i, h in enumerate(header) if h in _STATION_HEADERS))
        cd = next((i for i, h in enumerate(header) if h in _DAY_HEADERS))
        cc = next((i for i, h in enumerate(header) if h in count_headers))
    except StopIteration as exc:
        raise ValueError(f"{what} 머리글은 'Day,대여소 ID,{sorted(count_headers)[0]}' 꼴이어야 합니다.") from exc
    out: dict[str, dict[str, int]] = {}
    for ln, row in enumerate(rows[1:], start=2):
        if len(row) <= max(ci, cd, cc):
            raise ValueError(f'{ln}행의 열 수가 부족합니다.')
        sid = row[ci].strip()
        d = normalize_day_code(row[cd])
        if d not in days:
            raise ValueError(f"{ln}행의 날짜 '{row[cd]}' 는 평가일({', '.join(days)})이 아닙니다.")
        raw = row[cc].strip()
        if raw == '':
            raise ValueError(f'{ln}행 {sid} 의 {d} 값이 비어 있습니다.')
        try:
            fv = float(raw)
        except ValueError as exc:
            raise ValueError(f"{ln}행 {sid} 의 값 '{raw}' 은 숫자가 아닙니다.") from exc
        if not math.isfinite(fv):
            raise ValueError(f"{ln}행 {sid} 의 값 '{raw}' 은 숫자가 아닙니다.")
        if abs(fv - round(fv)) > 1e-09:
            raise ValueError(f'{ln}행 {sid} 의 값 {raw} 은 정수가 아닙니다.')
        col = out.setdefault(d, {})
        if sid in col:
            raise ValueError(f'{d} 의 {sid} 가 두 번 나옵니다({ln}행).')
        col[sid] = int(round(fv))
    return out

def parse_alloc_csv(text: str, days: list[str], *, require_all_days: bool=True) -> dict[str, dict[str, int]]:
    reader = csv.reader(io.StringIO(text.lstrip('\ufeff')))
    rows = [r for r in reader if any((c.strip() for c in r))]
    if not rows:
        raise ValueError('배치표가 비어 있습니다.')
    header = [_norm_header(h) for h in rows[0]]
    has_day_col = any((h in _DAY_HEADERS for h in header))
    has_count_col = any((h in _COUNT_HEADERS for h in header))
    if has_day_col and has_count_col:
        return _parse_long_csv(rows, header, days, _COUNT_HEADERS, '배치표')
    if header[0] not in _STATION_HEADERS:
        raise ValueError("첫 열 머리글은 '대여소' 또는 'station_id' 여야 합니다(또는 세로형 'Day,대여소 ID,배치대수').")
    day_cols: dict[str, int] = {}
    for i, h in enumerate(header[1:], start=1):
        hu = normalize_day_code(h)
        if hu in days:
            day_cols[hu] = i
    missing = [d for d in days if d not in day_cols]
    if not day_cols or (require_all_days and missing):
        raise ValueError(f"평가일 열 {', '.join(missing)} 이(가) 없습니다. 머리글은 '대여소,{','.join(days)}' 또는 세로형 'Day,대여소 ID,배치대수' 입니다.")
    out: dict[str, dict[str, int]] = {d: {} for d in day_cols}
    for ln, row in enumerate(rows[1:], start=2):
        sid = row[0].strip()
        if not sid:
            continue
        for d, i in day_cols.items():
            raw = row[i].strip() if i < len(row) else ''
            if raw == '':
                raise ValueError(f'{ln}행 {sid} 의 {d} 값이 비어 있습니다.')
            try:
                fv = float(raw)
            except ValueError as exc:
                raise ValueError(f"{ln}행 {sid} 의 {d} 값 '{raw}' 은 숫자가 아닙니다.") from exc
            if not math.isfinite(fv):
                raise ValueError(f"{ln}행 {sid} 의 {d} 값 '{raw}' 은 숫자가 아닙니다.")
            if abs(fv - round(fv)) > 1e-09:
                raise ValueError(f'{ln}행 {sid} 의 {d} 값 {raw} 은 정수가 아닙니다.')
            if sid in out[d]:
                raise ValueError(f'{sid} 가 두 번 나옵니다({ln}행).')
            out[d][sid] = int(round(fv))
    return out

def parse_rebalance_csv(text: str, days: list[str]) -> dict[str, dict[str, int]]:
    reader = csv.reader(io.StringIO(text.lstrip('\ufeff')))
    rows = [r for r in reader if any((c.strip() for c in r))]
    if not rows:
        raise ValueError('재배치표가 비어 있습니다.')
    header = [_norm_header(h) for h in rows[0]]
    if any((h in _DAY_HEADERS for h in header)) and any((h in _TARGET_HEADERS | _COUNT_HEADERS for h in header)):
        return _parse_long_csv(rows, header, days, _TARGET_HEADERS | _COUNT_HEADERS, '재배치표')
    return parse_alloc_csv(text, days)

def parse_pred_csv(text: str, days: list[str], hours: list[int]) -> dict[str, dict[str, list[float]]]:
    reader = csv.reader(io.StringIO(text.lstrip('\ufeff')))
    rows = [r for r in reader if any((c.strip() for c in r))]
    if not rows:
        raise ValueError('예측표가 비어 있습니다.')
    header = [_norm_header(h) for h in rows[0]]
    try:
        ci = next((i for i, h in enumerate(header) if h in _STATION_HEADERS))
        cd = next((i for i, h in enumerate(header) if h in _DAY_HEADERS))
        ch = next((i for i, h in enumerate(header) if h in _HOUR_HEADERS))
        cp = next((i for i, h in enumerate(header) if h in _PRED_HEADERS))
    except StopIteration as exc:
        raise ValueError("예측표 머리글은 '대여소,날짜,시각,예측대여' 여야 합니다.") from exc
    hidx = {h: k for k, h in enumerate(hours)}
    out: dict[str, dict[str, list[float]]] = {d: {} for d in days}
    for ln, row in enumerate(rows[1:], start=2):
        if len(row) <= max(ci, cd, ch, cp):
            raise ValueError(f'{ln}행의 열 수가 부족합니다.')
        sid = row[ci].strip()
        d = normalize_day_code(row[cd])
        if d not in out:
            raise ValueError(f"{ln}행의 날짜 '{row[cd]}' 는 평가일({', '.join(days)})이 아닙니다.")
        try:
            h = int(float(row[ch].strip()))
        except ValueError as exc:
            raise ValueError(f"{ln}행의 시각 '{row[ch]}' 은 정수가 아닙니다.") from exc
        if h not in hidx:
            raise ValueError(f"{ln}행의 시각 {h} 는 평가 시간대({', '.join(map(str, hours))})가 아닙니다.")
        try:
            pv = float(row[cp].strip())
        except ValueError as exc:
            raise ValueError(f"{ln}행의 예측값 '{row[cp]}' 은 숫자가 아닙니다.") from exc
        if pv < 0:
            raise ValueError(f'{ln}행의 예측값 {pv} 은 음수입니다.')
        arr = out[d].setdefault(sid, [float('nan')] * len(hours))
        arr[hidx[h]] = pv
    return out

def alloc_to_csv(alloc_by_day: dict[str, dict[str, int]], station_ids: list[str], days: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['대여소', *days])
    for sid in station_ids:
        w.writerow([sid, *[int(alloc_by_day.get(d, {}).get(sid, 0)) for d in days]])
    return buf.getvalue()

def alloc_to_long_csv(alloc_by_day: dict[str, dict[str, int]], station_ids: list[str], days: list[str], count_header: str='배치대수') -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['Day', '대여소 ID', count_header])
    for d in days:
        dn = d[1:] if d.upper().startswith('D') else d
        for sid in station_ids:
            w.writerow([int(dn) if str(dn).isdigit() else dn, sid, int(alloc_by_day.get(d, {}).get(sid, 0))])
    return buf.getvalue()


# 대회 계수(최종 문항 10/29): S = 20N + (6Q + 5O + E) / 30 — 부족량 합 U 의 계수 w_u 는 0(U 는 참고값).
# 함수로 불러 쓸 때도 이것을 넘긴다: score_day(..., formula=GSMMC_FORMULA), score_submission 은 config["score_formula"].
# 넘기지 않으면 코어 기본값 PENALTY_DEFAULTS(옛 계수)로 계산된다.
GSMMC_FORMULA = {"kind": "penalty_gamma", "lambda": 20, "w_u": 0, "gamma": 6, "w_o": 5, "w_e": 1, "p_ratio": 0.5}


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="학습 지역 시간별 기록으로 배치표 30일 채점")
    ap.add_argument("--hourly", required=True, help="01_시간별수요 시트를 CSV 로 저장한 파일(date,hour,station_id,rentals,returns)")
    ap.add_argument("--stations", required=True, help="02_대여소 시트를 CSV 로 저장한 파일(station_id,...,capacity)")
    ap.add_argument("--alloc", required=True, help="배치표 CSV (station_id,D01..D30 또는 Day,대여소 ID,배치대수)")
    ap.add_argument("--gamma", type=float, default=None, help="부족량 제곱합 Q 의 계수를 바꿔 볼 때만 (생략하면 GSMMC_FORMULA 의 대회 값 6)")
    a = ap.parse_args()

    def _read(path):
        raw = open(path, "rb").read()
        for enc in ("utf-8-sig", "cp949"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        raise SystemExit(path + ": UTF-8 로 저장해 주세요")

    def _sid(v):
        v = str(v).strip()
        try:
            f = float(v)
            return str(int(f)) if f.is_integer() else v
        except ValueError:
            return v

    rows = list(csv.DictReader(io.StringIO(_read(a.hourly))))
    key = "station_id" if rows and "station_id" in rows[0] else "station_no"
    dates = sorted({r["date"][:10] for r in rows})
    code = {d: "D%02d" % (i + 1) for i, d in enumerate(dates)}
    sids = sorted({_sid(r[key]) for r in rows})
    days = {c: {"rent": {s: [0] * 24 for s in sids}, "ret": {s: [0] * 24 for s in sids}} for c in code.values()}
    for r in rows:
        d = days[code[r["date"][:10]]]
        s = _sid(r[key])
        h = int(float(r["hour"]))
        d["rent"][s][h] += int(float(r["rentals"]))
        d["ret"][s][h] += int(float(r["returns"]))
    day_list = list(days)
    alloc = parse_alloc_csv(_read(a.alloc).replace("station_no", "station_id", 1), day_list)
    alloc = {d: {_sid(s): v for s, v in col.items()} for d, col in alloc.items()}
    docks = {}
    for r in csv.DictReader(io.StringIO(_read(a.stations))):
        docks[_sid(r.get("station_id") or r.get("station_no"))] = int(float(r["capacity"]))
    missing = [s for s in sids if s not in docks]
    if missing:
        raise SystemExit("대여소 표에 거치대 수가 없는 대여소: " + ", ".join(missing[:5]))
    truth = {"days": days, "stations": [{"id": s, "docks": docks[s]} for s in sids], "hours": list(range(24))}
    formula = dict(GSMMC_FORMULA)  # 대회 계수 — 위 상수
    if a.gamma is not None:
        formula["gamma"] = a.gamma
    cfg = {"hours": list(range(24)), "eval_days": day_list, "public_days": day_list, "scored_days": day_list,
           "equal_daily_total": True, "enforce_docks": True, "min_total": 1, "score_formula": formula}
    errs = validate_alloc(alloc, truth, cfg)
    if errs:
        print("형식 오류:")
        for e in errs[:20]:
            print("  -", e)
        return 1
    res = score_submission(alloc, None, truth, cfg)
    inv = {v: k for k, v in code.items()}
    print("날짜           N        Q      O      E          S_d  | 참고 U")
    for c in day_list:
        x = res["days"][c]
        print("%s %6d %8d %6d %6d %12.1f  | %6d" % (inv[c], x["N"], x["Q"], x["O"], x["E"], x["S"], x["U"]))
    al = res["all"]
    p = al["parts_avg"]
    print("30일 평균 벌점 S = %.2f  (20N %.1f + %gQ/30 %.1f + 5O/30 %.1f + E/30 %.1f) · 낮을수록 좋다"
          % (al["S_avg"], p["N"], formula["gamma"], p["Q"], p["O"], p["E"]))
    print("참고(점수에 안 들어감): 부족량 합 U = %d대(30일) · 못 빌린 비율 %.2f%%" % (al["U_sum"], al["unmet_rate"]))
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_cli())
