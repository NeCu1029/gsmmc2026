"""Minimize integer fleet within explicit predicted/public cost-increase budgets."""
import argparse
import json
from pathlib import Path
import bike_fleet_penalty as networks
import numpy as np
import pandas as pd
import torch
from neural_allocation import dp_all_totals, dp_reconstruct, realized_score, scenario_costs

ROOT = Path(__file__).resolve().parent
MODEL_NAME = 'bounded_fleet_model.pt'
SUBMISSION = '최종제출_배치표_B지역_벌점상한_N최소.csv'
TOLERANCES = (0., 0.005, 0.01, 0.02, 0.05)


def prepare(costs, capacities, public_truth):
    capacities = np.asarray(capacities, dtype=np.int64)
    if costs.ndim != 3 or costs.shape[1] != len(capacities) or (capacities <= 0).any() or costs.shape[-1] <= capacities.max():
        raise ValueError('Invalid cost/capacity dimensions')
    valid = np.arange(costs.shape[-1])[None, None] <= capacities[None, :, None]
    if not np.isfinite(costs[np.broadcast_to(valid, costs.shape)]).all():
        raise ValueError('Non-finite feasible costs')
    if public_truth.ndim != 3 or public_truth.shape[1:] != (len(capacities), 24) or not len(public_truth) or not np.isfinite(public_truth).all():
        raise ValueError('Invalid public truth')
    costs = np.where(valid, costs, np.inf)
    public_costs = scenario_costs(public_truth[:, :, None, :].astype(float), np.ones(public_truth.shape[:2] + (1,)), capacities)
    predicted_totals = np.stack([dp_all_totals(day, capacities) for day in costs])
    public_totals = np.stack([dp_all_totals(day, capacities) for day in public_costs])
    fleets = np.arange(predicted_totals.shape[-1])
    predicted = 20 * fleets + predicted_totals.mean(0)
    public = 20 * fleets + public_totals.mean(0)
    predicted[0] = public[0] = np.inf
    baseline_n = int(np.argmin(predicted))
    return {'costs': costs, 'capacities': capacities, 'predicted': predicted, 'public': public,
            'baseline_N': baseline_n, 'public_reference_score': float(public[baseline_n]),
            'predicted_baseline_relative': float(predicted[baseline_n])}


def select(prepared, tolerance=0.01):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    n0 = prepared['baseline_N']
    reference = prepared['public_reference_score']
    if reference <= 0 or not np.isfinite(reference):
        raise ValueError('Public reference score must be positive and finite')
    budget = tolerance * reference
    error_tolerance = 1e-9 * max(1., budget)
    predicted_delta = prepared['predicted'] - prepared['predicted_baseline_relative']
    public_delta = prepared['public'] - reference
    fleets = np.arange(len(predicted_delta))
    allowed = (fleets >= 1) & (fleets <= n0) & (predicted_delta <= budget + error_tolerance) & (public_delta <= budget + error_tolerance)
    candidates = np.flatnonzero(allowed)
    if not len(candidates):
        raise AssertionError('Baseline fleet unexpectedly infeasible')
    n = int(candidates[0])
    capacities = prepared['capacities']
    allocation = np.stack([dp_reconstruct(day, capacities, n) for day in prepared['costs']])
    if (allocation < 0).any() or (allocation > capacities).any() or not (allocation.sum(1) == n).all():
        raise AssertionError('Invalid constrained integer allocation')
    expected = prepared['predicted'][n] - 20 * n
    attained = prepared['costs'][np.arange(len(allocation))[:, None], np.arange(len(capacities)), allocation].sum(1).mean()
    if not np.isclose(attained, expected):
        raise AssertionError('Constrained DP reconstruction mismatch')
    report = {'tolerance': float(tolerance), 'baseline_N': n0, 'N': n, 'N_reduction_percent': float(100 * (n0 - n) / n0),
              'public_reference_score': reference, 'allowed_daily_score_increase': float(budget),
              'predicted_daily_score_increase': float(predicted_delta[n]), 'public_actual_score_at_N': float(prepared['public'][n]),
              'public_actual_score_increase': float(public_delta[n]), 'all_smaller_positive_N_infeasible': bool(not allowed[1:n].any()),
              'relative_objective': float(prepared['predicted'][n])}
    if n > 1:
        report['N_minus_one_predicted_increase'] = float(predicted_delta[n - 1])
        report['N_minus_one_public_increase'] = float(public_delta[n - 1])
    return report, allocation


def validate(data, signature_dir, output, tolerance):
    original = json.loads((signature_dir / 'A_ensemble_validation.json').read_text(encoding='utf-8'))
    public_days = np.array([8, 9])
    with np.load(signature_dir / 'A_ensemble_costs.npz', allow_pickle=False) as z:
        predictions = z['increments'].copy()
    future = np.arange(23, 30)
    tolerances = sorted(set(TOLERANCES + (float(tolerance),)))
    allocations = {r: np.empty(predictions.shape[:2], dtype=np.int64) for r in tolerances}
    records, totals = [], {r: [] for r in tolerances}
    for i, fold in enumerate(original['folds']):
        sites = data.city.stations.get_indexer(fold['test_station_ids'])
        if (sites < 0).any():
            raise ValueError('Unknown OOF station')
        costs = networks.neural.inventory.cost_table(predictions[:, sites], data.capacity[sites])
        prepared = prepare(costs, data.capacity[sites], data.city.net_flow[public_days][:, sites].numpy())
        for rate in tolerances:
            result, x = select(prepared, rate)
            result['actual_A'] = realized_score(data.city.net_flow[future][:, sites].numpy(), x, data.capacity[sites])
            allocations[rate][:, sites] = x
            totals[rate].append(result)
        records.append({'test_station_ids': fold['test_station_ids'], 'selected': totals[tolerance][-1]})
        print(f'Bounded policy fold {i+1}/5: {totals[tolerance][-1]}', flush=True)
    table = []
    for rate in tolerances:
        actual = realized_score(data.city.net_flow[future].numpy(), allocations[rate], data.capacity)
        table.append({'tolerance': rate, **actual, 'public_reference_score': sum(r['public_reference_score'] for r in totals[rate]),
                      'allowed_daily_score_increase': sum(r['allowed_daily_score_increase'] for r in totals[rate]),
                      'predicted_daily_score_increase': sum(r['predicted_daily_score_increase'] for r in totals[rate]),
                      'public_actual_score_increase': sum(r['public_actual_score_increase'] for r in totals[rate])})
    baseline = next(r for r in table if r['tolerance'] == 0)
    if abs(baseline['score'] - original['actual_score']['score']) > 1e-7:
        raise AssertionError('Baseline OOF result mismatch')
    actual = next(r for r in table if r['tolerance'] == tolerance)
    report = {'tolerance': float(tolerance), 'tolerance_selection': 'Fixed at 1% by task default; not optimized on held-out A scores.',
              'protocol': original['protocol'], 'budget_reference': 'Actual optimal public D09/D10 score at the original baseline fleet, per outer station group.',
              'guarantee': 'Predicted average cost increase and observed-public average cost increase each stay within tolerance*public_reference_score. Unobserved true cost is not bounded.',
              'baseline_actual': baseline, 'actual': actual, 'actual_A_score_increase_percent': 100 * (actual['score'] / baseline['score'] - 1),
              'folds': records, 'comparison': table}
    (output / 'A_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(table).to_csv(output / 'A_tolerance_comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(allocations[tolerance].T, index=data.city.stations, columns=[f'D{d+1:02d}' for d in future]).rename_axis('station_id').to_csv(output / 'A_검증배치.csv', encoding='utf-8-sig')
    print(json.dumps({'baseline': baseline, 'selected': actual}), flush=True)


def export(data, public_dir, output, tolerance_override=None):
    saved = torch.load(output / MODEL_NAME, map_location='cpu', weights_only=True)
    if saved.get('model_type') != 'bounded_neural_inventory_v1':
        raise ValueError('Incorrect bounded-fleet checkpoint')
    tolerance = saved['tolerance'] if tolerance_override is None else float(tolerance_override)
    public_days, truth, provenance = networks.neural.transfer.read_public(public_dir, data)
    costs = networks.predict_costs(saved['networks'], data, public_days, truth)
    prepared = prepare(costs, data.capacity, truth)
    report, allocation = select(prepared, tolerance)
    # Public rows use exact costs up to a constant, so their actual score must
    # equal the independently computed public frontier at this same fleet.
    public_actual = realized_score(truth, allocation[public_days], data.capacity)
    if abs(public_actual['score'] - report['public_actual_score_at_N']) > 1e-6:
        raise AssertionError('Public allocation does not attain its exact fixed-fleet cost')
    rows = [select(prepared, rate)[0] for rate in sorted(set(TOLERANCES + (float(tolerance),)))]
    if any(a['N'] < b['N'] for a, b in zip(rows, rows[1:])):
        raise AssertionError('Relaxing the bound increased N')
    pd.DataFrame(allocation.T, index=data.city.stations, columns=[f'D{d+1:02d}' for d in range(len(allocation))]).rename_axis('station_id').to_csv(output / SUBMISSION, encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(output / 'B_tolerance_comparison.csv', index=False, encoding='utf-8-sig')
    summary = {**report, 'model_type': saved['model_type'], 'public_actual': public_actual, 'public_files': provenance,
               'public_dates': data.city.dates[public_days].strftime('%Y-%m-%d').tolist(), 'stations': len(data.city.stations), 'days': len(allocation),
               'budget_reference': '1% of observed public-day baseline score, not 1% of unknown full-month B score.',
               'note': 'Costs omit C(0), which cancels in cost differences. No percentage is applied to a negative relative objective. Full B actual score remains unknown.'}
    (output / 'B_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'B_frontier.npz', predicted_relative=prepared['predicted'], public_actual=prepared['public'])
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build','predict'], nargs='?', default='predict')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1])
    parser.add_argument('--source-model-dir', type=Path, default=ROOT / 'results_fleet')
    parser.add_argument('--signature-dir', type=Path, default=ROOT / 'results_v2')
    parser.add_argument('--public-dir', type=Path, default=ROOT.parents[1] / 'open_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results_bounded')
    parser.add_argument('--tolerance', type=float, default=None)
    args = parser.parse_args()
    tolerance = 0.01 if args.tolerance is None else args.tolerance
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == 'build':
        source = torch.load(args.source_model_dir / networks.MODEL_NAME, map_location='cpu', weights_only=True)
        carrier = {k: source[k] for k in ('model_type','prior','signature','public_days','weights')}
        saved = {'model_type': 'bounded_neural_inventory_v1', 'tolerance': float(tolerance), 'networks': carrier,
                 'reference': 'public-day baseline official score', 'constraints': ['predicted_delta <= budget', 'public_actual_delta <= budget']}
        torch.save(saved, args.output_dir / MODEL_NAME)
        validate(networks.neural.base.read_data(args.data_dir, 'A'), args.signature_dir, args.output_dir, tolerance)
    export(networks.neural.base.read_data(args.data_dir, 'B'), args.public_dir, args.output_dir, args.tolerance)


if __name__ == '__main__':
    main()
