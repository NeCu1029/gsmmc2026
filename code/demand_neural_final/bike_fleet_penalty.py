"""Neural marginal-cost ensemble plus a learned fleet-penalty decision layer."""
import argparse
import json
from pathlib import Path
import ensemble_signature_net as ensemble
import bike_signature_net as neural
import numpy as np
import pandas as pd
import torch
from neural_allocation import dp_all_totals, dp_reconstruct, realized_score

ROOT = Path(__file__).resolve().parent
ALPHAS = (0., 5., 10., 20., 40., 80.)
MODEL_NAME = 'fleet_penalty_model.pt'
SUBMISSION = '최종제출_배치표_B지역_N추가벌점.csv'


def nonnegative(value, name):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def frontier(costs, capacities, alphas=ALPHAS, truth=None):
    """Exact optimizer of C_hat(x) + (20 + alpha) N; official formula stays 20 N."""
    for alpha in alphas:
        nonnegative(alpha, 'alpha')
    capacities = np.asarray(capacities, dtype=np.int64)
    if costs.ndim != 3 or costs.shape[1] != len(capacities) or (capacities <= 0).any():
        raise ValueError('Incorrect cost/capacity dimensions')
    valid = np.arange(costs.shape[-1])[None, None] <= capacities[None, :, None]
    if costs.shape[-1] <= capacities.max() or not np.isfinite(costs[np.broadcast_to(valid, costs.shape)]).all():
        raise ValueError('Missing or non-finite valid costs')
    masked = np.where(valid, costs, np.inf)
    totals = np.stack([dp_all_totals(day, capacities) for day in masked])
    fleets = np.arange(totals.shape[1])
    base = 20 * fleets + totals.mean(0)
    cache, rows = {}, []
    for alpha in alphas:
        objective = base + alpha * fleets
        objective[0] = np.inf
        n = int(np.argmin(objective))
        if n not in cache:
            cache[n] = np.stack([dp_reconstruct(day, capacities, n) for day in masked])
        allocation = cache[n]
        if (allocation < 0).any() or (allocation > capacities).any() or not (allocation.sum(1) == n).all():
            raise AssertionError('Invalid integer allocation')
        obtained = masked[np.arange(len(costs))[:, None], np.arange(len(capacities)), allocation].sum(1)
        if not np.allclose(obtained, totals[:, n]):
            raise AssertionError('DP reconstruction mismatch')
        row = {'alpha': float(alpha), 'N': n, 'effective_fleet_price': 20 + float(alpha),
               'relative_objective': float(objective[n]), 'relative_base_objective': float(base[n])}
        if truth is not None:
            row['actual'] = realized_score(truth, allocation, capacities)
        rows.append((row, allocation))
    ordered = sorted(rows, key=lambda item: item[0]['alpha'])
    if any(b[0]['N'] > a[0]['N'] for a, b in zip(ordered, ordered[1:])):
        raise AssertionError('Increasing fleet penalty increased N')
    return rows


def select_alpha(data, signature_inputs, target, sources, public_days, old_config, new_config, policy_weight):
    """One inner station holdout, D01-D16 fit / D17-D23 policy selection."""
    groups = np.array_split(sources, 3)
    core, dev = np.concatenate(groups[:2]), groups[2]
    past, validation = np.arange(16), np.arange(16, 23)
    if not set(public_days).issubset(past):
        raise ValueError('Public dates must precede inner validation')
    models = [neural.inventory.fit(data, target, past, core, seed, epoch)[0]
              for seed, epoch in zip(neural.inventory.SEEDS, old_config['epochs'])]
    saved = {'ns': data.similarity.shape[-1], 'nt': neural.inventory.day_features(data).shape[-1],
             'config': old_config, 'states': [m.state_dict() for m in models]}
    prior = ensemble.old_costs(saved, data, validation, dev, public_days, data.city.net_flow[public_days].numpy())
    models = [neural.fit(signature_inputs, target, np.setdiff1d(past, public_days), core, seed, epoch)[0]
              for seed, epoch in zip(neural.SEEDS, new_config['epochs'])]
    raw = neural.ensemble(models, signature_inputs, validation, dev)
    new_cost = neural.blend(raw, signature_inputs.anchor[dev].numpy(), data.city.off_day[validation],
                            new_config['weekday_blend'], new_config['holiday_blend'])
    costs = neural.inventory.cost_table(0.5 * (prior + new_cost), data.capacity[dev])
    entries = frontier(costs, data.capacity[dev], truth=data.city.net_flow[validation][:, dev].numpy())
    options = []
    for row, _ in entries:
        options.append({**row, 'selection_objective': row['actual']['score'] + policy_weight * row['N']})
    # Positive alpha is explicitly requested; alpha zero remains a reported baseline.
    chosen = min((r for r in options if r['alpha'] > 0), key=lambda r: (r['selection_objective'], r['N'], r['alpha']))
    return {'alpha': chosen['alpha'], 'policy_weight': float(policy_weight), 'validation_station_ids': data.city.stations[dev].tolist(),
            'selected_inner_score': chosen['actual'], 'options': options}


def train(data, prior_dir, signature_dir, output, policy_weight):
    nonnegative(policy_weight, 'policy_weight')
    old = json.loads((prior_dir / 'public_transfer_validation.json').read_text(encoding='utf-8'))
    new = json.loads((signature_dir / 'A_validation.json').read_text(encoding='utf-8'))
    baseline = json.loads((signature_dir / 'A_ensemble_validation.json').read_text(encoding='utf-8'))
    with np.load(signature_dir / 'A_ensemble_costs.npz', allow_pickle=False) as archive:
        oof = archive['increments'].copy()
    public_days = data.city.dates.get_indexer(pd.to_datetime(old['public_dates']))
    inputs = neural.make_inputs(data, data.city.net_flow[public_days].numpy())
    target = neural.transfer.increments(data.city.net_flow.numpy(), data.capacity)
    order = np.random.default_rng(neural.base.SEED).permutation(len(data.city.stations))
    groups = np.split(order, [i * (len(order) // 5) for i in range(1, 5)])
    future = np.arange(23, 30)
    allocation = np.empty(oof.shape[:2], dtype=np.int64)
    baseline_alloc = np.empty_like(allocation)
    records, fixed = [], {a: [] for a in ALPHAS}
    for i, test in enumerate(groups):
        print(f'Fleet penalty outer {i+1}/5: fitting inner policy', flush=True)
        sources = np.concatenate(groups[i+1:] + groups[:i])
        if data.city.stations[test].tolist() != baseline['folds'][i]['test_station_ids']:
            raise ValueError('OOF station folds mismatch')
        policy = select_alpha(data, inputs, target, sources, public_days, old['folds'][i]['config'], new['folds'][i]['config'], policy_weight)
        entries = frontier(neural.inventory.cost_table(oof[:, test], data.capacity[test]), data.capacity[test], truth=data.city.net_flow[future][:, test].numpy())
        for row, candidate in entries:
            fixed[row['alpha']].append(row)
            if row['alpha'] == 0:
                baseline_alloc[:, test] = candidate
            if row['alpha'] == policy['alpha']:
                allocation[:, test] = candidate
                result = row['actual']
        records.append({'test_station_ids': data.city.stations[test].tolist(), 'policy': policy, 'actual': result})
        print(json.dumps({'alpha': policy['alpha'], 'actual': result}), flush=True)
        (output / 'progress.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    baseline_score = realized_score(data.city.net_flow[future].numpy(), baseline_alloc, data.capacity)
    if abs(baseline_score['score'] - baseline['actual_score']['score']) > 1e-7:
        raise AssertionError('Baseline forecasts do not reproduce the previous model')
    print('Selecting final fleet policy on D17-D23 only', flush=True)
    policy = select_alpha(data, inputs, target, order, public_days, old['final_config'], new['final_config'], policy_weight)
    prior = torch.load(prior_dir / 'public_inventory.pt', map_location='cpu', weights_only=True)
    signature = torch.load(signature_dir / neural.MODEL, map_location='cpu', weights_only=True)
    torch.save({'model_type': 'neural_fleet_penalty_v1', 'alpha': policy['alpha'], 'policy_weight': float(policy_weight),
                'prior': prior, 'signature': signature, 'public_days': public_days.tolist(), 'weights': [0.5, 0.5]}, output / MODEL_NAME)
    actual = realized_score(data.city.net_flow[future].numpy(), allocation, data.capacity)
    table = []
    for alpha, entries in fixed.items():
        combined = {k: sum(r['actual'][k] for r in entries) for k in ('score', 'N', 'Q_daily', 'O_daily', 'E_daily')}
        table.append({'alpha': alpha, **combined, 'regularized_score': combined['score'] + policy_weight * combined['N']})
    pd.DataFrame(table).to_csv(output / 'A_alpha_comparison.csv', index=False, encoding='utf-8-sig')
    report = {'protocol': 'Five station outer folds. Neural OOF predictions fit other stations on D01-D23, target station public D09/D10 only. Each alpha selected on a separate inner station holdout: fit D01-D16, select D17-D23. Outer test D24-D30 never selects alpha.',
              'policy_objective': 'official_A_score + policy_weight * N', 'policy_weight': float(policy_weight), 'final_policy': policy,
              'folds': records, 'baseline_actual': baseline_score, 'actual': actual, 'fixed_alpha_comparison': table,
              'model_note': 'Neural forecast weights are reused unchanged. The newly fitted component is the fleet-penalty decision policy.'}
    (output / 'A_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(allocation.T, index=data.city.stations, columns=[f'D{d+1:02d}' for d in future]).rename_axis('station_id').to_csv(output / 'A_검증배치.csv', encoding='utf-8-sig')
    print(json.dumps({'alpha': policy['alpha'], 'baseline_actual': baseline_score, 'actual': actual}), flush=True)


def predict_costs(saved, data, public_days, truth):
    if saved.get('model_type') != 'neural_fleet_penalty_v1' or saved['public_days'] != public_days.tolist():
        raise ValueError('Fleet model/public dates mismatch')
    for checkpoint, kind in ((saved['prior'], 'public_marginal_inventory_v1'), (saved['signature'], 'public_signature_v2')):
        if checkpoint.get('model_type') != kind or checkpoint['public_days'] != public_days.tolist():
            raise ValueError('Ensemble checkpoint/public dates mismatch')
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    prior = ensemble.old_costs(saved['prior'], data, days, sites, public_days, truth)
    inputs = neural.make_inputs(data, truth)
    checkpoint = saved['signature']
    models = []
    for state in checkpoint['states']:
        model = neural.SignatureNet(checkpoint['ns'], checkpoint['nt'])
        model.load_state_dict(state)
        models.append(model)
    config = checkpoint['config']
    raw = neural.blend(neural.ensemble(models, inputs, days, sites), inputs.anchor.numpy(), data.city.off_day,
                        config['weekday_blend'], config['holiday_blend'])
    increments = 0.5 * (prior + raw)
    increments[public_days] = neural.transfer.increments(truth, data.capacity)
    if not np.isfinite(increments).all():
        raise ValueError('Non-finite neural costs')
    return neural.inventory.cost_table(increments, data.capacity)


def export(data, public_dir, output, alpha_override=None):
    public_days, truth, provenance = neural.transfer.read_public(public_dir, data)
    saved = torch.load(output / MODEL_NAME, map_location='cpu', weights_only=True)
    alpha = saved['alpha'] if alpha_override is None else float(alpha_override)
    nonnegative(alpha, 'alpha')
    costs = predict_costs(saved, data, public_days, truth)
    entries = frontier(costs, data.capacity, sorted(set(ALPHAS + (alpha,))))
    rows, chosen, baseline = [], None, None
    for row, allocation in entries:
        public = realized_score(truth, allocation[public_days], data.capacity)
        rows.append({**row, 'public_actual_score': public['score']})
        if row['alpha'] == alpha:
            chosen, selected_row, public_score = allocation, row, public
        if row['alpha'] == 0:
            baseline = allocation
    if chosen is None or baseline is None:
        raise AssertionError('Missing selected policy/baseline')
    pd.DataFrame(chosen.T, index=data.city.stations, columns=[f'D{d+1:02d}' for d in range(len(chosen))]).rename_axis('station_id').to_csv(output / SUBMISSION, encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(output / 'B_alpha_comparison.csv', index=False, encoding='utf-8-sig')
    summary = {'model_type': saved['model_type'], 'alpha': alpha, 'trained_alpha': saved['alpha'], 'policy_weight': saved['policy_weight'],
               'effective_fleet_price': 20 + alpha, 'N': int(chosen[0].sum()), 'baseline_N': int(baseline[0].sum()),
               'N_reduction_percent': 100 * (baseline[0].sum() - chosen[0].sum()) / baseline[0].sum(),
               'stations': len(data.city.stations), 'days': len(chosen), 'relative_objective': selected_row['relative_objective'],
               'public_actual': public_score, 'public_dates': data.city.dates[public_days].strftime('%Y-%m-%d').tolist(), 'public_files': provenance,
               'note': 'Official scoring stays 20*N. alpha*N is an internal reduction preference. Relative objectives omit C(0); full-month B actual score is unknown.'}
    summary['N_reduction_percent'] = float(summary['N_reduction_percent'])
    (output / 'B_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'predict'], nargs='?', default='predict')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1])
    parser.add_argument('--prior-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--signature-dir', type=Path, default=ROOT / 'results_v2')
    parser.add_argument('--public-dir', type=Path, default=ROOT.parents[1] / 'open_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results_fleet')
    parser.add_argument('--policy-weight', type=float, default=20.)
    parser.add_argument('--alpha', type=float, help='Override trained alpha at prediction time')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == 'train':
        train(neural.base.read_data(args.data_dir, 'A'), args.prior_dir, args.signature_dir, args.output_dir, args.policy_weight)
    export(neural.base.read_data(args.data_dir, 'B'), args.public_dir, args.output_dir, args.alpha)


if __name__ == '__main__':
    main()
