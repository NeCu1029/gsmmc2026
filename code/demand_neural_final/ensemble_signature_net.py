"""Equal-weight ensemble of prior and public-signature neural marginal costs."""
import argparse
import json
from pathlib import Path
import bike_signature_net as new
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent


def old_costs(saved, data, days, sites, public_days, truth):
    models = []
    for state in saved['states']:
        net = new.inventory.InventoryNet(saved['ns'], saved['nt'])
        net.load_state_dict(state)
        models.append(net)
    pred = new.inventory.ensemble(models, data, days, sites)
    public = new.inventory.ensemble(models, data, public_days, sites)
    config = saved['config']
    return new.transfer.adjusted(pred, public, new.transfer.increments(truth, data.capacity)[:, sites], data.city.off_day[days],
                                 data.city.off_day[public_days], config['weekday_weight'], config['offday_weight'])


def validate(data, previous, output):
    old_report = json.loads((previous / 'public_transfer_validation.json').read_text(encoding='utf-8'))
    raw_report = json.loads((output / 'A_validation.json').read_text(encoding='utf-8'))
    with np.load(output / 'A_signature_costs.npz', allow_pickle=False) as z:
        new_pred = z['increments'].copy()
    public_days = data.city.dates.get_indexer(pd.to_datetime(old_report['public_dates']))
    target = new.transfer.increments(data.city.net_flow.numpy(), data.capacity)
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    past, future = days[:-7], days[-7:]
    order = np.random.default_rng(new.base.SEED).permutation(len(sites))
    groups = np.split(order, [i * (len(sites) // 5) for i in range(1, 5)])
    allocated = np.empty((len(future), len(sites)), dtype=np.int64)
    costs = np.empty_like(new_pred)
    records = []
    for i, test in enumerate(groups):
        sources = np.concatenate(groups[i+1:] + groups[:i])
        config = old_report['folds'][i]['config']
        models = [new.inventory.fit(data, target, past, sources, seed, epoch)[0] for seed, epoch in zip(new.inventory.SEEDS, config['epochs'])]
        saved = {'ns': data.similarity.shape[-1], 'nt': new.inventory.day_features(data).shape[-1], 'config': config,
                 'states': [m.state_dict() for m in models]}
        prior = old_costs(saved, data, future, test, public_days, data.city.net_flow[public_days].numpy())
        old_alloc, _ = new.optimize(new.inventory.cost_table(prior, data.capacity[test]), data.capacity[test])
        old_score = new.realized_score(data.city.net_flow[future][:, test].numpy(), old_alloc, data.capacity[test])['score']
        assert abs(old_score - old_report['folds'][i]['actual_score']['score']) < 1e-7
        raw = new_pred[:, test]
        raw_alloc, _ = new.optimize(new.inventory.cost_table(raw, data.capacity[test]), data.capacity[test])
        raw_score = new.realized_score(data.city.net_flow[future][:, test].numpy(), raw_alloc, data.capacity[test])['score']
        assert abs(raw_score - raw_report['folds'][i]['actual_score']['score']) < 1e-7
        costs[:, test] = 0.5 * (prior + raw)
        allocated[:, test], _ = new.optimize(new.inventory.cost_table(costs[:, test], data.capacity[test]), data.capacity[test])
        result = new.realized_score(data.city.net_flow[future][:, test].numpy(), allocated[:, test], data.capacity[test])
        records.append({'test_station_ids': data.city.stations[test].tolist(), 'actual_score': result})
        print(f'Equal ensemble fold {i+1}/5: {result}', flush=True)
    report = {'protocol': raw_report['protocol'], 'weights': [0.5, 0.5], 'weight_selection': 'Equal weights fixed before evaluating this ensemble; repeated A development validation is not an independent final test.',
              'folds': records, 'actual_score': new.realized_score(data.city.net_flow[future].numpy(), allocated, data.capacity)}
    (output / 'A_ensemble_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(allocated.T, index=data.city.stations, columns=[f'D{d+1:02d}' for d in future]).rename_axis('station_id').to_csv(output / 'A_앙상블검증배치.csv', encoding='utf-8-sig')
    np.savez_compressed(output / 'A_ensemble_costs.npz', increments=costs)
    print(json.dumps(report['actual_score']), flush=True)


def export(data, public_dir, previous, output):
    public_days, truth, provenance = new.transfer.read_public(public_dir, data)
    saved = torch.load(previous / 'public_inventory.pt', map_location='cpu', weights_only=True)
    if saved.get('model_type') != 'public_marginal_inventory_v1' or saved['public_days'] != public_days.tolist():
        raise ValueError('Prior checkpoint/public dates mismatch')
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    prior = old_costs(saved, data, days, sites, public_days, truth)
    signature = torch.load(output / new.MODEL, map_location='cpu', weights_only=True)
    if signature.get('model_type') != 'public_signature_v2' or signature['public_days'] != public_days.tolist():
        raise ValueError('Signature checkpoint/public dates mismatch')
    inputs = new.make_inputs(data, truth)
    models = []
    for state in signature['states']:
        net = new.SignatureNet(signature['ns'], signature['nt'])
        net.load_state_dict(state)
        models.append(net)
    config = signature['config']
    raw = new.blend(new.ensemble(models, inputs, days, sites), inputs.anchor.numpy(), data.city.off_day, config['weekday_blend'], config['holiday_blend'])
    costs = 0.5 * (prior + raw)
    costs[public_days] = new.transfer.increments(truth, data.capacity)
    allocation, relative = new.optimize(new.inventory.cost_table(costs, data.capacity), data.capacity)
    pd.DataFrame(allocation.T, index=data.city.stations, columns=[f'D{d+1:02d}' for d in days]).rename_axis('station_id').to_csv(output / new.SUBMISSION, encoding='utf-8-sig')
    result = {'model_type': 'equal_public_signature_ensemble_v2', 'weights': [0.5, 0.5], 'N': int(allocation[0].sum()), 'stations': len(sites), 'days': len(days),
              'public_files': provenance, 'public_dates': data.city.dates[public_days].strftime('%Y-%m-%d').tolist(),
              'public_actual_score': new.realized_score(truth, allocation[public_days], data.capacity), 'relative_objective': relative,
              'note': 'Unknown full-month B score. Public-day score is in-sample. Relative objective is not actual competition score.'}
    (output / 'B_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'predict'], default='predict', nargs='?')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1])
    parser.add_argument('--public-dir', type=Path, default=ROOT.parents[1] / 'open_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results_v2')
    parser.add_argument('--previous-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.action == 'validate':
        validate(new.base.read_data(args.data_dir, 'A'), args.previous_dir, args.output_dir)
    export(new.base.read_data(args.data_dir, 'B'), args.public_dir, args.previous_dir, args.output_dir)


if __name__ == '__main__':
    main()
