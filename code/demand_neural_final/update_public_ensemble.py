"""Update the original unpenalized 1:1 neural ensemble with additional public days."""
import argparse
import json
from pathlib import Path
import ensemble_signature_net as original
import bike_signature_net as neural
import numpy as np
import pandas as pd
import torch
from neural_allocation import optimize, realized_score

ROOT = Path(__file__).resolve().parent
MODEL_NAME = 'public_update_model.pt'
SUBMISSION = '최종제출_배치표_B지역_D23추가.csv'


def check_public_protocol(prior, signature, public_days):
    if prior.get('model_type') != 'public_marginal_inventory_v1' or signature.get('model_type') != 'public_signature_v2':
        raise ValueError('Expected the original unpenalized neural predictors')
    if prior['public_days'] != signature['public_days']:
        raise ValueError('Original checkpoint public-day protocols differ')
    if not set(prior['public_days']).issubset(set(np.asarray(public_days).tolist())):
        raise ValueError('An originally available public day is missing')


def infer(data, prior, signature, public_days, truth, days, sites):
    check_public_protocol(prior, signature, public_days)
    prior_costs = original.old_costs(prior, data, days, sites, public_days, truth)
    inputs = neural.make_inputs(data, truth)
    if inputs.signature.shape[-1] != signature['ns']:
        raise ValueError('Public signature feature dimension changed')
    models = []
    for state in signature['states']:
        net = neural.SignatureNet(signature['ns'], signature['nt'])
        net.load_state_dict(state)
        models.append(net)
    config = signature['config']
    raw = neural.blend(neural.ensemble(models, inputs, days, sites), inputs.anchor[sites].numpy(), data.city.off_day[days],
                        config['weekday_blend'], config['holiday_blend'])
    result = 0.5 * (prior_costs + raw)
    actual = neural.transfer.increments(truth, data.capacity)
    positions = {int(d): i for i, d in enumerate(public_days)}
    for i, day in enumerate(days):
        if int(day) in positions:
            result[i] = actual[positions[int(day)], sites]
    if not np.isfinite(result).all():
        raise ValueError('Non-finite updated cost predictions')
    return result


def validate(data, public_days, prior_dir, signature_dir, output):
    old_report = json.loads((prior_dir / 'public_transfer_validation.json').read_text(encoding='utf-8'))
    signature_report = json.loads((signature_dir / 'A_validation.json').read_text(encoding='utf-8'))
    baseline_report = json.loads((signature_dir / 'A_ensemble_validation.json').read_text(encoding='utf-8'))
    trained_public = data.city.dates.get_indexer(pd.to_datetime(old_report['public_dates']))
    if (public_days >= 23).any():
        raise ValueError('Public dates inside D24-D30 require a revised validation split')
    target = neural.transfer.increments(data.city.net_flow.numpy(), data.capacity)
    original_inputs = neural.make_inputs(data, data.city.net_flow[trained_public].numpy())
    with np.load(signature_dir / 'A_ensemble_costs.npz', allow_pickle=False) as z:
        previous_predictions = z['increments'].copy()
    order = np.random.default_rng(neural.base.SEED).permutation(len(data.city.stations))
    groups = np.split(order, [i * (len(order) // 5) for i in range(1, 5)])
    past, future = np.arange(23), np.arange(23, 30)
    all_predictions = np.empty_like(previous_predictions)
    allocated, previous_alloc = np.empty(previous_predictions.shape[:2], dtype=np.int64), np.empty(previous_predictions.shape[:2], dtype=np.int64)
    records = []
    for i, test in enumerate(groups):
        print(f'Public update validation fold {i+1}/5', flush=True)
        sources = np.concatenate(groups[i+1:] + groups[:i])
        old_config = old_report['folds'][i]['config']
        new_config = signature_report['folds'][i]['config']
        models = [neural.inventory.fit(data, target, past, sources, seed, epoch)[0]
                  for seed, epoch in zip(neural.inventory.SEEDS, old_config['epochs'])]
        prior = {'model_type': 'public_marginal_inventory_v1', 'public_days': trained_public.tolist(),
                 'ns': data.similarity.shape[-1], 'nt': neural.inventory.day_features(data).shape[-1],
                 'config': old_config, 'states': [m.state_dict() for m in models]}
        models = [neural.fit(original_inputs, target, np.setdiff1d(past, trained_public), sources, seed, epoch)[0]
                  for seed, epoch in zip(neural.SEEDS, new_config['epochs'])]
        signature = {'model_type': 'public_signature_v2', 'public_days': trained_public.tolist(),
                     'ns': original_inputs.signature.shape[-1], 'nt': neural.inventory.day_features(data).shape[-1],
                     'config': new_config, 'states': [m.state_dict() for m in models]}
        reproduced = infer(data, prior, signature, trained_public, data.city.net_flow[trained_public].numpy(), future, test)
        np.testing.assert_allclose(reproduced, previous_predictions[:, test], rtol=0, atol=1e-7)
        predicted = infer(data, prior, signature, public_days, data.city.net_flow[public_days].numpy(), future, test)
        all_predictions[:, test] = predicted
        allocated[:, test], _ = optimize(neural.inventory.cost_table(predicted, data.capacity[test]), data.capacity[test])
        previous_alloc[:, test], _ = optimize(neural.inventory.cost_table(reproduced, data.capacity[test]), data.capacity[test])
        actual = data.city.net_flow[future][:, test].numpy()
        entry = {'test_station_ids': data.city.stations[test].tolist(), 'previous': realized_score(actual, previous_alloc[:, test], data.capacity[test]),
                 'updated': realized_score(actual, allocated[:, test], data.capacity[test])}
        records.append(entry)
        print(json.dumps(entry['updated']), flush=True)
        (output / 'validation_progress.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    actual = data.city.net_flow[future].numpy()
    previous = realized_score(actual, previous_alloc, data.capacity)
    if abs(previous['score'] - baseline_report['actual_score']['score']) > 1e-7:
        raise AssertionError('Prior ensemble validation mismatch')
    updated = realized_score(actual, allocated, data.capacity)
    report = {'protocol': baseline_report['protocol'], 'trained_public_dates': data.city.dates[trained_public].strftime('%Y-%m-%d').tolist(),
              'updated_public_dates': data.city.dates[public_days].strftime('%Y-%m-%d').tolist(), 'previous': previous, 'updated': updated,
              'score_change_percent': 100 * (updated['score'] / previous['score'] - 1), 'folds': records,
              'note': 'Original neural parameters/configurations are reproduced unchanged. Only inference-time public observations grow from D09/D10 to D09/D10/D23; D24-D30 labels do not enter the update.'}
    (output / 'A_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(allocated.T, index=data.city.stations, columns=[f'D{d+1:02d}' for d in future]).rename_axis('station_id').to_csv(output / 'A_검증배치.csv', encoding='utf-8-sig')
    np.savez_compressed(output / 'A_updated_costs.npz', increments=all_predictions)
    print(json.dumps({'previous': previous, 'updated': updated, 'score_change_percent': report['score_change_percent']}), flush=True)


def export(data, public_dir, signature_dir, output):
    saved = torch.load(output / MODEL_NAME, map_location='cpu', weights_only=True)
    if saved.get('model_type') != 'public_observation_update_v1':
        raise ValueError('Incorrect public update model')
    public_days, truth, provenance = neural.transfer.read_public(public_dir, data)
    if saved['observed_public_days'] != public_days.tolist():
        raise ValueError('Public-day set changed; run build to validate and record the new observations')
    prior, signature = saved['prior'], saved['signature']
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    predictions = infer(data, prior, signature, public_days, truth, days, sites)
    allocated, relative = optimize(neural.inventory.cost_table(predictions, data.capacity), data.capacity)
    trained_public = np.array(prior['public_days'])
    lookup = {int(d): i for i, d in enumerate(public_days)}
    old_truth = truth[[lookup[int(d)] for d in trained_public]]
    original_predictions = infer(data, prior, signature, trained_public, old_truth, days, sites)
    baseline, _ = optimize(neural.inventory.cost_table(original_predictions, data.capacity), data.capacity)
    previous_file = signature_dir / '최종제출_배치표_B지역_개선2.csv'
    if previous_file.is_file():
        expected = pd.read_csv(previous_file, index_col='station_id').loc[data.city.stations].to_numpy().T
        np.testing.assert_array_equal(baseline, expected)
    pd.DataFrame(allocated.T, index=data.city.stations, columns=[f'D{d+1:02d}' for d in days]).rename_axis('station_id').to_csv(output / SUBMISSION, encoding='utf-8-sig')
    previous_public = realized_score(truth, baseline[public_days], data.capacity)
    updated_public = realized_score(truth, allocated[public_days], data.capacity)
    individual = []
    for i, day in enumerate(public_days):
        individual.append({'date': data.city.dates[day].strftime('%Y-%m-%d'),
                           'previous_score': realized_score(truth[i:i+1], baseline[day:day+1], data.capacity)['score'],
                           'updated_score': realized_score(truth[i:i+1], allocated[day:day+1], data.capacity)['score']})
    pd.DataFrame(individual).to_csv(output / 'B_public_day_comparison.csv', index=False, encoding='utf-8-sig')
    result = {'model_type': saved['model_type'], 'N': int(allocated[0].sum()), 'baseline_N': int(baseline[0].sum()),
              'stations': len(sites), 'days': len(days), 'weights': [0.5,0.5], 'official_fleet_coefficient': 20,
              'extra_fleet_penalty': 0, 'cost_bound': None, 'trained_public_dates': data.city.dates[trained_public].strftime('%Y-%m-%d').tolist(),
              'updated_public_dates': data.city.dates[public_days].strftime('%Y-%m-%d').tolist(), 'public_files': provenance,
              'previous_public_actual': previous_public, 'updated_public_actual': updated_public,
              'changed_station_days': int((allocated != baseline).sum()), 'relative_objective': relative,
              'note': 'Neural weights and 1:1 ensemble are unchanged. Public observation summaries/calibration now include D23. Public scores are in-sample; unobserved full-month B score remains unknown.'}
    (output / 'B_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build','predict'], nargs='?', default='predict')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1])
    parser.add_argument('--prior-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--signature-dir', type=Path, default=ROOT / 'results_v2')
    parser.add_argument('--public-dir', type=Path, default=ROOT.parents[1] / 'open_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results_public_d23')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b = neural.base.read_data(args.data_dir, 'B')
    if args.action == 'build':
        public_days, _, _ = neural.transfer.read_public(args.public_dir, b)
        prior = torch.load(args.prior_dir / 'public_inventory.pt', map_location='cpu', weights_only=True)
        signature = torch.load(args.signature_dir / neural.MODEL, map_location='cpu', weights_only=True)
        check_public_protocol(prior, signature, public_days)
        torch.save({'model_type':'public_observation_update_v1','prior':prior,'signature':signature,
                    'observed_public_days':public_days.tolist(),'weights':[0.5,0.5]}, args.output_dir / MODEL_NAME)
        validate(neural.base.read_data(args.data_dir, 'A'), public_days, args.prior_dir, args.signature_dir, args.output_dir)
    export(b, args.public_dir, args.signature_dir, args.output_dir)


if __name__ == '__main__':
    main()
