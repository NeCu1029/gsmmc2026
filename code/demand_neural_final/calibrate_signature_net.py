"""Select a marginal-cost calibration on inner folds, then evaluate untouched outer days."""
import argparse
import itertools
import json
from pathlib import Path
import bike_signature_net as model
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
OFFSETS = (-5.0, 0.0, 5.0, 10.0)
MODEL = 'signature_calibrated.pt'
SUBMISSION = '최종제출_배치표_B지역_개선2.csv'


def corrected(pred, off, config):
    return pred + np.where(off, config['holiday_offset'], config['weekday_offset'])[:, None, None]


def fitted(inputs, target, days, sites, config):
    return [model.fit(inputs, target, days, sites, seed, epoch)[0] for seed, epoch in zip(model.SEEDS, config['epochs'])]


def mixed_prediction(models, inputs, days, sites, config):
    pred = model.ensemble(models, inputs, days, sites)
    return model.blend(pred, inputs.anchor[sites].numpy(), inputs.source.city.off_day[days], config['weekday_blend'], config['holiday_blend'])


def choose_offset(inputs, target, sites, days, public_days, config):
    past, validation = np.setdiff1d(days[:-7], public_days), days[-7:]
    groups = np.array_split(sites, 3)
    candidates = list(itertools.product(OFFSETS, repeat=2))
    scores = {c: [] for c in candidates}
    data = inputs.source
    for i, dev in enumerate(groups):
        core = np.concatenate(groups[:i] + groups[i+1:])
        networks = fitted(inputs, target, past, core, config)
        pred = mixed_prediction(networks, inputs, validation, dev, config)
        for c in candidates:
            adjusted = corrected(pred, data.city.off_day[validation], dict(weekday_offset=c[0], holiday_offset=c[1]))
            allocation, _ = model.optimize(model.inventory.cost_table(adjusted, data.capacity[dev]), data.capacity[dev])
            scores[c].append(model.realized_score(data.city.net_flow[validation][:, dev].numpy(), allocation, data.capacity[dev])['score'])
        print(f'  cost calibration inner {i+1}/3 done', flush=True)
    options = [dict(weekday_offset=c[0], holiday_offset=c[1], selection_score=sum(v)) for c, v in scores.items()]
    return min(options, key=lambda x: x['selection_score']), options


def train(data, public_days, output):
    previous = json.loads((output / 'A_validation.json').read_text(encoding='utf-8'))
    inputs = model.make_inputs(data, data.city.net_flow[public_days].numpy())
    target = model.transfer.increments(data.city.net_flow.numpy(), data.capacity)
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    past, future = days[:-7], days[-7:]
    order = np.random.default_rng(model.base.SEED).permutation(len(sites))
    groups = np.split(order, [i * (len(sites) // 5) for i in range(1, 5)])
    predictions = np.empty((len(future), len(sites), int(data.capacity.max())))
    allocated = np.empty((len(future), len(sites)), dtype=np.int64)
    records = []
    for i, test in enumerate(groups):
        print(f'Calibrated outer {i+1}/5', flush=True)
        sources = np.concatenate(groups[i+1:] + groups[:i])
        config = previous['folds'][i]['config']
        offset, search = choose_offset(inputs, target, sources, past, public_days, config)
        networks = fitted(inputs, target, np.setdiff1d(past, public_days), sources, config)
        pred = mixed_prediction(networks, inputs, future, test, config)
        predictions[:, test] = corrected(pred, data.city.off_day[future], offset)
        allocated[:, test], _ = model.optimize(model.inventory.cost_table(predictions[:, test], data.capacity[test]), data.capacity[test])
        result = model.realized_score(data.city.net_flow[future][:, test].numpy(), allocated[:, test], data.capacity[test])
        records.append({'test_station_ids': data.city.stations[test].tolist(), 'config': config, 'offset': offset, 'search': search, 'actual_score': result})
        print(json.dumps({'offset': offset, 'actual_score': result}), flush=True)
        (output / 'calibration_progress.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    offset, search = choose_offset(inputs, target, order, past, public_days, previous['final_config'])
    saved = torch.load(output / model.MODEL, map_location='cpu', weights_only=True)
    saved.update({'model_type': 'public_signature_calibrated_v2', 'offset': offset})
    torch.save(saved, output / MODEL)
    report = {'protocol': previous['protocol'] + ' Cost offsets selected on inner D17-D23 only; official penalty coefficients are unchanged.',
              'public_dates': previous['public_dates'], 'folds': records, 'final_config': previous['final_config'], 'final_offset': offset, 'final_search': search,
              'actual_score': model.realized_score(data.city.net_flow[future].numpy(), allocated, data.capacity)}
    (output / 'A_calibrated_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(allocated.T, index=data.city.stations, columns=[f'D{d+1:02d}' for d in future]).rename_axis('station_id').to_csv(output / 'A_보정검증배치.csv', encoding='utf-8-sig')
    np.savez_compressed(output / 'A_oof_costs.npz', increments=predictions, caps=data.capacity, ids=data.city.stations.to_numpy(str), days=future)
    print(json.dumps({'offset': offset, 'actual_score': report['actual_score']}), flush=True)


def export(data, public_dir, output):
    public_days, truth, provenance = model.transfer.read_public(public_dir, data)
    saved = torch.load(output / MODEL, map_location='cpu', weights_only=True)
    if saved.get('model_type') != 'public_signature_calibrated_v2' or saved['public_days'] != public_days.tolist():
        raise ValueError('Checkpoint/public-day protocol mismatch')
    inputs = model.make_inputs(data, truth)
    models = []
    for state in saved['states']:
        net = model.SignatureNet(saved['ns'], saved['nt'])
        net.load_state_dict(state)
        models.append(net)
    days, sites = np.arange(len(data.city.dates)), np.arange(len(data.city.stations))
    pred = corrected(mixed_prediction(models, inputs, days, sites, saved['config']), data.city.off_day, saved['offset'])
    pred[public_days] = model.transfer.increments(truth, data.capacity)
    if not np.isfinite(pred).all():
        raise ValueError('Non-finite cost predictions')
    allocation, relative = model.optimize(model.inventory.cost_table(pred, data.capacity), data.capacity)
    pd.DataFrame(allocation.T, index=data.city.stations, columns=[f'D{d+1:02d}' for d in days]).rename_axis('station_id').to_csv(output / SUBMISSION, encoding='utf-8-sig')
    np.savez_compressed(output / 'B_calibrated_costs.npz', increments=pred, capacity=data.capacity, ids=data.city.stations.to_numpy(str))
    report = {'N': int(allocation[0].sum()), 'stations': len(sites), 'days': len(days), 'model_type': saved['model_type'],
              'config': saved['config'], 'offset': saved['offset'], 'public_files': provenance,
              'public_dates': data.city.dates[public_days].strftime('%Y-%m-%d').tolist(),
              'public_actual_score': model.realized_score(truth, allocation[public_days], data.capacity), 'relative_objective': relative,
              'note': 'Public scores are in-sample; full-month B score unknown. Cost offsets calibrate neural predictions, not the official score formula.'}
    (output / 'B_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train','predict'], nargs='?', default='predict')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1])
    parser.add_argument('--public-dir', type=Path, default=ROOT.parents[1] / 'open_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results_v2')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data_b = model.base.read_data(args.data_dir, 'B')
    if args.action == 'train':
        public_days, _, _ = model.transfer.read_public(args.public_dir, data_b)
        train(model.base.read_data(args.data_dir, 'A'), public_days, args.output_dir)
    export(data_b, args.public_dir, args.output_dir)


if __name__ == '__main__':
    main()
