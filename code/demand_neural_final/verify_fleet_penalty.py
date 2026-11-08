import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import bike_fleet_penalty as model
from verify_neural_submission import official_score
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent


def verify(data_dir, public_dir, signature_dir, output):
    spec = importlib.util.spec_from_file_location('official_scorer', data_dir / 'score.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    a = model.neural.base.read_data(data_dir, 'A')
    b = model.neural.base.read_data(data_dir, 'B')
    summary = json.loads((output / 'B_summary.json').read_text(encoding='utf-8'))
    validation = json.loads((output / 'A_validation.json').read_text(encoding='utf-8'))
    file = output / model.SUBMISSION
    frame = pd.read_csv(file, encoding='utf-8-sig')
    assert list(frame) == ['station_id'] + [f'D{i:02d}' for i in range(1, 31)]
    assert frame.station_id.tolist() == b.city.stations.tolist()
    allocation = frame.iloc[:, 1:].to_numpy(float).T
    assert np.isfinite(allocation).all() and np.equal(allocation, np.floor(allocation)).all()
    assert (allocation >= 0).all() and (allocation <= b.capacity).all()
    assert (allocation.sum(1) == summary['N']).all() and summary['N'] > 0
    assert summary['effective_fleet_price'] == 20 + summary['alpha']
    public_days, truth, _ = model.neural.transfer.read_public(public_dir, b)
    public_counts = np.stack([np.maximum(-truth, 0), np.maximum(truth, 0)], axis=-1)
    public_score = official_score(b, public_days, allocation[public_days], public_counts, official)
    assert abs(public_score - summary['public_actual']['score']) < 0.01
    selected_a = pd.read_csv(output / 'A_검증배치.csv', index_col='station_id').loc[a.city.stations].to_numpy().T
    days = np.arange(23, 30)
    selected_score = official_score(a, days, selected_a, a.counts[days].numpy(), official)
    assert abs(selected_score - validation['actual']['score']) < 0.01
    # Reproduce both the exported policy and the prior unpenalized allocation.
    saved = torch.load(output / model.MODEL_NAME, map_location='cpu', weights_only=True)
    costs = model.predict_costs(saved, b, public_days, truth)
    entries = model.frontier(costs, b.capacity, [0., summary['alpha']])
    old_b = pd.read_csv(signature_dir / '최종제출_배치표_B지역_개선2.csv', index_col='station_id').loc[b.city.stations].to_numpy().T
    np.testing.assert_array_equal(entries[0][1], old_b)
    np.testing.assert_array_equal(entries[1][1], allocation)
    with np.load(signature_dir / 'A_ensemble_costs.npz', allow_pickle=False) as z:
        oof = z['increments']
    fixed_a = np.empty(selected_a.shape, dtype=np.int64)
    for fold in validation['folds']:
        sites = a.city.stations.get_indexer(fold['test_station_ids'])
        [(row, x)] = model.frontier(model.neural.inventory.cost_table(oof[:, sites], a.capacity[sites]), a.capacity[sites], [summary['alpha']])
        fixed_a[:, sites] = x
    fixed_score = official_score(a, days, fixed_a, a.counts[days].numpy(), official)
    table = pd.read_csv(output / 'A_alpha_comparison.csv').set_index('alpha')
    assert abs(fixed_score - table.loc[summary['alpha'], 'score']) < 0.01
    result = {'constraints': 'PASS', 'stations': len(frame), 'days': 30, 'N': summary['N'], 'alpha': summary['alpha'],
              'baseline_N': summary['baseline_N'], 'N_reduction_percent': summary['N_reduction_percent'],
              'A_nested_policy_official_score': selected_score, 'A_fixed_selected_alpha_official_score': fixed_score,
              'A_baseline_official_score': validation['baseline_actual']['score'],
              'A_fixed_alpha_score_increase_percent': 100 * (fixed_score / validation['baseline_actual']['score'] - 1),
              'B_public_actual_score': public_score, 'full_B_actual_score': None,
              'alpha_zero_exactly_matches_previous_submission': True, 'saved_model_reproduces_final_allocation': True,
              'csv_sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    (output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1])
    parser.add_argument('--signature-dir', type=Path, default=ROOT / 'results_v2')
    parser.add_argument('--public-dir', type=Path, default=ROOT.parents[1] / 'open_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results_fleet')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verify(args.data_dir, args.public_dir, args.signature_dir, args.output_dir)
