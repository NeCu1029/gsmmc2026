"""Official-score and submission checks for the public-signature model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import bike_signature_net as model
from verify_neural_submission import official_score
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def verify(data_dir, public_dir, output, previous):
    spec = importlib.util.spec_from_file_location('official_scorer', data_dir / 'score.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    a, b = model.base.read_data(data_dir, 'A'), model.base.read_data(data_dir, 'B')
    csv = output / model.SUBMISSION
    frame = pd.read_csv(csv, encoding='utf-8-sig')
    assert list(frame) == ['station_id'] + [f'D{i:02d}' for i in range(1, 31)]
    assert frame.station_id.tolist() == b.city.stations.tolist()
    values = frame.iloc[:, 1:].to_numpy(float).T
    assert np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
    assert (values >= 0).all() and (values <= b.capacity).all()
    assert (values.sum(1) == values.sum(1)[0]).all() and values.sum(1)[0] > 0
    days = np.arange(23, 30)
    summary = json.loads((output / 'B_summary.json').read_text(encoding='utf-8'))
    kind = summary['model_type']
    names = {'public_signature_v2': ('A_검증배치.csv', 'A_validation.json'),
             'public_signature_calibrated_v2': ('A_보정검증배치.csv', 'A_calibrated_validation.json'),
             'equal_public_signature_ensemble_v2': ('A_앙상블검증배치.csv', 'A_ensemble_validation.json')}
    allocation_file, report_file = names[kind]
    val = pd.read_csv(output / allocation_file, index_col='station_id').loc[a.city.stations].to_numpy().T
    score = official_score(a, days, val, a.counts[days].numpy(), official)
    report = json.loads((output / report_file).read_text(encoding='utf-8'))
    assert abs(score - report['actual_score']['score']) < 0.01
    public_days, public_truth, _ = model.transfer.read_public(public_dir, b)
    # Equivalent counts are valid here because the official recurrence uses net flow.
    public_counts = np.stack([np.maximum(-public_truth, 0), np.maximum(public_truth, 0)], axis=-1)
    public_score = official_score(b, public_days, values[public_days], public_counts, official)
    assert abs(public_score - summary['public_actual_score']['score']) < 0.01
    result = {'constraints': 'PASS', 'stations': len(frame), 'days': 30, 'N': int(values.sum(1)[0]),
              'A_official_score': score, 'B_public_in_sample_score': public_score,
              'csv_sha256': hashlib.sha256(csv.read_bytes()).hexdigest(),
              'full_B_new_score': None, 'previous_full_B_server_score': 119958.7}
    if previous is not None:
        prior = pd.read_csv(previous / 'A_공개일활용_검증배치.csv', index_col='station_id').loc[a.city.stations].to_numpy().T
        old = official_score(a, days, prior, a.counts[days].numpy(), official)
        result.update({'A_previous_official_score': old, 'A_reduction_percent': 100 * (old - score) / old})
        daily = []
        for i, d in enumerate(days):
            actual = a.city.net_flow[d:d+1].numpy()
            daily.append({'date': a.city.dates[d].strftime('%Y-%m-%d'), 'off_day': bool(a.city.off_day[d]),
                          'previous_score': model.realized_score(actual, prior[i:i+1], a.capacity)['score'],
                          'new_score': model.realized_score(actual, val[i:i+1], a.capacity)['score']})
        result['A_daily_comparison'] = daily
        old_b = pd.read_csv(previous / '최종제출_배치표_B지역.csv', index_col='station_id').loc[b.city.stations].to_numpy().T
        result['B_changed_station_days'] = int((values != old_b).sum())
        # Confirm that the locally evaluated prior submission is the screenshot's one.
        expected = [176570.0, 154854.0]
        if public_days.tolist() == [8, 9]:
            actual = [model.realized_score(public_truth[i:i+1], old_b[d:d+1], b.capacity)['score'] for i, d in enumerate(public_days)]
            assert actual == expected, actual
            result['screenshot_public_scores_reproduced'] = True
    (output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parents[1])
    parser.add_argument('--public-dir', type=Path, default=ROOT.parents[1] / 'open_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results_v2')
    parser.add_argument('--previous-dir', type=Path)
    args = parser.parse_args()
    verify(args.data_dir, args.public_dir, args.output_dir, args.previous_dir)
