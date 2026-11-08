import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import bike_bounded_fleet as model
from verify_neural_submission import official_score
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent


def verify(data_dir, public_dir, signature_dir, output):
    spec = importlib.util.spec_from_file_location('official_scorer', data_dir / 'score.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    a = model.networks.neural.base.read_data(data_dir, 'A')
    b = model.networks.neural.base.read_data(data_dir, 'B')
    summary = json.loads((output / 'B_summary.json').read_text(encoding='utf-8'))
    report = json.loads((output / 'A_validation.json').read_text(encoding='utf-8'))
    file = output / model.SUBMISSION
    frame = pd.read_csv(file, encoding='utf-8-sig')
    assert list(frame) == ['station_id'] + [f'D{i:02d}' for i in range(1, 31)]
    assert frame.station_id.tolist() == b.city.stations.tolist()
    x = frame.iloc[:, 1:].to_numpy(float).T
    assert np.isfinite(x).all() and (x == np.floor(x)).all() and (x >= 0).all() and (x <= b.capacity).all()
    assert (x.sum(1) == summary['N']).all() and summary['N'] >= 1
    public_days, truth, _ = model.networks.neural.transfer.read_public(public_dir, b)
    counts = np.stack([np.maximum(-truth, 0), np.maximum(truth, 0)], axis=-1)
    public = official_score(b, public_days, x[public_days], counts, official)
    assert abs(public - summary['public_actual']['score']) < 0.01
    assert public <= summary['public_reference_score'] + summary['allowed_daily_score_increase'] + 1e-8
    selected_a = pd.read_csv(output / 'A_검증배치.csv', index_col='station_id').loc[a.city.stations].to_numpy().T
    a_score = official_score(a, np.arange(23,30), selected_a, a.counts[23:].numpy(), official)
    assert abs(a_score - report['actual']['score']) < .01
    saved = torch.load(output / model.MODEL_NAME, map_location='cpu', weights_only=True)
    costs = model.networks.predict_costs(saved['networks'], b, public_days, truth)
    prepared = model.prepare(costs, b.capacity, truth)
    selected, regenerated = model.select(prepared, summary['tolerance'])
    _, baseline = model.select(prepared, 0.)
    old = pd.read_csv(signature_dir / '최종제출_배치표_B지역_개선2.csv', index_col='station_id').loc[b.city.stations].to_numpy().T
    np.testing.assert_array_equal(old, baseline)
    np.testing.assert_array_equal(regenerated, x)
    n = selected['N']
    predicted_increases = prepared['predicted'] - prepared['predicted_baseline_relative']
    public_increases = prepared['public'] - prepared['public_reference_score']
    budget = selected['allowed_daily_score_increase']
    assert all(predicted_increases[k] > budget or public_increases[k] > budget for k in range(1,n))
    result = {'constraints':'PASS','stations':len(frame),'days':30,'N':n,'baseline_N':selected['baseline_N'],
              'N_reduction_percent':selected['N_reduction_percent'],'tolerance':summary['tolerance'],
              'budget_reference':'Observed public-day score at the original 515-bike baseline; unknown full-month score is not guaranteed.',
              'budget':budget,'predicted_increase':selected['predicted_daily_score_increase'],
              'public_reference':selected['public_reference_score'],'public_score':public,
              'public_increase_percent':100*(public/selected['public_reference_score']-1),
              'A_baseline_score':report['baseline_actual']['score'],'A_actual_score':a_score,
              'A_actual_increase_percent':report['actual_A_score_increase_percent'],
              'all_smaller_positive_N_violate_a_bound':True,'zero_tolerance_reproduces_previous_csv':True,
              'saved_model_reproduces_final_csv':True,'csv_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    (output / 'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT.parents[1])
    parser.add_argument('--signature-dir',type=Path,default=ROOT / 'results_v2')
    parser.add_argument('--public-dir',type=Path,default=ROOT.parents[1] / 'open_data')
    parser.add_argument('--output-dir',type=Path,default=ROOT / 'results_bounded')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verify(args.data_dir,args.public_dir,args.signature_dir,args.output_dir)
