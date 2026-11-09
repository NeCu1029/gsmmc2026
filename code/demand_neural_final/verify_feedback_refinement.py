"""Independent scorer checks for the delivered feedback-conditioned allocation."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import run_feedback_refinement as run
from verify_neural_submission import official_score
import numpy as np
import pandas as pd
import torch
from neural_allocation import optimize

ROOT=Path(__file__).resolve().parent


def verify(args):
    spec=importlib.util.spec_from_file_location('official',args.data_dir/'score.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    load=run.model.baseline.core.base.neural.base.read_data
    a,b=load(args.data_dir,'A'),load(args.data_dir,'B')
    summary=json.loads((args.output_dir/'B_summary.json').read_text(encoding='utf-8'))
    file=args.output_dir/run.SUBMISSION
    frame=pd.read_csv(file,encoding='utf-8-sig')
    assert frame.columns.tolist()==['station_id']+[f'D{i:02d}' for i in range(1,31)]
    assert frame.station_id.tolist()==b.city.stations.tolist()
    x=frame.iloc[:,1:].to_numpy(float).T
    assert np.isfinite(x).all() and (x==np.floor(x)).all() and (x>=0).all() and (x<=b.capacity).all()
    assert np.all(x.sum(1)==summary['N']) and summary['N']>0
    public,truth,_=run.model.baseline.core.base.neural.transfer.read_public(args.public_dir,b)
    counts=run.model.read_public_counts(args.public_dir,b,public)
    measured=official_score(b,public,x[public],counts,official)
    assert abs(measured-summary['public_actual']['score'])<.01
    saved=torch.load(args.output_dir/run.MODEL,map_location='cpu',weights_only=True)
    feedback=run.screenshot_feedback()
    np.testing.assert_array_equal(run.model.observed_metrics(truth,saved['baseline_x'].numpy()[public],b.capacity),feedback[public])
    with np.load(args.output_dir/'B_feedback_costs.npz') as z:
        regenerated,_=optimize(z['costs'],b.capacity)
        np.testing.assert_array_equal(regenerated,x)
        exact=run.model.profile.relative_public_costs(truth,b.capacity)
        mask=np.isfinite(z['costs'][public])
        np.testing.assert_allclose(z['costs'][public][mask],exact[mask],rtol=0,atol=1e-7)
        assert (z['rates']>=.625).all() and (z['rates']<=1.6).all()
    report=json.loads((args.output_dir/'A_feedback_validation.json').read_text(encoding='utf-8'))
    last=report['folds'][-1]
    sites=a.city.stations.get_indexer(last['stations'])
    dates=a.city.dates.get_indexer(pd.to_datetime(report['evaluation_dates']))
    assert not np.intersect1d(dates,public).size
    subset=copy.copy(a);subset.city=copy.copy(a.city)
    subset.capacity=a.capacity[sites];subset.city.stations=a.city.stations[sites]
    scores={}
    for label in ('baseline','refined'):
        ax=pd.read_csv(args.output_dir/f'A_holdout_{label}.csv',index_col='station_id').loc[subset.city.stations].to_numpy().T
        scores[label]=official_score(subset,dates,ax,a.counts[dates][:,sites].numpy(),official)
    assert abs(scores['baseline']-last['baseline']['score'])<.01
    assert abs(scores['refined']-last['candidates'][0]['actual']['score'])<.01
    assert saved['config']==report['selected_config']==summary['config']
    result={'constraints':'PASS','N':summary['N'],'official_fleet_coefficient':20,'extra_fleet_penalty':0,
            'public_actual_official':measured,'A_unseen_station_test_official':scores,
            'A_test_station_count':len(sites),'A_test_day_count':len(dates),
            'A_test_change_percent':100*(scores['refined']/scores['baseline']-1),
            'public_costs_exact':True,'screenshot_feedback_consistent_with_original_submission':True,
            'integer_DP_reproduces_final_file':True,'full_B_new_score':None,
            'csv_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    run.dump(args.output_dir/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT.parents[1])
    p.add_argument('--public-dir',type=Path,default=ROOT.parents[1]/'open_data')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results_feedback')
    args=p.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    verify(args)
