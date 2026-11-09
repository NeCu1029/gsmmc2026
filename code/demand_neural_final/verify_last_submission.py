"""Official-score parity, saved-model replay, and decimal-feedback sensitivity."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import run_last_submission as run
from verify_neural_submission import official_score
import numpy as np
import pandas as pd
import torch
from neural_allocation import dp_reconstruct,realized_score

ROOT=Path(__file__).resolve().parent


def verify(args):
    load=run.consensus.physical.baseline.core.base.neural.base.read_data
    a,b=load(args.data_dir,'A'),load(args.data_dir,'B')
    spec=importlib.util.spec_from_file_location('official',args.data_dir/'score.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    public,truth,_=run.consensus.physical.baseline.core.base.neural.transfer.read_public(args.public_dir,b)
    file=args.output_dir/run.SUBMISSION
    frame=pd.read_csv(file,encoding='utf-8-sig')
    assert frame.columns.tolist()==['station_id']+[f'D{i:02d}' for i in range(1,31)]
    assert frame.station_id.tolist()==b.city.stations.tolist()
    x=frame.iloc[:,1:].to_numpy(float).T
    assert np.isfinite(x).all() and np.equal(x,np.floor(x)).all() and (x>=0).all() and (x<=b.capacity).all()
    assert np.all(x.sum(1)==x[0].sum()) and x[0].sum()>0
    summary=json.loads((args.output_dir/'B_summary.json').read_text(encoding='utf-8'))
    assert summary['N']==x[0].sum() and summary['extra_fleet_penalty']==0 and summary['official_fleet_coefficient']==20
    with np.load(args.output_dir/'consensus_model.npz',allow_pickle=False) as z:
        assert str(z['model_type'])=='score_calibrated_consensus_v1'
        assert str(z['input_key'])==run.input_key(b)
        np.testing.assert_array_equal(z['public_days'],public);np.testing.assert_array_equal(z['public_truth'],truth)
        c0,c1,first,second=[z[n].copy() for n in ('neural','scenarios','first','second')]
        s0,s1=float(z['first_score']),float(z['second_score'])
        replay,costs,fit=run.consensus.solve(c0,c1,first,second,s0,s1,b.capacity,restrict=True,known_days=public)
        np.testing.assert_array_equal(replay,x)
        np.testing.assert_array_equal(costs,z['costs'])
    assert s1==118429.0 and round(s0,1)==118428.6
    unknown=np.setdiff1d(np.arange(30),public)
    assert (x[unknown]>=np.minimum(first,second)[unknown]).all()
    assert (x[unknown]<=np.maximum(first,second)[unknown]).all()
    assert abs(fit['fitted_contrast']-(s1-s0))<1e-8
    equivalent=np.stack([np.maximum(-truth,0),np.maximum(truth,0)],axis=-1)
    measured=official_score(b,public,x[public],equivalent,official)
    assert abs(measured-summary['public_actual']['score'])<1e-8
    # Every exact public date gets the best allocation at the selected N over
    # all dock-feasible stocks, including stocks outside the reference range.
    exact=run.consensus.physical.profile.relative_public_costs(truth,b.capacity)
    public_optimal=np.stack([dp_reconstruct(c,b.capacity,int(x[0].sum())) for c in exact])
    np.testing.assert_allclose(realized_score(truth,public_optimal,b.capacity)['score'],measured,rtol=0,atol=1e-8)
    sensitivity=[]
    # Integer daily scores imply month-average steps of 1/30. These are the
    # only three averages that round to the reported one-decimal 118,429.0.
    for monthly_sum in (3552869,3552870,3552871):
        reported=monthly_sum/30
        out,_,settings=run.consensus.solve(c0,c1,first,second,s0,reported,b.capacity,restrict=True,known_days=public)
        sensitivity.append({'possible_actual_score':reported,'N':settings['N'],
                            'changed_cells_from_final':int(np.count_nonzero(out!=x))})
    wide=json.loads((args.output_dir/'A_full_month_validation.json').read_text(encoding='utf-8'))
    assert wide['primary_policy']=='reference_range_restriction_except_exact_public_days'
    hidden=a.city.dates.get_indexer(pd.to_datetime(wide['unrevealed_evaluation_dates']))
    assert len(hidden)==25 and not np.intersect1d(hidden,public).size
    a_scores={}
    for label in ('neural','scenarios','consensus','unrestricted'):
        alloc=pd.read_csv(args.output_dir/f'A_full_{label}.csv',index_col='station_id').loc[a.city.stations].to_numpy().T
        total=official_score(a,np.arange(30),alloc,a.counts.numpy(),official)
        unseen=official_score(a,hidden,alloc[hidden],a.counts[hidden].numpy(),official)
        assert abs(total-wide['scores'][label]['all_30_days']['score'])<.01
        assert abs(unseen-wide['scores'][label]['unrevealed_25_days']['score'])<.01
        a_scores[label]={'all_30_days':total,'unrevealed_25_days':unseen}
    result={'constraints':'PASS','N':int(x[0].sum()),'stations':159,'days':30,'extra_fleet_penalty':0,
            'official_fleet_coefficient':20,'saved_model_replay':'PASS','reference_range_on_unrevealed_days':'PASS',
            'exact_public_dates_optimized_over_all_dock_levels':'PASS','B_public_actual_official':measured,
            'known_scores':{'first_463':s0,'second_427':s1},'scenario_weight':fit['scenario_weight'],
            'decimal_score_sensitivity':sensitivity,'A_full_month_official':a_scores,
            'A_unrevealed_change_percent':100*(a_scores['consensus']['unrevealed_25_days']/a_scores['neural']['unrevealed_25_days']-1),
            'new_full_B_actual_score':None,'csv_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    run.dump(args.output_dir/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT.parents[1])
    p.add_argument('--public-dir',type=Path,default=ROOT.parents[1]/'open_data')
    p.add_argument('--output-dir',type=Path,default=run.OUT)
    args=p.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    verify(args)


if __name__=='__main__':main()
