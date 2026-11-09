"""Final one-shot candidate using score-calibrated neural/scenario consensus."""
import argparse
import hashlib
import json
from pathlib import Path
import bike_score_consensus as consensus
import numpy as np
import pandas as pd
import torch
from neural_allocation import optimize,realized_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_last'
SUBMISSION='최종제출_B지역_마지막.csv'
FIRST='최종제출_배치표_B지역_D13주말추가.csv'
SECOND='최종제출_배치표_B지역_피드백개선.csv'
# User-confirmed 427-bike result. The first full-precision score is recoverable
# from the supplied screenshots' integer daily totals.
SECOND_SCORE=118429.0


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def validate_a(data,public,neural_dir,output):
    report=json.loads((neural_dir/'A_validation.json').read_text(encoding='utf-8'))
    with np.load(neural_dir/'A_updated_costs.npz') as z:
        days=z['days'];increments=z['increments']
    assert not np.intersect1d(days,public).size
    c0=consensus.physical.baseline.core.base.neural.inventory.cost_table(increments,data.capacity)
    records=[];combined={label:np.empty((len(days),len(data.capacity)),dtype=np.int64) for label in ('neural','scenarios','consensus','unrestricted')}
    for number,fold in enumerate(report['folds']):
        sites=data.city.stations.get_indexer(fold['test_station_ids'])
        source_sites=np.setdiff1d(np.arange(len(data.capacity)),sites)
        source=consensus.physical.reference(data,source_sites)
        caps=data.capacity[sites]
        target=data.city.net_flow[days][:,sites].numpy()
        paths,weights=consensus.physical.scenarios(source,data,sites,public,data.counts[public][:,sites].numpy())
        paths,weights=paths[days],weights[days]
        neural=c0[:,sites,:int(caps.max())+1]
        x,_=optimize(neural,caps)
        # The same already-supplied information channels are simulated in A:
        # city totals at the first allocation, followed by two scalar scores.
        feedback=consensus.physical.observed_metrics(target,x,caps)
        rates,_,_=consensus.physical.fit_rates(paths,weights,x,caps,feedback,10.)
        raw=consensus.physical.costs(paths,weights,caps,np.ones((len(days),2)))
        adjusted=consensus.physical.costs(paths,weights,caps,rates)
        scenarios=consensus.physical.blend(raw,adjusted,.5)
        y,_=optimize(scenarios,caps)
        s0=realized_score(target,x,caps);s1=realized_score(target,y,caps)
        allocation,costs,fit=consensus.solve(neural,scenarios,x,y,s0['score'],s1['score'],caps,restrict=True)
        free,_,_=consensus.solve(neural,scenarios,x,y,s0['score'],s1['score'],caps,restrict=False)
        measured=realized_score(target,allocation,caps)
        free_score=realized_score(target,free,caps)
        for label,value in [('neural',x),('scenarios',y),('consensus',allocation),('unrestricted',free)]:combined[label][:,sites]=value
        record={'fold':number,'test_station_ids':fold['test_station_ids'],'neural':s0,'scenarios':s1,
                'consensus':measured,'unrestricted':free_score,'calibration':fit,
                'change_from_best_observed':measured['score']-min(s0['score'],s1['score'])}
        records.append(record)
        print(json.dumps(record),flush=True)
        dump(output/'A_validation_progress.json',records)
    all_scores={label:realized_score(data.city.net_flow[days].numpy(),x,data.capacity) for label,x in combined.items()}
    for label,x in combined.items():
        pd.DataFrame(x.T,index=data.city.stations,columns=[f'D{d+1:02d}' for d in days]).rename_axis('station_id').to_csv(output/f'A_{label}.csv',encoding='utf-8-sig')
    result={'protocol':'Five held-out station folds. Neural costs are the existing out-of-fold D25-D30 forecasts. Scenario source excludes each target fold; target context consists of the five public dates and aggregate feedback at the neural proposal. Only the two proposal scores calibrate the ensemble. No new allocation labels enter its prediction. This is feedback-conditioned allocation, not blind forecasting. Existing A data have been used repeatedly during development.',
            'evaluation_dates':data.city.dates[days].strftime('%Y-%m-%d').tolist(),'public_days':public.tolist(),
            'fixed_scenario_settings':{'regularization':10.,'blend':.5},'primary_policy':'reference_range_restriction',
            'scores':all_scores,'folds':records,'best_reference_per_fold_sum':sum(min(r['neural']['score'],r['scenarios']['score']) for r in records)}
    dump(output/'A_validation.json',result)
    print(json.dumps({'A_scores':all_scores,'best_reference_sum':result['best_reference_per_fold_sum']}),flush=True)
    return result


def input_key(data):
    h=hashlib.sha256()
    for value in (data.capacity,data.similarity,data.day_weather,data.city.temporal.numpy()):h.update(value.tobytes())
    h.update('|'.join(data.city.stations).encode())
    h.update('|'.join(data.city.dates.strftime('%Y-%m-%d')).encode())
    return h.hexdigest()


def export_b(data,public,truth,provenance,first_dir,second_dir,output,reuse_model=False):
    import run_feedback_refinement as previous
    first_score=float(np.asarray(previous.SCREENSHOT_ROWS)[:,-1].mean())
    model_file=output/'consensus_model.npz'
    if reuse_model:
        with np.load(model_file,allow_pickle=False) as z:
            if str(z['model_type'])!='score_calibrated_consensus_v1' or str(z['input_key'])!=input_key(data):raise ValueError('Model/B feature mismatch; rebuild using matching inputs')
            np.testing.assert_array_equal(z['public_days'],public);np.testing.assert_array_equal(z['public_truth'],truth)
            assert float(z['first_score'])==first_score and float(z['second_score'])==SECOND_SCORE
            neural,scenarios,x,y=[z[name].copy() for name in ('neural','scenarios','first','second')]
            first_sha,second_sha=str(z['first_sha256']),str(z['second_sha256'])
    else:
        first_file=first_dir/FIRST;second_file=second_dir/SECOND
        x=pd.read_csv(first_file,index_col='station_id').loc[data.city.stations].to_numpy().T
        y=pd.read_csv(second_file,index_col='station_id').loc[data.city.stations].to_numpy().T
        with np.load(second_dir/'B_feedback_costs.npz') as z:neural=z['baseline'];scenarios=z['costs']
        first_sha=hashlib.sha256(first_file.read_bytes()).hexdigest();second_sha=hashlib.sha256(second_file.read_bytes()).hexdigest()
    xr,_=optimize(neural,data.capacity);yr,_=optimize(scenarios,data.capacity)
    np.testing.assert_array_equal(x,xr);np.testing.assert_array_equal(y,yr)
    allocation,costs,report=consensus.solve(neural,scenarios,x,y,first_score,SECOND_SCORE,data.capacity,restrict=True,known_days=public)
    file=output/SUBMISSION
    pd.DataFrame(allocation.T,index=data.city.stations,columns=[f'D{i:02d}' for i in range(1,31)]).rename_axis('station_id').to_csv(file,encoding='utf-8-sig')
    result={'model_type':'score_calibrated_consensus_v1','N':int(allocation[0].sum()),'stations':len(data.capacity),'days':30,
            'extra_fleet_penalty':0,'official_fleet_coefficient':20,'calibration':report,'public_files':provenance,
            'known_server_scores':{'neural_463':first_score,'scenario_427':SECOND_SCORE},
            'public_actual':realized_score(truth,allocation[public],data.capacity),'full_B_actual_score':None,
            'first_csv_sha256':first_sha,'second_csv_sha256':second_sha,
            'note':'Only two observed full-month submission scores fit one ensemble weight. Projected changes are model estimates, not actual final server scores. Unrevealed per-station stocks stay between the two submitted reference stocks; exact public dates use the full known cost curves.'}
    dump(output/'B_summary.json',result)
    np.savez_compressed(model_file,neural=neural,scenarios=scenarios,first=x,second=y,
                        first_score=first_score,second_score=SECOND_SCORE,capacity=data.capacity,ids=data.city.stations.to_numpy(str),
                        public_days=public,public_truth=truth,costs=costs,input_key=input_key(data),
                        model_type='score_calibrated_consensus_v1',first_sha256=first_sha,second_sha256=second_sha)
    print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['validate','build','predict'],nargs='?',default='predict')
    p.add_argument('--data-dir',type=Path,default=ROOT.parents[1])
    p.add_argument('--public-dir',type=Path,default=ROOT.parents[1]/'open_data')
    p.add_argument('--first-dir',type=Path,default=ROOT/'results_public_d13')
    p.add_argument('--second-dir',type=Path,default=ROOT/'results_feedback')
    p.add_argument('--output-dir',type=Path,default=OUT)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    load=consensus.physical.baseline.core.base.neural.base.read_data
    b=load(args.data_dir,'B')
    public,truth,provenance=consensus.physical.baseline.core.base.neural.transfer.read_public(args.public_dir,b)
    if args.action=='validate':validate_a(load(args.data_dir,'A'),public,args.first_dir,args.output_dir)
    export_b(b,public,truth,provenance,args.first_dir,args.second_dir,args.output_dir,reuse_model=args.action=='predict')


if __name__=='__main__':main()
