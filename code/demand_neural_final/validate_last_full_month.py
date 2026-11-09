"""Spatial cross-city simulation on the full month, with fixed model settings."""
import argparse
import hashlib
import json
from pathlib import Path
import run_last_submission as run
import numpy as np
import pandas as pd
import torch
from neural_allocation import optimize,realized_score

ROOT=Path(__file__).resolve().parent


def neural_fold(data,public,sites,source_sites,saved,output,number):
    current=run.consensus.physical.baseline
    signature=current.core.base.neural
    days=np.arange(len(data.city.dates))
    trained=np.asarray(saved['prior']['public_days'])
    cfg={k:saved[k]['config'] for k in ('prior','signature')}
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()+data.city.net_flow[:,source_sites].numpy().tobytes()+data.city.net_flow[public][:,sites].numpy().tobytes()).hexdigest()
    file=output/f'A_full_neural_fold{number}.npz'
    if file.exists():
        with np.load(file) as z:
            if str(z['cache_key'])==fingerprint:return z['increments'].copy()
    target=signature.transfer.increments(data.city.net_flow.numpy(),data.capacity)
    prior_models=[signature.inventory.fit(data,target,days,source_sites,seed,epoch)[0]
                  for seed,epoch in zip(signature.inventory.SEEDS,cfg['prior']['epochs'])]
    inputs=signature.make_inputs(data,data.city.net_flow[trained].numpy())
    signature_models=[signature.fit(inputs,target,np.setdiff1d(days,trained),source_sites,seed,epoch)[0]
                      for seed,epoch in zip(signature.SEEDS,cfg['signature']['epochs'])]
    prior={**saved['prior'],'states':[m.state_dict() for m in prior_models]}
    public_signature={**saved['signature'],'states':[m.state_dict() for m in signature_models]}
    increments=current.infer_daytypes(data,prior,public_signature,public,data.city.net_flow[public].numpy(),days,sites)
    np.savez_compressed(file,increments=increments,cache_key=fingerprint)
    print(f'Full-month neural fold {number+1}/5 fitted with fixed epochs',flush=True)
    return increments


def validate(data,public,neural_dir,output):
    saved=torch.load(neural_dir/'public_update_model.pt',map_location='cpu',weights_only=True)
    previous=json.loads((neural_dir/'A_validation.json').read_text(encoding='utf-8'))
    days=np.arange(len(data.city.dates));unknown=np.setdiff1d(days,public)
    combined={label:np.empty((len(days),len(data.capacity)),np.int64) for label in ('neural','scenarios','consensus','unrestricted')}
    records=[]
    for number,fold in enumerate(previous['folds']):
        sites=data.city.stations.get_indexer(fold['test_station_ids'])
        source_sites=np.setdiff1d(np.arange(len(data.capacity)),sites)
        caps=data.capacity[sites]
        increments=neural_fold(data,public,sites,source_sites,saved,output,number)
        neural=run.consensus.physical.baseline.core.base.neural.inventory.cost_table(increments,caps)
        # Increments retain the common A level axis. Trim to this city's docks.
        neural=neural[:,:,:int(caps.max())+1]
        source=run.consensus.physical.reference(data,source_sites)
        counts=data.counts[public][:,sites].numpy()
        truth=data.city.net_flow[:,sites].numpy()
        paths,weights=run.consensus.physical.scenarios(source,data,sites,public,counts)
        x,_=optimize(neural,caps)
        totals=run.consensus.physical.observed_metrics(truth,x,caps)
        rates,_,_=run.consensus.physical.fit_rates(paths,weights,x,caps,totals,10.)
        raw=run.consensus.physical.costs(paths,weights,caps,np.ones((len(days),2)))
        adjusted=run.consensus.physical.costs(paths,weights,caps,rates)
        scenarios=run.consensus.physical.exact_public(run.consensus.physical.blend(raw,adjusted,.5),public,truth[public],caps)
        y,_=optimize(scenarios,caps)
        scores={};s0=realized_score(truth,x,caps);s1=realized_score(truth,y,caps)
        allocation,_,fit=run.consensus.solve(neural,scenarios,x,y,s0['score'],s1['score'],caps,restrict=True,known_days=public)
        free,_,_=run.consensus.solve(neural,scenarios,x,y,s0['score'],s1['score'],caps,restrict=False,known_days=public)
        for label,value in [('neural',x),('scenarios',y),('consensus',allocation),('unrestricted',free)]:
            combined[label][:,sites]=value
            scores[label]={'all_30_days':realized_score(truth,value,caps),
                           'unrevealed_25_days':realized_score(truth[unknown],value[unknown],caps)}
        chosen='neural' if s0['score']<=s1['score'] else 'scenarios'
        record={'fold':number,'test_station_ids':fold['test_station_ids'],'calibration':fit,'scores':scores,
                'best_known_reference':chosen,'change_all_days_from_best_known':scores['consensus']['all_30_days']['score']-scores[chosen]['all_30_days']['score'],
                'change_unrevealed_days_from_best_known':scores['consensus']['unrevealed_25_days']['score']-scores[chosen]['unrevealed_25_days']['score']}
        records.append(record)
        run.dump(output/'A_full_month_progress.json',records)
        print(json.dumps({'full_month_fold':number,'scores':scores,'weight':fit['scenario_weight']}),flush=True)
    result={'protocol':'Five held-out station folds simulate target B using the full month. Neural parameters are fitted on other stations only, with original fixed epochs/configs; target stations supply five public dates. Both proposals are optimized on all 30 dates. Only aggregates at the first proposal and the two proposal scores are exposed for calibration. Evaluate the new proposal on all 30 dates and separately on the 25 unrevealed dates. The new proposal labels are not used to select weights, epochs, constraints, or settings. Source A data were reused in earlier development, so this is not an untouched final test.',
            'neural_fit_epochs':{name:saved[name]['config']['epochs'] for name in ('prior','signature')},
            'neural_source_dates':data.city.dates.strftime('%Y-%m-%d').tolist(),
            'unrevealed_evaluation_dates':data.city.dates[unknown].strftime('%Y-%m-%d').tolist(),
            'public_days':public.tolist(),'primary_policy':'reference_range_restriction_except_exact_public_days',
            'scores':{label:{'all_30_days':realized_score(data.city.net_flow.numpy(),x,data.capacity),
                             'unrevealed_25_days':realized_score(data.city.net_flow[unknown].numpy(),x[unknown],data.capacity)} for label,x in combined.items()},
            'folds':records,'improved_all_days_fold_count':sum(r['change_all_days_from_best_known']<0 for r in records),
            'improved_unrevealed_days_fold_count':sum(r['change_unrevealed_days_from_best_known']<0 for r in records)}
    for label,x in combined.items():
        pd.DataFrame(x.T,index=data.city.stations,columns=[f'D{i:02d}' for i in range(1,31)]).rename_axis('station_id').to_csv(output/f'A_full_{label}.csv',encoding='utf-8-sig')
    run.dump(output/'A_full_month_validation.json',result)
    print(json.dumps({'full_month_scores':result['scores'],'improved_all_days_folds':result['improved_all_days_fold_count'],
                      'improved_unrevealed_folds':result['improved_unrevealed_days_fold_count']}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT.parents[1])
    p.add_argument('--public-dir',type=Path,default=ROOT.parents[1]/'open_data')
    p.add_argument('--first-dir',type=Path,default=ROOT/'results_public_d13')
    p.add_argument('--output-dir',type=Path,default=run.OUT)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    load=run.consensus.physical.baseline.core.base.neural.base.read_data
    a=load(args.data_dir,'A');b=load(args.data_dir,'B')
    public,_,_=run.consensus.physical.baseline.core.base.neural.transfer.read_public(args.public_dir,b)
    validate(a,public,args.first_dir,args.output_dir)


if __name__=='__main__':main()
