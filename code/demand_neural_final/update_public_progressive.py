"""Frozen unpenalized neural ensemble with progressively revealed public days."""
import argparse
import hashlib
import json
from pathlib import Path
import update_public_ensemble as base
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
MODEL_NAME='public_update_model.pt'
SUBMISSION='최종제출_배치표_B지역_D24추가.csv'


def evaluation_days(public_days,total_days=30):
    result=np.setdiff1d(np.arange(23,total_days),np.asarray(public_days))
    if not len(result):
        raise ValueError('No unrevealed D24-D30 evaluation days remain')
    return result


def subset_truth(public_days,truth,selected):
    indices={int(d):i for i,d in enumerate(public_days)}
    if not set(np.asarray(selected).tolist()).issubset(indices):
        raise ValueError('A predecessor public day is missing')
    return truth[[indices[int(d)] for d in selected]]


def validate(data,public_days,predecessor_dir,prior_dir,signature_dir,output):
    source=torch.load(predecessor_dir/MODEL_NAME,map_location='cpu',weights_only=True)
    predecessor_public=np.array(source['observed_public_days'])
    old_report=json.loads((prior_dir/'public_transfer_validation.json').read_text(encoding='utf-8'))
    signature_report=json.loads((signature_dir/'A_validation.json').read_text(encoding='utf-8'))
    previous_report=json.loads((predecessor_dir/'A_validation.json').read_text(encoding='utf-8'))
    previous_public=data.city.dates.get_indexer(pd.to_datetime(previous_report['updated_public_dates']))
    np.testing.assert_array_equal(previous_public,predecessor_public)
    trained=np.array(source['prior']['public_days'])
    future=evaluation_days(public_days,len(data.city.dates))
    cached_dates=previous_report.get('evaluation_dates',data.city.dates[23:].strftime('%Y-%m-%d').tolist())
    cached_days=data.city.dates.get_indexer(pd.to_datetime(cached_dates))
    if (cached_days<23).any() or np.intersect1d(cached_days,predecessor_public).size:
        raise ValueError('Predecessor cache includes invalid or already revealed evaluation dates')
    with np.load(predecessor_dir/'A_updated_costs.npz',allow_pickle=False) as z:
        cached=z['increments'].copy()
    if len(cached)!=len(cached_days) or not set(future).issubset(cached_days):
        raise ValueError('Predecessor prediction cache dates do not cover remaining test dates')
    # Preserve the original seven-day inference batching. Changing its size can
    # change float32 GEMM rounding; only the scoring/optimization rows shrink.
    inference_days=np.arange(23,len(data.city.dates))
    cached_rows=cached_days-23
    evaluation_rows=future-23
    target=base.neural.transfer.increments(data.city.net_flow.numpy(),data.capacity)
    original_inputs=base.neural.make_inputs(data,data.city.net_flow[trained].numpy())
    order=np.random.default_rng(base.neural.base.SEED).permutation(len(data.city.stations))
    groups=np.split(order,[i*(len(order)//5) for i in range(1,5)])
    shape=(len(future),len(data.city.stations))
    allocated,previous_alloc=np.empty(shape,dtype=np.int64),np.empty(shape,dtype=np.int64)
    updated_predictions=np.empty(shape+(cached.shape[-1],))
    past=np.arange(23)
    records=[]
    for i,test in enumerate(groups):
        print(f'Progressive public validation fold {i+1}/5',flush=True)
        sources=np.concatenate(groups[i+1:]+groups[:i])
        old_config,new_config=old_report['folds'][i]['config'],signature_report['folds'][i]['config']
        models=[base.neural.inventory.fit(data,target,past,sources,seed,epoch)[0]
                for seed,epoch in zip(base.neural.inventory.SEEDS,old_config['epochs'])]
        prior={'model_type':'public_marginal_inventory_v1','public_days':trained.tolist(),
               'ns':data.similarity.shape[-1],'nt':base.neural.inventory.day_features(data).shape[-1],
               'config':old_config,'states':[m.state_dict() for m in models]}
        models=[base.neural.fit(original_inputs,target,np.setdiff1d(past,trained),sources,seed,epoch)[0]
                for seed,epoch in zip(base.neural.SEEDS,new_config['epochs'])]
        signature={'model_type':'public_signature_v2','public_days':trained.tolist(),
                   'ns':original_inputs.signature.shape[-1],'nt':base.neural.inventory.day_features(data).shape[-1],
                   'config':new_config,'states':[m.state_dict() for m in models]}
        original_all=base.infer(data,prior,signature,predecessor_public,data.city.net_flow[predecessor_public].numpy(),inference_days,test)
        np.testing.assert_allclose(original_all[cached_rows],cached[:,test],rtol=0,atol=1e-7)
        original=original_all[evaluation_rows]
        updated_all=base.infer(data,prior,signature,public_days,data.city.net_flow[public_days].numpy(),inference_days,test)
        updated=updated_all[evaluation_rows]
        updated_predictions[:,test]=updated
        previous_alloc[:,test],_=base.optimize(base.neural.inventory.cost_table(original,data.capacity[test]),data.capacity[test])
        allocated[:,test],_=base.optimize(base.neural.inventory.cost_table(updated,data.capacity[test]),data.capacity[test])
        truth=data.city.net_flow[future][:,test].numpy()
        record={'test_station_ids':data.city.stations[test].tolist(),
                'previous':base.realized_score(truth,previous_alloc[:,test],data.capacity[test]),
                'updated':base.realized_score(truth,allocated[:,test],data.capacity[test])}
        records.append(record)
        print(json.dumps(record['updated']),flush=True)
        (output/'validation_progress.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    actual=data.city.net_flow[future].numpy()
    previous=base.realized_score(actual,previous_alloc,data.capacity)
    updated=base.realized_score(actual,allocated,data.capacity)
    report={'protocol':'Five held-out station folds; neural models reconstructed with original D01-D23 fitting and original D09/D10 input setup. Only public observations are updated at inference. Both predecessor and updated allocations are optimized and scored on exactly the same unrevealed dates.',
            'evaluation_dates':data.city.dates[future].strftime('%Y-%m-%d').tolist(),
            'inference_dates':data.city.dates[inference_days].strftime('%Y-%m-%d').tolist(),
            'excluded_revealed_evaluation_dates':data.city.dates[np.intersect1d(np.arange(23,len(data.city.dates)),public_days)].strftime('%Y-%m-%d').tolist(),
            'trained_public_dates':data.city.dates[trained].strftime('%Y-%m-%d').tolist(),
            'previous_public_dates':data.city.dates[predecessor_public].strftime('%Y-%m-%d').tolist(),
            'updated_public_dates':data.city.dates[public_days].strftime('%Y-%m-%d').tolist(),
            'previous':previous,'updated':updated,'score_change_percent':100*(updated['score']/previous['score']-1),'folds':records,
            'note':'D24 is now public input and is excluded from test scoring. Previous D23 report covered seven test days; this comparison reoptimizes both policies on the six remaining days.'}
    (output/'A_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(allocated.T,index=data.city.stations,columns=[f'D{d+1:02d}' for d in future]).rename_axis('station_id').to_csv(output/'A_검증배치.csv',encoding='utf-8-sig')
    np.savez_compressed(output/'A_updated_costs.npz',increments=updated_predictions,days=future)
    print(json.dumps({'previous':previous,'updated':updated,'score_change_percent':report['score_change_percent']}),flush=True)


def export(data,public_dir,predecessor_dir,output):
    saved=torch.load(output/MODEL_NAME,map_location='cpu',weights_only=True)
    if saved.get('model_type')!='public_observation_update_v2':
        raise ValueError('Incorrect progressive update checkpoint')
    public_days,truth,provenance=base.neural.transfer.read_public(public_dir,data)
    if saved['observed_public_days']!=public_days.tolist():
        raise ValueError('Public-day set changed; rebuild and validate the updated observation set')
    days,sites=np.arange(len(data.city.dates)),np.arange(len(data.city.stations))
    predictions=base.infer(data,saved['prior'],saved['signature'],public_days,truth,days,sites)
    allocation,relative=base.optimize(base.neural.inventory.cost_table(predictions,data.capacity),data.capacity)
    previous_public=np.array(saved['predecessor_public_days'])
    previous_truth=subset_truth(public_days,truth,previous_public)
    previous_predictions=base.infer(data,saved['prior'],saved['signature'],previous_public,previous_truth,days,sites)
    previous_allocation,_=base.optimize(base.neural.inventory.cost_table(previous_predictions,data.capacity),data.capacity)
    previous_summary_file=predecessor_dir/'B_summary.json'
    previous_summary=json.loads(previous_summary_file.read_text(encoding='utf-8')) if previous_summary_file.is_file() else {}
    previous_file=predecessor_dir/previous_summary.get('submission_file',base.SUBMISSION)
    if previous_file.is_file():
        expected=pd.read_csv(previous_file,index_col='station_id').loc[data.city.stations].to_numpy().T
        np.testing.assert_array_equal(previous_allocation,expected)
    pd.DataFrame(allocation.T,index=data.city.stations,columns=[f'D{d+1:02d}' for d in days]).rename_axis('station_id').to_csv(output/SUBMISSION,encoding='utf-8-sig')
    previous_actual=base.realized_score(truth,previous_allocation[public_days],data.capacity)
    updated_actual=base.realized_score(truth,allocation[public_days],data.capacity)
    comparisons=[]
    for i,day in enumerate(public_days):
        comparisons.append({'date':data.city.dates[day].strftime('%Y-%m-%d'),
                            'previous_score':base.realized_score(truth[i:i+1],previous_allocation[day:day+1],data.capacity)['score'],
                            'updated_score':base.realized_score(truth[i:i+1],allocation[day:day+1],data.capacity)['score']})
    pd.DataFrame(comparisons).to_csv(output/'B_public_day_comparison.csv',index=False,encoding='utf-8-sig')
    summary={'model_type':saved['model_type'],'submission_file':SUBMISSION,'N':int(allocation[0].sum()),'baseline_N':int(previous_allocation[0].sum()),
             'stations':len(sites),'days':len(days),'weights':[.5,.5],'official_fleet_coefficient':20,'extra_fleet_penalty':0,'cost_bound':None,
             'trained_public_dates':data.city.dates[saved['prior']['public_days']].strftime('%Y-%m-%d').tolist(),
             'previous_public_dates':data.city.dates[previous_public].strftime('%Y-%m-%d').tolist(),
             'updated_public_dates':data.city.dates[public_days].strftime('%Y-%m-%d').tolist(),'public_files':provenance,
             'previous_public_actual':previous_actual,'updated_public_actual':updated_actual,
             'changed_station_days':int((allocation!=previous_allocation).sum()),'relative_objective':relative,
             'note':'Original neural weights and all mixing/calibration parameters are unchanged. Public inputs now include D24. Public scores are in-sample; the full-month B score remains unknown.'}
    (output/'B_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','predict'],nargs='?',default='predict')
    parser.add_argument('--data-dir',type=Path,default=ROOT.parents[1])
    parser.add_argument('--predecessor-dir',type=Path,default=ROOT/'results_public_d23')
    parser.add_argument('--prior-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--signature-dir',type=Path,default=ROOT/'results_v2')
    parser.add_argument('--public-dir',type=Path,default=ROOT.parents[1]/'open_data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results_public_d24')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    b=base.neural.base.read_data(args.data_dir,'B')
    if args.action=='build':
        public_days,_,_=base.neural.transfer.read_public(args.public_dir,b)
        source_file=args.predecessor_dir/MODEL_NAME
        source=torch.load(source_file,map_location='cpu',weights_only=True)
        if source.get('model_type') not in ('public_observation_update_v1','public_observation_update_v2'):
            raise ValueError('Expected an unpenalized public-observation model')
        if not set(source['observed_public_days']).issubset(public_days.tolist()):
            raise ValueError('A predecessor public day is missing')
        base.check_public_protocol(source['prior'],source['signature'],public_days)
        saved={'model_type':'public_observation_update_v2','prior':source['prior'],'signature':source['signature'],
               'predecessor_public_days':source['observed_public_days'],'observed_public_days':public_days.tolist(),
               'weights':[.5,.5],'source_model_sha256':hashlib.sha256(source_file.read_bytes()).hexdigest()}
        torch.save(saved,args.output_dir/MODEL_NAME)
        validate(base.neural.base.read_data(args.data_dir,'A'),public_days,args.predecessor_dir,args.prior_dir,args.signature_dir,args.output_dir)
    export(b,args.public_dir,args.predecessor_dir,args.output_dir)


if __name__=='__main__':
    main()
