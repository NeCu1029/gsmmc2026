"""Use existing five public dates and the supplied submission's daily feedback."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import bike_feedback_calibration as model
import numpy as np
import pandas as pd
import torch
from neural_allocation import optimize,realized_score

ROOT=Path(__file__).resolve().parent
SUBMISSION='최종제출_배치표_B지역_피드백개선.csv'
MODEL='feedback_refined_model.pt'
LAMBDA=(.1,1.,10.)
WEIGHTS=(.5,1.)

# User-provided screenshots for submission #26, N=463. These are aggregate
# observations, not station-level target labels. No leaderboard target N is used.
SCREENSHOT_ROWS=[
 (2169,19943,1405,4563,140506),(1911,19127,1364,4837,135679),
 (882,3420,610,3622,36452),(1554,13214,1191,5139,99638),
 (2195,25929,1482,4686,176930),(1149,5465,766,3615,49495),
 (989,4103,546,3127,39735),(2197,26189,1494,4949,178813),
 (2143,25669,1692,4996,176730),(1953,22327,1403,4847,155084),
 (1946,19142,1395,4837,135924),(1966,25816,1319,4859,175610),
 (803,3595,522,2971,36411),(764,2560,474,3682,30672),
 (1983,18959,1378,4719,134623),(2019,24163,1403,4814,166067),
 (1868,21988,1460,5019,153507),(1956,22612,1408,4919,156891),
 (1105,8671,916,5020,70886),(296,800,133,2742,17467),
 (1037,3915,616,3475,39305),(2163,22219,1604,5476,156070),
 (2243,27025,1684,4702,184532),(2248,26316,1666,4905,180391),
 (2105,26167,1523,5017,178894),(2186,19450,1549,4799,138504),
 (1035,4521,619,3279,42760),(981,3669,541,3306,37285),
 (2255,24053,1590,4636,166164),(2153,23263,1588,5054,161832)]


def screenshot_feedback():
    row=np.asarray(SCREENSHOT_ROWS,float)
    np.testing.assert_array_equal(20*463+6*row[:,1]+5*row[:,2]+row[:,3],row[:,4])
    np.testing.assert_array_equal(row[:,:4].sum(0),[50254,494290,35341,132612])
    assert round(float(row[:,4].mean()),1)==118428.6
    return row[:,:4]


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def validate_on_a(a,public,output):
    order=np.random.default_rng(20261110).permutation(len(a.capacity))
    groups=np.array_split(order,3)
    unknown=np.setdiff1d(np.arange(30),public)
    records=[];chosen=None
    for fold,sites in enumerate(groups):
        # The final test stations are absent from tuning source labels as well
        # as tuning target labels. They first enter as public context at fold 2.
        source_sites=groups[1-fold] if fold<2 else np.concatenate(groups[:2])
        source=model.reference(a,source_sites)
        caps=a.capacity[sites]
        counts=a.counts[public][:,sites].numpy()
        truth=a.city.net_flow[public][:,sites].numpy()
        paths,weights=model.scenarios(source,a,sites,public,counts)
        raw=model.costs(paths,weights,caps,np.ones((30,2)))
        raw=model.exact_public(raw,public,truth,caps)
        reference_alloc,_=optimize(raw,caps)
        feedback=model.observed_metrics(a.city.net_flow[:,sites].numpy(),reference_alloc,caps)
        actual=a.city.net_flow[unknown][:,sites].numpy()
        baseline=realized_score(actual,reference_alloc[unknown],caps)
        table=[]
        # Freeze the configuration before the final station group is revealed.
        configs=[chosen] if fold==2 else [{'regularization':l,'blend':w} for l in LAMBDA for w in WEIGHTS]
        for cfg in configs:
            rates,before,after=model.fit_rates(paths,weights,reference_alloc,caps,feedback,cfg['regularization'])
            corrected=model.costs(paths,weights,caps,rates)
            corrected=model.exact_public(corrected,public,truth,caps)
            allocated,_=optimize(model.blend(raw,corrected,cfg['blend']),caps)
            measured=realized_score(actual,allocated[unknown],caps)
            table.append({'config':cfg,'actual':measured})
            if fold==2:
                pd.DataFrame(reference_alloc[unknown].T,index=a.city.stations[sites],columns=[f'D{d+1:02d}' for d in unknown]).rename_axis('station_id').to_csv(output/'A_holdout_baseline.csv',encoding='utf-8-sig')
                pd.DataFrame(allocated[unknown].T,index=a.city.stations[sites],columns=[f'D{d+1:02d}' for d in unknown]).rename_axis('station_id').to_csv(output/'A_holdout_refined.csv',encoding='utf-8-sig')
        records.append({'fold':fold,'role':'unseen_station_test' if fold==2 else 'configuration_tuning',
                        'stations':a.city.stations[sites].tolist(),'baseline':baseline,'candidates':table})
        if fold==1:
            chosen=min(configs,key=lambda cfg:sum(next(r['actual']['score'] for r in fold_result['candidates'] if r['config']==cfg) for fold_result in records))
        dump(output/'A_feedback_validation.json',{'protocol':'Two station groups tune aggregate-calibration strength; configuration is frozen before the third station group. Source stations provide all A dates, matching the available complete source month. Target stations reveal only five public dates and daily U/Q/O/E at the reference allocation. Evaluation covers the other 25 dates. This validates feedback-conditioned allocation, not prediction without aggregate feedback.',
             'evaluation_dates':a.city.dates[unknown].strftime('%Y-%m-%d').tolist(),'selected_config':chosen,'folds':records})
        print(json.dumps({'A_fold':fold,'baseline':baseline,'candidates':table,'selected':chosen}),flush=True)
    return chosen,records


def validate_public_context(b,public,counts,source,baseline_x,config,output):
    """Hide station labels, retaining the already supplied aggregate feedback.

    The reference allocation and its totals are observed inputs for both
    candidates. This is deliberately distinct from an ordinary blind LOO test.
    No setting is selected or changed by these results.
    """
    feedback=screenshot_feedback();records=[]
    for hidden in range(len(public)):
        keep=np.delete(np.arange(len(public)),hidden)
        used=public[keep];revealed=counts[keep]
        truth=revealed[...,1]-revealed[...,0]
        paths,weights=model.scenarios(source,b,np.arange(len(b.capacity)),used,revealed)
        raw=model.exact_public(model.costs(paths,weights,b.capacity,np.ones((30,2))),used,truth,b.capacity)
        rates,_,_=model.fit_rates(paths,weights,baseline_x,b.capacity,feedback,config['regularization'])
        corrected=model.exact_public(model.costs(paths,weights,b.capacity,rates),used,truth,b.capacity)
        target=counts[hidden:hidden+1,...,1]-counts[hidden:hidden+1,...,0]
        day=int(public[hidden]);scores={}
        for label,cost in [('uncalibrated',raw),('refined',model.blend(raw,corrected,config['blend']))]:
            x,_=optimize(cost,b.capacity)
            scores[label]=realized_score(target,x[day:day+1],b.capacity)
        records.append({'day':day,'date':str(b.city.dates[day].date()),'scores':scores})
        print(json.dumps({'B_hidden_day':day+1,'scores':scores}),flush=True)
    def mean(label):
        return sum(float(np.mean(b.city.off_day==kind))*np.mean([r['scores'][label]['score'] for r in records if bool(b.city.off_day[r['day']])==kind]) for kind in (False,True))
    report={'protocol':'Hide one B public date of station-level counts. Retain the existing submitted allocation and all 30 dates of its screenshot U/Q/O/E feedback. A-only configuration is frozen. This evaluates feedback-conditioned allocation; it is not blind demand forecasting or an estimate of the full-month score.',
            'config':config,'calendar_weighted_scores':{label:mean(label) for label in ('uncalibrated','refined')},'days':records,
            'settings_changed_using_this_test':False}
    dump(output/'B_feedback_context_validation.json',report)
    return report


def export(b,public,truth,public_counts,provenance,source,baseline_saved,baseline_x,config,output):
    feedback=screenshot_feedback()
    measured_public=model.observed_metrics(truth,baseline_x[public],b.capacity)
    np.testing.assert_allclose(measured_public,feedback[public],rtol=0,atol=1e-7)
    paths,weights=model.scenarios(source,b,np.arange(len(b.capacity)),public,public_counts)
    rates,before,after=model.fit_rates(paths,weights,baseline_x,b.capacity,feedback,config['regularization'])
    corrected=model.costs(paths,weights,b.capacity,rates)
    corrected=model.exact_public(corrected,public,truth,b.capacity)
    inc=model.baseline.infer_daytypes(b,baseline_saved['prior'],baseline_saved['signature'],public,truth,np.arange(30),np.arange(len(b.capacity)))
    neural=model.baseline.core.base.neural.inventory.cost_table(inc,b.capacity)
    reproduced,_=optimize(neural,b.capacity)
    np.testing.assert_array_equal(reproduced,baseline_x)
    raw=model.exact_public(model.costs(paths,weights,b.capacity,np.ones((30,2))),public,truth,b.capacity)
    costs=model.blend(raw,corrected,config['blend'])
    allocation,relative=optimize(costs,b.capacity)
    pd.DataFrame(allocation.T,index=b.city.stations,columns=[f'D{d+1:02d}' for d in range(30)]).rename_axis('station_id').to_csv(output/SUBMISSION,encoding='utf-8-sig')
    np.savez_compressed(output/'B_feedback_costs.npz',costs=costs,baseline=neural,rates=rates,before=before,after=after,feedback=feedback)
    report={'N':int(allocation[0].sum()),'baseline_N':463,'stations':len(b.capacity),'days':30,'model_type':'aggregate_feedback_scenarios_v1','config':config,
            'official_fleet_coefficient':20,'extra_fleet_penalty':0,'public_files':provenance,
            'public_actual':realized_score(truth,allocation[public],b.capacity),'baseline_public_actual':realized_score(truth,baseline_x[public],b.capacity),
            'screenshot_baseline_score':118428.6,'full_B_new_score':None,'relative_objective':relative,
            'daily_rates':rates.tolist(),'feedback_relative_log_rmse_before':float(np.sqrt(np.mean(np.log((before+1)/(feedback+1))**2))),
            'feedback_relative_log_rmse_after':float(np.sqrt(np.mean(np.log((after+1)/(feedback+1))**2))),
            'note':'Daily aggregates at the previous allocation are supplied screenshot observations, used for model calibration. They do not identify station truth. No full-month actual score is available for the new allocation.'}
    dump(output/'B_summary.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','predict'],nargs='?',default='predict')
    p.add_argument('--data-dir',type=Path,default=ROOT.parents[1])
    p.add_argument('--public-dir',type=Path,default=ROOT.parents[1]/'open_data')
    p.add_argument('--baseline-dir',type=Path,default=ROOT/'results_public_d13')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results_feedback')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    b=model.baseline.core.base.neural.base.read_data(args.data_dir,'B')
    public,truth,provenance=model.baseline.core.base.neural.transfer.read_public(args.public_dir,b)
    counts=model.read_public_counts(args.public_dir,b,public)
    if args.action=='train':
        a=model.baseline.core.base.neural.base.read_data(args.data_dir,'A')
        config,records=validate_on_a(a,public,args.output_dir)
        source=model.reference(a,np.arange(len(a.capacity)))
        baseline_saved=torch.load(args.baseline_dir/'public_update_model.pt',map_location='cpu',weights_only=True)
        baseline_x=pd.read_csv(args.baseline_dir/model.baseline.SUBMISSION,index_col='station_id').loc[b.city.stations].to_numpy().T
        assert np.all(baseline_x.sum(1)==463)
        validate_public_context(b,public,counts,source,baseline_x,config,args.output_dir)
        np.savez_compressed(args.output_dir/'feedback_reference.npz',**source)
        torch.save({'model_type':'aggregate_feedback_scenarios_v1','config':config,'neural':baseline_saved,
                    'baseline_x':torch.as_tensor(baseline_x),'public_days':public.tolist(),
                    'public_counts':torch.as_tensor(counts),'public_hashes':[r['sha256'] for r in provenance]},args.output_dir/MODEL)
    else:
        saved=torch.load(args.output_dir/MODEL,map_location='cpu',weights_only=True)
        if saved['model_type']!='aggregate_feedback_scenarios_v1' or saved['public_days']!=public.tolist():raise ValueError('Wrong checkpoint/public days')
        config,baseline_saved,baseline_x=saved['config'],saved['neural'],saved['baseline_x'].numpy()
        np.testing.assert_array_equal(counts,saved['public_counts'].numpy())
        with np.load(args.output_dir/'feedback_reference.npz',allow_pickle=False) as z:source={k:z[k] for k in z.files}
    export(b,public,truth,counts,provenance,source,baseline_saved,baseline_x,config,args.output_dir)


if __name__=='__main__':main()
