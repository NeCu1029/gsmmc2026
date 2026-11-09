"""Add a first weekend observation without diluting weekday public signatures."""
import argparse
import hashlib
import json
from pathlib import Path
import progressive_public_core as core
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
POLICY='daytype_residual_v1'
MODEL_NAME=core.MODEL_NAME
SUBMISSION='최종제출_배치표_B지역_D13주말추가.csv'


def infer_daytypes(data,prior,signature,public_days,truth,days,sites):
    core.base.check_public_protocol(prior,signature,public_days)
    trained_kinds=np.unique(data.city.off_day[prior['public_days']])
    if len(trained_kinds)!=1:
        raise ValueError('This observation adapter expects original signatures from one day type')
    observed_kind=data.city.off_day[public_days]
    reference=observed_kind==trained_kinds[0]
    reference_days=public_days[reference]
    reference_truth=truth[reference]
    result=core.base.infer(data,prior,signature,reference_days,reference_truth,days,sites)
    # For the original weekday-only public set, this is exactly the prior model.
    if reference.all():
        return result
    actual=core.base.neural.transfer.increments(truth,data.capacity)
    for kind in np.unique(observed_kind[~reference]):
        matched=observed_kind==kind
        predicted_public=core.base.infer(data,prior,signature,reference_days,reference_truth,public_days[matched],sites)
        residual=(actual[matched][:,sites]-predicted_public).mean(0)
        key='offday_weight' if kind else 'weekday_weight'
        weight=float(prior['config'][key])
        result[data.city.off_day[days]==kind]+=weight*residual
    # Observed dates themselves always use exact inventory costs.
    positions={int(d):i for i,d in enumerate(public_days)}
    for i,day in enumerate(days):
        if int(day) in positions:
            result[i]=actual[positions[int(day)],sites]
    if not np.isfinite(result).all():
        raise ValueError('Non-finite day-type corrected costs')
    return result


def predictor(policy):
    if policy=='all_public_mean_v1':
        return core.base.infer
    if policy==POLICY:
        return infer_daytypes
    raise ValueError(f'Unknown predecessor observation policy: {policy}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','predict'],nargs='?',default='predict')
    parser.add_argument('--data-dir',type=Path,default=ROOT.parents[1])
    parser.add_argument('--predecessor-dir',type=Path,default=ROOT/'results_public_d24')
    parser.add_argument('--prior-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--signature-dir',type=Path,default=ROOT/'results_v2')
    parser.add_argument('--public-dir',type=Path,default=ROOT.parents[1]/'open_data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results_public_d13')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    b=core.base.neural.base.read_data(args.data_dir,'B')
    if args.action=='build':
        public_days,_,_=core.base.neural.transfer.read_public(args.public_dir,b)
        source_file=args.predecessor_dir/MODEL_NAME
        source=torch.load(source_file,map_location='cpu',weights_only=True)
        if source.get('model_type') not in ('public_observation_update_v1','public_observation_update_v2','public_observation_update_v3'):
            raise ValueError('Expected the original unpenalized ensemble')
        if not set(source['observed_public_days']).issubset(public_days.tolist()):
            raise ValueError('A predecessor public date is missing')
        previous_policy=source.get('public_input_policy','all_public_mean_v1')
        previous=predictor(previous_policy)
        saved={'model_type':'public_observation_update_v3','public_input_policy':POLICY,'predecessor_input_policy':previous_policy,
               'prior':source['prior'],'signature':source['signature'],'weights':[.5,.5],
               'observed_public_days':public_days.tolist(),'predecessor_public_days':source['observed_public_days'],
               'source_model_sha256':hashlib.sha256(source_file.read_bytes()).hexdigest()}
        torch.save(saved,args.output_dir/MODEL_NAME)
        core.validate(core.base.neural.base.read_data(args.data_dir,'A'),public_days,args.predecessor_dir,args.prior_dir,args.signature_dir,args.output_dir,
                      previous_infer=previous,updated_infer=infer_daytypes)
        report_file=args.output_dir/'A_validation.json'
        report=json.loads(report_file.read_text(encoding='utf-8'))
        report['public_input_policy']=POLICY
        report['weekend_residual_weight']=float(source['prior']['config']['offday_weight'])
        report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    saved=torch.load(args.output_dir/MODEL_NAME,map_location='cpu',weights_only=True)
    core.export(b,args.public_dir,args.predecessor_dir,args.output_dir,
                previous_infer=predictor(saved.get('predecessor_input_policy','all_public_mean_v1')),
                updated_infer=infer_daytypes,submission_name=SUBMISSION,policy_name=POLICY,model_type='public_observation_update_v3')


if __name__=='__main__':
    main()
