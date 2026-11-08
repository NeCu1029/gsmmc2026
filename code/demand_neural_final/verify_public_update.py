import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import update_public_ensemble as update
from verify_neural_submission import official_score
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent


def verify(data_dir,public_dir,prior_dir,signature_dir,output):
    spec=importlib.util.spec_from_file_location('official_scorer',data_dir/'score.py')
    official=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    a,b=update.neural.base.read_data(data_dir,'A'),update.neural.base.read_data(data_dir,'B')
    summary=json.loads((output/'B_summary.json').read_text(encoding='utf-8'))
    report=json.loads((output/'A_validation.json').read_text(encoding='utf-8'))
    file=output/update.SUBMISSION
    frame=pd.read_csv(file,encoding='utf-8-sig')
    assert list(frame)==['station_id']+[f'D{i:02d}' for i in range(1,31)]
    assert frame.station_id.tolist()==b.city.stations.tolist()
    x=frame.iloc[:,1:].to_numpy(float).T
    assert np.isfinite(x).all() and np.equal(x,np.floor(x)).all() and (x>=0).all() and (x<=b.capacity).all()
    assert (x.sum(1)==summary['N']).all() and summary['N']>0
    assert summary['official_fleet_coefficient']==20 and summary['extra_fleet_penalty']==0 and summary['cost_bound'] is None
    public_days,truth,_=update.neural.transfer.read_public(public_dir,b)
    counts=np.stack([np.maximum(-truth,0),np.maximum(truth,0)],axis=-1)
    public_score=official_score(b,public_days,x[public_days],counts,official)
    assert abs(public_score-summary['updated_public_actual']['score'])<.01
    selected_a=pd.read_csv(output/'A_검증배치.csv',index_col='station_id').loc[a.city.stations].to_numpy().T
    a_score=official_score(a,np.arange(23,30),selected_a,a.counts[23:].numpy(),official)
    assert abs(a_score-report['updated']['score'])<.01
    saved=torch.load(output/update.MODEL_NAME,map_location='cpu',weights_only=True)
    original_prior=torch.load(prior_dir/'public_inventory.pt',map_location='cpu',weights_only=True)
    original_signature=torch.load(signature_dir/update.neural.MODEL,map_location='cpu',weights_only=True)
    for name,original in (('prior',original_prior),('signature',original_signature)):
        checkpoint=saved[name]
        assert checkpoint['config']==original['config'] and checkpoint['public_days']==original['public_days']
        assert len(checkpoint['states'])==len(original['states'])
        for state,old_state in zip(checkpoint['states'],original['states']):
            assert state.keys()==old_state.keys()
            for k,v in state.items():
                torch.testing.assert_close(v,old_state[k],rtol=0,atol=0)
    days,sites=np.arange(len(b.city.dates)),np.arange(len(b.city.stations))
    predictions=update.infer(b,saved['prior'],saved['signature'],public_days,truth,days,sites)
    exact=update.neural.transfer.increments(truth,b.capacity)
    np.testing.assert_array_equal(predictions[public_days],exact)
    regenerated,_=update.optimize(update.neural.inventory.cost_table(predictions,b.capacity),b.capacity)
    np.testing.assert_array_equal(regenerated,x)
    result={'constraints':'PASS','N':summary['N'],'baseline_N':summary['baseline_N'],'stations':159,'days':30,
            'public_dates':summary['updated_public_dates'],'extra_fleet_penalty':0,'cost_bound':None,
            'A_previous_score':report['previous']['score'],'A_updated_score':a_score,'A_change_percent':report['score_change_percent'],
            'B_public_actual_score':public_score,'full_B_actual_score':None,
            'original_neural_weights_and_configs_identical':True,'all_three_public_days_use_exact_costs':True,
            'saved_model_reproduces_final_csv':True,'csv_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT.parents[1])
    parser.add_argument('--prior-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--signature-dir',type=Path,default=ROOT/'results_v2')
    parser.add_argument('--public-dir',type=Path,default=ROOT.parents[1] / 'open_data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results_public_d23')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verify(args.data_dir,args.public_dir,args.prior_dir,args.signature_dir,args.output_dir)
