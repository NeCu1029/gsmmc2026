"""Episodic public-day attention for inventory marginal costs (no extra N penalty)."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import update_public_ensemble as old
from verify_neural_submission import official_score
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_attention'
SEEDS = (20261109, 20261110)
SUBMISSION = '최종제출_배치표_B지역_Attention.csv'
ALPHA = .5  # Predeclared before outer evaluation; do not tune on D25-D30.


class AttentionInventory(nn.Module):
    def __init__(self, ns, nt):
        super().__init__()
        for name, n in [('s', ns), ('t', nt)]:
            self.register_buffer(name+'_mean', torch.zeros(n))
            self.register_buffer(name+'_scale', torch.ones(n))
        self.query = nn.Sequential(nn.Linear(ns+nt, 32), nn.SiLU(), nn.Linear(32, 32))
        self.key = nn.Sequential(nn.Linear(72+nt, 32), nn.SiLU(), nn.Linear(32, 32))
        self.value = nn.Sequential(nn.Linear(72+nt, 32), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(64, 32), nn.SiLU())
        self.base = nn.Sequential(nn.Linear(35, 32), nn.SiLU(), nn.Linear(32, 1))
        self.residual = nn.Sequential(nn.Linear(37, 32), nn.SiLU(), nn.Linear(32, 1))
        self.gate = nn.Linear(34, 1)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def scales(self, data, days, sites):
        for prefix, x in [('s', torch.tensor(data.similarity[sites], dtype=torch.float32)),
                          ('t', old.neural.inventory.day_features(data)[days])]:
            getattr(self, prefix+'_mean').copy_(x.mean(0))
            getattr(self, prefix+'_scale').copy_(x.std(0, unbiased=False).clamp_min(.2))

    def forward(self, data, days, sites, support_days, support_net, support_inc):
        s = torch.tensor(data.similarity[sites.numpy()], dtype=torch.float32)
        s = ((s-self.s_mean)/self.s_scale).clamp(-5, 5)
        all_t = ((old.neural.inventory.day_features(data)-self.t_mean)/self.t_scale).clamp(-5, 5)
        q = self.query(torch.cat([s, all_t[days]], -1))
        cap = torch.tensor(data.capacity[sites.numpy()], dtype=torch.float32)[:, None]
        net = support_net / cap[:, :, None]
        p = torch.cat([torch.asinh(net), torch.asinh(net.cumsum(-1)), torch.log1p(net.abs())], -1)
        token = torch.cat([p, all_t[support_days]], -1)
        logits = (self.key(token)*q[:, None]).sum(-1)/(32**.5)
        w = logits.softmax(-1)
        c = self.context(torch.cat([q, (w[:, :, None]*self.value(token)).sum(1)], -1))
        anchor = (w[:, :, None]*support_inc).sum(1)
        spread = ((support_inc-anchor[:, None]).square()*w[:, :, None]).sum(1).clamp_min(1e-8).sqrt()
        x = torch.arange(anchor.shape[-1], dtype=torch.float32)[None].expand(len(days), -1)
        stock = torch.stack([x/cap, torch.log1p(x)/3, (x >= torch.floor(cap/2)).float()], -1)
        context = c[:, None].expand(-1, x.shape[1], -1)
        prior = 20*self.base(torch.cat([context, stock], -1)).squeeze(-1)
        off = torch.tensor(np.array(data.city.off_day, copy=True), dtype=torch.float32)
        mismatch = (off[days, None]-off[support_days]).abs().mean(1, keepdim=True)
        entropy = -(w*w.clamp_min(1e-8).log()).sum(1, keepdim=True)/np.log(max(2,w.shape[1]))
        gate = self.gate(torch.cat([c, mismatch, entropy], -1)).sigmoid()
        residual = 20*self.residual(torch.cat([context, stock, torch.asinh(anchor[..., None]/20),
                                            torch.log1p(spread[..., None]/20)], -1)).squeeze(-1)
        return gate*anchor + (1-gate)*prior + residual


def episodic_context(rng, days, targets, count):
    result = []
    # Half the episodes mirror the Tuesday/Wednesday-only B observations.
    narrow = rng.random() < .5
    for d in targets:
        pool = days[days != d]
        if narrow:
            restricted = pool[np.isin(pool % 7, [1, 2])]
            if len(restricted) >= count:
                pool = restricted
        result.append(rng.choice(pool, count, replace=False))
    context = np.asarray(result)
    assert not (context == np.asarray(targets)[:, None]).any()
    return torch.as_tensor(context)


@torch.no_grad()
def predict(model, data, public_days, truth, days, sites):
    model.eval()
    increments = old.neural.transfer.increments(truth, data.capacity)
    pairs = old.neural.base.base.examples(days, sites)
    values = []
    for pair in pairs.split(256):
        d, s = pair.unbind(1)
        support = torch.as_tensor(np.broadcast_to(public_days, (len(d), len(public_days))).copy())
        net = torch.tensor(truth[:, s.numpy()].transpose(1,0,2), dtype=torch.float32)
        inc = torch.tensor(increments[:, s.numpy()].transpose(1,0,2), dtype=torch.float32)
        values.append(model(data, d, s, support, net, inc).numpy())
    return np.concatenate(values).reshape(len(days), len(sites), -1).astype(float)


def fit(data, target, days, sites, seed, epochs, validation=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AttentionInventory(data.similarity.shape[-1], old.neural.inventory.day_features(data).shape[-1])
    model.scales(data, days, sites)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.03)
    pairs = old.neural.base.base.examples(days, sites)
    truth = data.city.net_flow
    targets = torch.tensor(target, dtype=torch.float32)
    best, selected = float('inf'), epochs
    for epoch in range(1, epochs+1):
        model.train()
        for ix in torch.randperm(len(pairs)).split(192):
            d, s = pairs[ix].unbind(1)
            support = episodic_context(rng, days, d.numpy(), int(rng.choice([2,3,4])))
            net, inc = truth[support, s[:, None]], targets[support, s[:, None]]
            pred = model(data, d, s, support, net, inc)
            mask = torch.arange(pred.shape[-1])[None] < torch.as_tensor(data.capacity[s.numpy()])[:, None]
            loss = ((pred-targets[d,s])[mask]/20).square().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        if validation is not None and epoch % 4 == 0:
            vd, vs, public = validation
            pred = predict(model, data, public, truth[public].numpy(), vd, vs)
            allocation, _ = old.optimize(old.neural.inventory.cost_table(pred, data.capacity[vs]), data.capacity[vs])
            score = old.realized_score(truth[vd][:,vs].numpy(), allocation, data.capacity[vs])['score']
            print(json.dumps({'inner_epoch':epoch,'score':score}), flush=True)
            if score < best:
                best, selected = score, epoch
    return model, selected


def allocation(pred, data, groups=None):
    if groups is None:
        return old.optimize(old.neural.inventory.cost_table(pred, data.capacity), data.capacity)[0]
    x = np.empty(pred.shape[:2], dtype=np.int64)
    for sites in groups:
        x[:,sites] = old.optimize(old.neural.inventory.cost_table(pred[:,sites], data.capacity[sites]), data.capacity[sites])[0]
    return x


def run(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    a, b = old.neural.base.read_data(args.data_dir,'A'), old.neural.base.read_data(args.data_dir,'B')
    public, btruth, provenance = old.neural.transfer.read_public(args.public_dir,b)
    future = np.setdiff1d(np.arange(23,30), public)
    assert not np.intersect1d(future, public).size
    target = old.neural.transfer.increments(a.city.net_flow.numpy(),a.capacity)
    order = np.random.default_rng(old.neural.base.SEED).permutation(len(a.capacity))
    groups = np.split(order,[i*(len(order)//5) for i in range(1,5)])
    oof = np.empty((len(future),len(a.capacity),int(a.capacity.max())))
    records=[]
    for i, test in enumerate(groups):
        print(f'Attention outer fold {i+1}/5',flush=True)
        sources=np.concatenate(groups[i+1:]+groups[:i])
        inner_dev, inner_fit = sources[:len(sources)//4], sources[len(sources)//4:]
        _, epoch=fit(a,target,np.arange(17),inner_fit,SEEDS[0],32,
                     (np.arange(17,23),inner_dev,np.array([8,9,15,16])))
        models=[fit(a,target,np.arange(23),sources,seed,epoch)[0] for seed in SEEDS]
        oof[:,test]=np.mean([predict(m,a,public,a.city.net_flow[public].numpy(),future,test) for m in models],axis=0)
        records.append({'fold':i,'epochs':epoch,'test_station_ids':a.city.stations[test].tolist()})
        (output/'progress.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    with np.load(args.baseline_dir/'A_updated_costs.npz') as z:
        np.testing.assert_array_equal(z['days'],future)
        baseline=z['increments'].copy()
    predictions={'baseline':baseline,'attention':oof,'ensemble':(1-ALPHA)*baseline+ALPHA*oof}
    report={'evaluation_dates':a.city.dates[future].strftime('%Y-%m-%d').tolist(),
            'public_days':public.tolist(),'folds':records,'fixed_attention_weight':ALPHA,
            'protocol':'5 held-out station folds. Outer fit D01-D23; D24 observed only as context for held-out stations. Inner select epochs on separate stations D18-D23, context D09,D10,D16,D17; inner fit D01-D17. D25-D30 never used for parameter or weight selection.',
            'scores':{},'extra_fleet_penalty':0}
    spec=importlib.util.spec_from_file_location('official',args.data_dir/'score.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    for name,pred in predictions.items():
        x=allocation(pred,a,groups)
        score=official_score(a,future,x,a.counts[future].numpy(),official)
        report['scores'][name]={'score':score,**old.realized_score(a.city.net_flow[future].numpy(),x,a.capacity)}
        pd.DataFrame(x.T,index=a.city.stations,columns=[f'D{d+1:02d}' for d in future]).rename_axis('station_id').to_csv(output/f'A_{name}.csv',encoding='utf-8-sig')
    np.savez_compressed(output/'A_attention_costs.npz',increments=oof,days=future)
    (output/'A_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['scores']),flush=True)
    epoch=int(np.median([r['epochs'] for r in records]))
    # All A labels are allowed for final fit; outer validation above remains untouched.
    models=[fit(a,target,np.arange(30),np.arange(len(a.capacity)),seed,epoch)[0] for seed in SEEDS]
    prior=torch.load(args.baseline_dir/'public_update_model.pt',map_location='cpu',weights_only=True)
    saved={'model_type':'episodic_public_attention_v1','ns':a.similarity.shape[-1],
           'nt':old.neural.inventory.day_features(a).shape[-1],'states':[m.state_dict() for m in models],
           'epochs':epoch,'seeds':list(SEEDS),'attention_weight':ALPHA,'baseline':prior,'public_days':public.tolist()}
    torch.save(saved,output/'attention_model.pt')
    export(args,b,btruth,public,provenance,saved)


def export(args,b,truth,public,provenance,saved):
    sites,days=np.arange(len(b.capacity)),np.arange(30)
    models=[]
    for state in saved['states']:
        model=AttentionInventory(saved['ns'],saved['nt']);model.load_state_dict(state);models.append(model)
    att=np.mean([predict(m,b,public,truth,days,sites) for m in models],axis=0)
    prior=saved['baseline']
    baseline=old.infer(b,prior['prior'],prior['signature'],public,truth,days,sites)
    exact=old.neural.transfer.increments(truth,b.capacity)
    att[public]=exact
    mixed=(1-saved['attention_weight'])*baseline+saved['attention_weight']*att
    mixed[public]=exact
    x=allocation(mixed,b)
    assert np.isfinite(x).all() and (x>=0).all() and (x<=b.capacity).all() and (x.sum(1)==x[0].sum()).all()
    file=args.output_dir/SUBMISSION
    pd.DataFrame(x.T,index=b.city.stations,columns=[f'D{d+1:02d}' for d in days]).rename_axis('station_id').to_csv(file,encoding='utf-8-sig')
    result={'N':int(x[0].sum()),'stations':len(b.capacity),'days':30,'extra_fleet_penalty':0,
            'official_fleet_coefficient':20,'fixed_attention_weight':saved['attention_weight'],
            'public_files':provenance,'public_in_sample_score':old.realized_score(truth,x[public],b.capacity),
            'full_B_score':None,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    (args.output_dir/'B_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','predict'],default='train',nargs='?')
    p.add_argument('--data-dir',type=Path,default=ROOT.parents[1])
    p.add_argument('--public-dir',type=Path,default=ROOT.parents[1]/'open_data')
    p.add_argument('--baseline-dir',type=Path,default=ROOT/'results_public_d24')
    p.add_argument('--output-dir',type=Path,default=OUT)
    args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.action=='train':run(args)
    else:
        data=old.neural.base.read_data(args.data_dir,'B')
        public,truth,provenance=old.neural.transfer.read_public(args.public_dir,data)
        saved=torch.load(args.output_dir/'attention_model.pt',map_location='cpu',weights_only=True)
        assert public.tolist()==saved['public_days']
        export(args,data,truth,public,provenance,saved)
