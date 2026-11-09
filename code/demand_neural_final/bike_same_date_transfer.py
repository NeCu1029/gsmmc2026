"""Transfer source-city same-date inventory-cost deviations to public B profiles."""
import update_public_daytypes as baseline
import numpy as np
from numba import njit
from neural_allocation import scenario_costs


def source_reference(data,sites):
    return {'net':data.city.net_flow[:,sites].numpy().astype(float),'capacity':data.capacity[sites],
            'similarity':data.similarity[sites],'off':data.city.off_day,
            'dates':data.city.dates.strftime('%Y-%m-%d').to_numpy(str)}


def relative_public_costs(truth,caps):
    costs=scenario_costs(truth[:,:,None,:].astype(float),np.ones(truth.shape[:2]+(1,)),caps)
    relative=costs-costs[:,:,:1]
    return np.where(np.isfinite(relative),relative,0.)


def public_features(similarity,caps,off,public_days,truth):
    costs=relative_public_costs(truth,caps)
    result=[]
    for kind in (False,True):
        matched=off[public_days]==kind
        if not matched.any():matched=np.ones(len(public_days),dtype=bool)
        net=truth[matched].mean(0)
        curve=costs[matched].mean(0)
        sampled=np.stack([np.interp(np.linspace(0,c,9),np.arange(c+1),curve[i,:c+1]) for i,c in enumerate(caps)])
        result.extend([np.arcsinh(sampled/caps[:,None]/20),np.arcsinh(net[:,:12]/caps[:,None]),
                       np.arcsinh(net.cumsum(-1)[:,[5,11,17,23]]/caps[:,None])])
    return np.concatenate(result+[.3*similarity],axis=-1)


@njit(cache=True)
def neighbor_costs(net,neighbors,scales,caps):
    days,_,hours=net.shape
    sites,k=neighbors.shape
    costs=np.zeros((days,sites,k,int(caps.max())+1))
    for d in range(days):
        for i in range(sites):
            cap=caps[i]
            for j in range(k):
                s=neighbors[i,j]
                scale=scales[i,j]
                for x in range(cap+1):
                    b=float(x);cost=0.
                    for h in range(hours):
                        z=b+scale*net[d,s,h]
                        u=max(-z,0.);o=max(z-cap,0.)
                        b=min(float(cap),max(0.,z))
                        cost+=6*u*u+5*o+max(b-cap//2,0.)
                    costs[d,i,j,x]=cost
    return costs


def prepare_transfer(source,target,sites,public_days,truth,max_neighbors=32):
    """Reads target observations only through truth; no hidden target labels."""
    dates=target.city.dates.strftime('%Y-%m-%d').to_numpy(str)
    if not np.array_equal(source['dates'],dates):raise ValueError('Source and target dates must align')
    caps=target.capacity[sites]
    source_public=source['net'][public_days]
    sf=public_features(source['similarity'],source['capacity'],source['off'],public_days,source_public)
    tf=public_features(target.similarity[sites],caps,target.city.off_day,public_days,truth)
    feature_scale=np.maximum(sf.std(0),.2)
    distances=np.mean(((tf[:,None]-sf[None])/feature_scale)**2,axis=-1)
    k=min(max_neighbors,len(source['capacity']))
    nearest=np.argsort(distances,axis=1,kind='stable')[:,:k]
    nearest_distance=np.take_along_axis(distances,nearest,axis=1)
    # Public net activity estimates volume; one dock-capacity of pseudo-volume
    # stabilizes low-activity stations. All activity is observed, not future truth.
    source_volume=np.abs(source_public).sum(-1).mean(0)+source['capacity']
    target_volume=np.abs(truth).sum(-1).mean(0)+caps
    scales=np.clip(target_volume[:,None]/source_volume[nearest],.5,2.)
    raw=neighbor_costs(source['net'],nearest,scales,caps)
    raw-=raw[:,:,:,:1]
    anchor_public=relative_public_costs(truth,caps)
    anchor=np.empty((len(dates),len(sites),int(caps.max())+1))
    for d in range(len(dates)):
        matched=target.city.off_day[public_days]==target.city.off_day[d]
        if not matched.any():matched=np.ones(len(public_days),dtype=bool)
        anchor[d]=anchor_public[matched].mean(0)
    return {'raw':raw,'distance':nearest_distance,'anchor':anchor,'anchor_public':anchor_public,
            'public_days':public_days,'off':target.city.off_day,'caps':caps}


def transfer_costs(prepared,k=32,weight=.5):
    k=min(k,prepared['distance'].shape[1])
    distance=prepared['distance'][:,:k]
    bandwidth=np.maximum(np.median(distance,axis=1,keepdims=True),1e-6)
    mass=np.exp(-distance/(2*bandwidth))
    mass/=mass.sum(1,keepdims=True)
    source_mean=(prepared['raw'][:,:,:k]*mass[None,:,:,None]).sum(2)
    costs=prepared['anchor'].copy()
    for d in range(len(costs)):
        matched=prepared['off'][prepared['public_days']]==prepared['off'][d]
        if not matched.any():matched=np.ones(len(prepared['public_days']),dtype=bool)
        reference=source_mean[prepared['public_days'][matched]].mean(0)
        costs[d]+=weight*(source_mean[d]-reference)
    for i,d in enumerate(prepared['public_days']):costs[d]=prepared['anchor_public'][i]
    mask=np.arange(costs.shape[-1])[None,None]<=prepared['caps'][None,:,None]
    if not np.isfinite(costs[np.broadcast_to(mask,costs.shape)]).all():raise ValueError('Non-finite predicted costs')
    return np.where(mask,costs,np.inf)


TRANSFER_CONFIGS=[{'k':8,'weight':.5},{'k':16,'weight':.5},{'k':32,'weight':.5},
                  {'k':32,'weight':.25},{'k':32,'weight':1.}]
