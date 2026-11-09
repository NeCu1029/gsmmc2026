"""Calibrate physical demand scenarios to aggregate feedback, never station labels."""
import update_public_daytypes as baseline
import bike_same_date_transfer as profile
import numpy as np
import pandas as pd
from numba import njit
from neural_allocation import scenario_costs


def read_public_counts(folder,data,public):
    frames=[pd.read_csv(p,encoding='utf-8-sig').rename(columns={'날짜':'date','시':'hour'}) for p in sorted(folder.glob('*.csv'))]
    frame=pd.concat(frames,ignore_index=True)
    frame['date']=pd.to_datetime(frame['date'])
    index=pd.MultiIndex.from_product([data.city.dates[public],data.city.stations,range(24)],names=['date','station_id','hour'])
    if frame.duplicated(index.names).any():raise ValueError('Duplicate public station-hour')
    values=frame.set_index(index.names).loc[index,['rentals','returns']].to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or not np.equal(values,np.floor(values)).all():raise ValueError('Invalid counts')
    return values.reshape(len(public),len(data.capacity),24,2)


def reference(data,sites):
    result=profile.source_reference(data,sites)
    result['counts']=data.counts[:,sites].numpy().astype(float)
    return result


def scenarios(source,target,sites,public,counts):
    """Station matching sees supplied public observations and static features only."""
    truth=counts[...,1]-counts[...,0]
    source_truth=source['net'][public]
    sf=profile.public_features(source['similarity'],source['capacity'],source['off'],public,source_truth)
    tf=profile.public_features(target.similarity[sites],target.capacity[sites],target.city.off_day,public,truth)
    distance=np.mean(((tf[:,None]-sf[None])/np.maximum(sf.std(0),.2))**2,axis=-1)
    nearest=np.argsort(distance,axis=1,kind='stable')[:,:min(32,len(sf))]
    delta=np.take_along_axis(distance,nearest,axis=1)
    mass=np.exp(-delta/(2*np.maximum(np.median(delta,axis=1,keepdims=True),1e-6)))
    mass/=mass.sum(1,keepdims=True)
    source_mean=(source['counts'][:,nearest]*mass[None,:,:,None,None]).sum(2)
    paths=[];weights=[]
    for d in range(len(target.city.dates)):
        use=np.flatnonzero(target.city.off_day[public]==target.city.off_day[d])
        if not len(use):use=np.arange(len(public))
        # Ratios retain the source city's calendar and hourly weather response.
        # Pseudocounts and clipping prevent amplification in nearly empty hours.
        ratio=np.clip((source_mean[d,:,None]+.5)/(source_mean[public[use]].transpose(1,0,2,3)+.5),.2,3.)
        sample=counts[use].transpose(1,0,2,3)*ratio
        # A fixed support axis allows variable weekday/weekend support sizes.
        padded=np.zeros((len(sites),len(public),24,2))
        padded[:,:len(use)]=sample
        weight=np.zeros((len(sites),len(public)))
        weight[:,:len(use)]=1/len(use)
        paths.append(padded);weights.append(weight)
    return np.asarray(paths),np.asarray(weights)


@njit(cache=True)
def metrics_one(counts,weight,allocation,caps,rates):
    result=np.zeros(4)
    for i in range(len(caps)):
        cap=caps[i]
        for s in range(counts.shape[1]):
            if weight[i,s]==0:continue
            b=float(allocation[i])
            for h in range(24):
                z=b+rates[1]*counts[i,s,h,1]-rates[0]*counts[i,s,h,0]
                u=max(-z,0.);o=max(z-cap,0.)
                b=min(float(cap),max(0.,z))
                result[0]+=weight[i,s]*u
                result[1]+=weight[i,s]*u*u
                result[2]+=weight[i,s]*o
                result[3]+=weight[i,s]*max(b-cap//2,0.)
    return result


def observed_metrics(truth,allocation,caps):
    counts=np.stack([np.maximum(-truth,0),np.maximum(truth,0)],axis=-1)[:,:,None]
    weight=np.ones(counts.shape[:3])
    return np.stack([metrics_one(counts[d],weight[d],allocation[d],caps,np.ones(2)) for d in range(len(truth))])


def fit_rates(paths,weights,allocation,caps,feedback,regularization):
    """Two bounded daily volume factors; four city totals constrain each fit.

    Feedback is an observation at a previously submitted allocation. This is
    conditional calibration, not an independent held-out demand prediction.
    """
    rates=[];before=[];after=[]
    for d in range(len(paths)):
        target=feedback[d]
        def loss(log_rates):
            predicted=metrics_one(paths[d],weights[d],allocation[d],caps,np.exp(log_rates))
            error=np.log((predicted+1)/(target+1))
            return float(np.dot(error*error,np.array([1.,1.,.5,.25]))+regularization*np.dot(log_rates,log_rates))
        center=np.zeros(2)
        before.append(metrics_one(paths[d],weights[d],allocation[d],caps,np.ones(2)))
        for step in (.16,.08,.04,.02,.01):
            best=loss(center);chosen=center
            for a in (-2,-1,0,1,2):
                for b in (-2,-1,0,1,2):
                    trial=np.clip(center+step*np.array([a,b]),np.log(.625),np.log(1.6))
                    value=loss(trial)
                    if value<best-1e-12:best,chosen=value,trial
            center=chosen
        rates.append(np.exp(center))
        after.append(metrics_one(paths[d],weights[d],allocation[d],caps,np.exp(center)))
    return np.asarray(rates),np.asarray(before),np.asarray(after)


def costs(paths,weights,caps,rates):
    net=paths[...,1]*rates[:,None,None,None,1]-paths[...,0]*rates[:,None,None,None,0]
    raw=scenario_costs(net,weights,caps)
    return raw-raw[:,:,:1]


def exact_public(prediction,public,truth,caps):
    exact=profile.relative_public_costs(truth,caps)
    mask=np.arange(prediction.shape[-1])[None,:]<=caps[:,None]
    prediction[public]=np.where(mask[None],exact,np.inf)
    return prediction


def blend(a,b,weight):
    if weight==1:return b.copy()
    if weight==0:return a.copy()
    return (1-weight)*a+weight*b
