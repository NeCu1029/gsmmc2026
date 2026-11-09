"""Calibrate a two-model inventory ensemble from two observed submission scores."""
import bike_feedback_calibration as physical
import numpy as np
from neural_allocation import optimize,realized_score


def predicted_score(costs,allocation):
    if costs.shape[:2]!=allocation.shape:raise ValueError('Cost/allocation grid mismatch')
    n=allocation.sum(1)
    if not np.all(n==n[0]) or n[0]<1:raise ValueError('Positive common fleet required')
    selected=costs[np.arange(len(costs))[:,None],np.arange(costs.shape[1]),allocation]
    if not np.isfinite(selected).all():raise ValueError('Reference allocation outside model support')
    return float(20*n[0]+selected.sum(1).mean())


def fit_pair(neural,scenarios,first,second,first_score,second_score):
    """Only the two scalar scores enter the calibration, never a new allocation's labels."""
    d0=predicted_score(neural,second)-predicted_score(neural,first)
    d1=predicted_score(scenarios,second)-predicted_score(scenarios,first)
    actual=float(second_score-first_score)
    if abs(d1-d0)<1e-8:
        raise ValueError('Two cost models have indistinguishable reference contrasts')
    raw=(actual-d0)/(d1-d0)
    weight=float(np.clip(raw,0.,1.))
    return weight,{'neural_predicted_contrast':d0,'scenario_predicted_contrast':d1,
                   'observed_contrast':actual,'scenario_weight':weight,'unclipped_weight':float(raw),
                   'contrast_inside_model_range':bool(0<=raw<=1),
                   'fitted_contrast':float((1-weight)*d0+weight*d1)}


def mix(neural,scenarios,weight,caps):
    mask=np.arange(neural.shape[-1])[None,None]<=caps[None,:,None]
    if neural.shape!=scenarios.shape:raise ValueError('Cost model level dimensions differ')
    result=(1-weight)*np.where(mask,neural,0.)+weight*np.where(mask,scenarios,0.)
    if not np.isfinite(result[np.broadcast_to(mask,result.shape)]).all():raise ValueError('Non-finite valid costs')
    return np.where(mask,result,np.inf)


def restrict_to_reference_range(costs,first,second,known_days=()):
    """Avoid extrapolation beyond either submitted stock at a station/date."""
    levels=np.arange(costs.shape[-1])[None,None]
    allowed=(levels>=np.minimum(first,second)[...,None])&(levels<=np.maximum(first,second)[...,None])
    # Exact observed cost curves need no extrapolation protection.
    allowed[np.asarray(known_days,dtype=int)]=True
    return np.where(allowed,costs,np.inf)


def solve(neural,scenarios,first,second,first_score,second_score,caps,restrict=True,known_days=()):
    weight,report=fit_pair(neural,scenarios,first,second,first_score,second_score)
    costs=mix(neural,scenarios,weight,caps)
    if restrict:costs=restrict_to_reference_range(costs,first,second,known_days)
    allocation,value=optimize(costs,caps)
    report.update({'reference_range_restriction':restrict,'N':int(allocation[0].sum()),
                   'predicted_change_from_first':value-predicted_score(costs,first),
                   'predicted_change_from_second':value-predicted_score(costs,second),
                   'reference_scores':[float(first_score),float(second_score)]})
    return allocation,costs,report
