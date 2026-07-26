"""Pair-aware detector evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta

@dataclass(frozen=True)
class DetectionResult:
    predictions: np.ndarray
    scores: np.ndarray
    accuracy: float
    ci_low: float
    ci_high: float
    complete_pair_rate: float
    pair_ci_low: float
    pair_ci_high: float

def fit_ols_detector(X, y):
    X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float).reshape(-1)
    design=np.column_stack([np.ones(len(X)),X])
    return np.linalg.lstsq(design,y,rcond=None)[0]

def score_ols_detector(beta_hat,X):
    X=np.asarray(X,dtype=float)
    return np.column_stack([np.ones(len(X)),X])@np.asarray(beta_hat)

def clopper_pearson(successes,trials,alpha=0.05):
    if not 0<=successes<=trials or trials<=0: raise ValueError("invalid successes/trials")
    low=0.0 if successes==0 else float(beta.ppf(alpha/2,successes,trials-successes+1))
    high=1.0 if successes==trials else float(beta.ppf(1-alpha/2,successes+1,trials-successes))
    return low,high

def paired_loso_predictions(profiles,labels,pair_ids,threshold=0.5):
    profiles=np.asarray(profiles,dtype=float); labels=np.asarray(labels,dtype=int).reshape(-1); pair_ids=np.asarray(pair_ids).reshape(-1)
    unique=np.unique(pair_ids)
    if len(unique)<3: raise ValueError("at least three pairs are required")
    if any(np.sum(pair_ids==p)!=2 for p in unique): raise ValueError("each pair_id must identify exactly two networks")
    scores=np.empty(len(labels))
    for held in unique:
        train=pair_ids!=held; test=~train
        scores[test]=score_ols_detector(fit_ols_detector(profiles[train],labels[train]),profiles[test])
    pred=(scores>=threshold).astype(int); successes=int(np.sum(pred==labels)); low,high=clopper_pearson(successes,len(labels))
    pair_successes=sum(bool(np.all(pred[pair_ids==p]==labels[pair_ids==p])) for p in unique)
    pl,ph=clopper_pearson(pair_successes,len(unique))
    return DetectionResult(pred,scores,successes/len(labels),low,high,pair_successes/len(unique),pl,ph)

def pair_bootstrap_interval(pair_ids,statistic,n_resamples=10000,seed=0,alpha=0.05):
    pair_ids=np.asarray(pair_ids).reshape(-1); unique=np.unique(pair_ids); members={p:np.flatnonzero(pair_ids==p) for p in unique}
    rng=np.random.RandomState(seed); values=[]
    for _ in range(n_resamples):
        sampled=rng.choice(unique,size=len(unique),replace=True); idx=np.concatenate([members[p] for p in sampled]); values.append(float(statistic(idx)))
    return tuple(np.quantile(np.asarray(values),[alpha/2,1-alpha/2]).astype(float))

def paired_accuracy_difference_interval(first_predictions,second_predictions,labels,pair_ids,n_resamples=10000,seed=0):
    first_predictions=np.asarray(first_predictions); second_predictions=np.asarray(second_predictions); labels=np.asarray(labels)
    def stat(idx): return float(np.mean(first_predictions[idx]==labels[idx])-np.mean(second_predictions[idx]==labels[idx]))
    estimate=stat(np.arange(len(labels))); low,high=pair_bootstrap_interval(pair_ids,stat,n_resamples,seed)
    return estimate,low,high
