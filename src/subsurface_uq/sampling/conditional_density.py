"""Exact nested Gaussian conditioning of multiple native fields in blocks.

Whiten each spatial covariance block once for the largest observation prefix.
Apply all prefixes and draws together, without retaining a grid-by-data matrix.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap
from scipy.linalg import solve_triangular

from .controlled_fields import KnownMeanConditioner,physical_permeability


def condition_prefixes(system,readings,prior_unit,counts,directories,*,chunk_size=8192,progress=None):
    """Save every positive-permeability draw and exact log-space moments.

    Inputs contain only geometry, revealed readings and independent Gaussian
    driving fields. No hidden reference is available to this function.
    """
    if not isinstance(system,KnownMeanConditioner):
        raise ValueError("known-mean Gaussian conditioner required")
    counts=list(counts);shape=system.shape;prior=np.asarray(prior_unit)
    if (not counts or counts!=sorted(set(counts)) or counts[0]<1 or counts[-1]!=len(system.indices)
            or any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) for n in counts)):
        raise ValueError("increasing positive prefixes ending at the full geometry required")
    if prior.ndim!=3 or tuple(prior.shape[1:])!=shape or len(prior)<2 or chunk_size<1:
        raise ValueError("multiple matching native prior fields and positive block size required")
    d=system._readings(readings,counts[-1]);samples=len(prior);width=samples+1
    at_data=prior[:,system.indices[:,0],system.indices[:,1]].T
    if not np.all(np.isfinite(at_data)):
        raise ValueError("finite Gaussian driving fields required")
    coefficients=np.zeros((counts[-1],len(counts)*width))
    stores=[]
    for j,n in enumerate(counts):
        target=Path(directories[n]);target.mkdir(parents=True,exist_ok=True)
        innovations=d[:n,None]-system.mean_log10-system.spec.sigma_R*at_data[:n]
        rhs=np.column_stack((innovations,d[:n]-system.mean_log10))
        coefficients[:n,j*width:(j+1)*width]=solve_triangular(system.cholesky[:n,:n],rhs,lower=True)
        values=open_memmap(target/'permeability_samples.npy',mode='w+',dtype=np.float32,shape=(samples,*shape))
        mean=open_memmap(target/'posterior_log10_mean.npy',mode='w+',dtype=np.float32,shape=shape)
        std=open_memmap(target/'posterior_log10_std.npy',mode='w+',dtype=np.float32,shape=shape)
        stores.append((values.reshape(samples,-1),mean.ravel(),std.ravel()))
    size=int(np.prod(shape));flat=prior.reshape(samples,-1);variance_sums=np.zeros(len(counts))
    for start in range(0,size,chunk_size):
        stop=min(start+chunk_size,size)
        queries=np.column_stack(np.unravel_index(np.arange(start,stop),shape))
        features=system.features(queries)
        projected=features@coefficients
        base=system.mean_log10+system.spec.sigma_R*np.asarray(flat[:,start:stop]).T
        if not np.all(np.isfinite(base)):
            raise ValueError("nonfinite prior; stop without replacing the draw")
        squared=np.cumsum(features*features,axis=1)
        for j,n in enumerate(counts):
            values,mean,std=stores[j]
            logs=base+projected[:,j*width:j*width+samples]
            values[:,start:stop]=physical_permeability(logs).T
            mean[start:stop]=system.mean_log10+projected[:,j*width+samples]
            unit_variance=1-squared[:,n-1]
            if unit_variance.min(initial=0)<-1.e-8:
                raise ArithmeticError("negative conditional variance; no covariance alteration")
            variance=system.spec.sigma_R**2*np.maximum(unit_variance,0.)
            std[start:stop]=np.sqrt(variance);variance_sums[j]+=variance.sum()
        if progress is not None:progress(stop,size)
    for values,mean,std in stores:
        values.flush();mean.flush();std.flush()
    del stores
    return {n:float(variance_sums[j]/size) for j,n in enumerate(counts)}
