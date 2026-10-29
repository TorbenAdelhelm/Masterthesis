"""Moment/variogram profiles of supplied maps, not geological error models.

A profile is a stationary Matérn-3/2 log-Gaussian law. Its expected finite-map
spatial variance matches the observed map variance without normalizing draws.
Mean-removal bias is accounted for through the variance of a grid average.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.ndimage import label
from scipy.optimize import least_squares
from scipy.special import ndtri

from .gstools_reference import GSToolsReferenceSpec


DIRECTIONS={"row":(1,0),"column":(0,1),"diagonal_positive":(1,1),"diagonal_negative":(1,-1)}


def pairs(logs,dr,dc):
    h,w=logs.shape
    ar=slice(max(dr,0),min(h,h+dr));br=slice(max(-dr,0),min(h,h-dr))
    ac=slice(max(dc,0),min(w,w+dc));bc=slice(max(-dc,0),min(w,w-dc))
    return logs[ar,ac],logs[br,bc]


def variogram_curves(logs,lags):
    a=np.asarray(logs,dtype=float)
    if a.ndim!=2 or not np.all(np.isfinite(a)) or min(a.shape)<=max(lags):
        raise ValueError("finite log map and lags inside both axes required")
    return {name:np.array([.5*np.mean((pairs(a,dr*int(k),dc*int(k))[0]-
                                     pairs(a,dr*int(k),dc*int(k))[1])**2) for k in lags])
            for name,(dr,dc) in DIRECTIONS.items()}


def rectangle_quadrature(shape,cell_size_m,order):
    nodes,weights=leggauss(order);t=(nodes+1)/2;weights=weights/2*(1-t)
    dy=(shape[0]*cell_size_m*t)[:,None];dx=(shape[1]*cell_size_m*t)[None,:]
    return dy,dx,weights[:,None]*weights[None,:]


def approximate_grid_mean_correlation(spec,quadrature):
    """Continuous-rectangle approximation used only during parameter fitting."""
    dy,dx,weight=quadrature
    # Joint reversal is a symmetry, separate coordinate reversal need not be.
    return float(2*np.sum(weight*(spec.correlation(dy,dx)+spec.correlation(dy,-dx))))


def grid_mean_correlation(spec,shape,cell_size_m,chunk_rows=64):
    """Exact discrete q = sum_ij rho(x_i-x_j)/N², without an N² matrix."""
    h,w=shape
    columns=np.arange(1-w,w);cw=w-np.abs(columns)
    result=0.
    for start in range(0,h,chunk_rows):
        rows=np.arange(start,min(start+chunk_rows,h))
        rw=(h-rows)*np.where(rows==0,1,2)
        rho=spec.correlation(rows[:,None]*cell_size_m,columns[None,:]*cell_size_m)
        result+=float(np.sum(rho*rw[:,None]*cw[None,:]))
    q=result/(h*w)**2
    if not 0<q<1:
        raise ArithmeticError("grid-average correlation must lie in (0,1)")
    return q


def canonical_spec(spec):
    row,col,angle=spec.ell_row_m,spec.ell_col_m,spec.angle_deg
    if col<row:
        row,col=col,row;angle+=90.
    angle=(angle+90.)%180.-90.
    return GSToolsReferenceSpec(spec.family,float(row),float(col),sigma_R=spec.sigma_R,angle_deg=float(angle))


def fitted_curves(spec,lags,cell):
    return {name:spec.sigma_R**2*(1-spec.correlation(np.array(lags)*dr*cell,np.array(lags)*dc*cell))
            for name,(dr,dc) in DIRECTIONS.items()}


def fit_profile(logs,*,cell_size_m,lags_cells,length_bounds_m,quadrature_order=48):
    """Fit one explicitly assumed covariance family to map structure only.

    Equal weights for log semivariance at each declared direction/lag balance
    short and long scales. Correlated pixel pairs do not count as iid evidence.
    """
    logs=np.asarray(logs,dtype=float)
    curves=variogram_curves(logs,lags_cells)
    if any(np.any(v<=0) for v in curves.values()) or np.var(logs)<=0:
        raise ValueError("positive directional increments and spatial variance required")
    variance=float(np.var(logs));mean=float(np.mean(logs))
    quad=rectangle_quadrature(logs.shape,cell_size_m,quadrature_order)
    empirical=np.concatenate(list(curves.values()))
    def parameters(z):
        row,col=np.exp(z[:2]);angle=float(np.rad2deg(z[2]))
        unit=GSToolsReferenceSpec("matern32_radial",row,col,sigma_R=1.,angle_deg=angle)
        q=approximate_grid_mean_correlation(unit,quad)
        if not 0<q<1:
            raise ArithmeticError("invalid rectangle covariance integral")
        return GSToolsReferenceSpec(unit.family,row,col,sigma_R=np.sqrt(variance/(1-q)),angle_deg=angle)
    def residual(z):
        values=np.concatenate(list(fitted_curves(parameters(z),lags_cells,cell_size_m).values()))
        if np.any(values<=0):
            raise ArithmeticError("nonpositive fitted semivariance")
        return np.log(values)-np.log(empirical)
    gradients=[np.sqrt(2*curves[k][0])/cell_size_m for k in ("row","column")]
    guess=np.sqrt(3*variance)*1.05/np.array(gradients)
    bounds=([np.log(length_bounds_m[0])]*2+[-np.pi/2],[np.log(length_bounds_m[1])]*2+[np.pi/2])
    attempts=[];successful=[]
    for angle in [0.,np.pi/4,-np.pi/4]:
        initial=[*np.log(np.clip(guess,*length_bounds_m)),angle]
        fit=least_squares(residual,initial,bounds=bounds,max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        attempts.append({"initial_angle_deg":float(np.rad2deg(angle)),"success":bool(fit.success),
            "cost":float(fit.cost),"evaluations":fit.nfev,"message":fit.message})
        if fit.success and np.isfinite(fit.cost):successful.append(fit)
    if not successful:
        raise ValueError("no structural fit converged")
    best=min(successful,key=lambda z:z.cost);spec=canonical_spec(parameters(best.x))
    qapprox=approximate_grid_mean_correlation(spec,quad)
    qexact=grid_mean_correlation(spec,logs.shape,cell_size_m)
    spec=GSToolsReferenceSpec(spec.family,spec.ell_row_m,spec.ell_col_m,
        sigma_R=float(np.sqrt(variance/(1-qexact))),angle_deg=spec.angle_deg)
    predicted=fitted_curves(spec,lags_cells,cell_size_m)
    return {"law":{"mean_log10_k":mean,"covariance":asdict(spec)},
        "spatial_variance_log10":variance,"spatial_sd_log10":float(np.sqrt(variance)),
        "pointwise_sd_log10":spec.sigma_R,"grid_mean_unit_variance":qexact,
        "quadrature_grid_mean_unit_variance":qapprox,
        "expected_grid_mean_variance_log10":spec.sigma_R**2*qexact,
        "expected_grid_spatial_variance_log10":spec.sigma_R**2*(1-qexact),
        "empirical_variograms":{k:v.tolist() for k,v in curves.items()},
        "fitted_variograms":{k:v.tolist() for k,v in predicted.items()},
        "log_variogram_rmse":float(np.sqrt(np.mean((np.log(np.concatenate(list(predicted.values())))-np.log(empirical))**2))),
        "maximum_multiplicative_variogram_mismatch":float(np.exp(np.max(np.abs(np.log(np.concatenate(list(predicted.values()))/empirical))))),
        "optimization_attempts":attempts,
        "length_at_bound":bool(np.any(np.minimum(best.x[:2]-np.array(bounds[0][:2]),np.array(bounds[1][:2])-best.x[:2])<.01)),
        "interpretation":"stationary synthetic-map structure fit; total heterogeneity, not reconstruction-error or instrument-noise calibration"}


def structural_features(logs,*,cell_size_m,mean_log10,sigma_log10,stride=8):
    """Full-map moments/gradients and explicitly coarser connectivity checks."""
    a=np.asarray(logs,dtype=float)
    if a.ndim!=2 or min(a.shape)<3 or not np.all(np.isfinite(a)) or stride<1:
        raise ValueError("finite native log map and positive diagnostic stride required")
    result={"spatial_mean_log10":float(a.mean()),"spatial_variance_log10":float(a.var()),
        "q05_log10":float(np.quantile(a,.05)),"q50_log10":float(np.quantile(a,.5)),"q95_log10":float(np.quantile(a,.95)),
        "gradient_rms_row_log10_per_m":float(np.sqrt(np.mean(np.diff(a,axis=0)**2))/cell_size_m),
        "gradient_rms_column_log10_per_m":float(np.sqrt(np.mean(np.diff(a,axis=1)**2))/cell_size_m)}
    sample=a[::stride,::stride]
    for direction,mask in [("high",sample>mean_log10+ndtri(.9)*sigma_log10),
                           ("low",sample<mean_log10-ndtri(.9)*sigma_log10)]:
        components,count=label(mask)  # four-neighbour connectivity
        areas=np.bincount(components.ravel());areas[0]=0
        row_span=bool(set(components[0])&set(components[-1])-{0})
        col_span=bool(set(components[:,0])&set(components[:,-1])-{0})
        result[f"{direction}_fraction"]=float(mask.mean())
        result[f"{direction}_largest_component_fraction"]=float(areas.max()/mask.size)
        result[f"{direction}_spans_rows"]=row_span
        result[f"{direction}_spans_columns"]=col_span
    return result
