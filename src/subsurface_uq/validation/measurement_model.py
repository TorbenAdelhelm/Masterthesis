"""Measurement-only spatial validation and Gaussian observation models.

No TIFF, reference field or LGCNN output enters fitting. A fitted nugget is an
unresolved observation-scale residual; its allocation to instrument error,
microscale geology or support mismatch is NOT identified by this model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from ..sampling.calibration import calibrate_point_covariance_candidates, correlation_for_offsets
from ..sampling.exact_kriging import exact_simple_kriging_predict_points
from ..sampling.measurement_density import gaussian_crps
from .reconstruction import predictive_scores


MODELS = ("iid_normal", "legacy_matern32", "legacy_exponential", "legacy_radial_exponential",
          "reml_exponential_constant", "reml_exponential_linear",
          "reml_matern32_constant", "reml_matern32_linear")


def _xy(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("nonempty finite XY coordinates required")
    return x


def _data(x, y):
    x = _xy(x); y = np.asarray(y, dtype=float)
    if y.shape != (len(x),) or len(x) < 10 or not np.all(np.isfinite(y)):
        raise ValueError("at least ten finite aligned log10 measurements required")
    return x, y


def block_coordinates(x, size):
    x = _xy(x)
    if not np.isfinite(size) or size <= 0:
        raise ValueError("positive block size required")
    # A global projected-grid origin is stable across folds/subsets.
    return np.floor(x/size).astype(np.int64)


def buffered_training_mask(x, train, test, *, block_size_m, buffer_m):
    """Exclude entire expanded test-block rectangles, without reading values."""
    x = _xy(x); train = np.asarray(train, dtype=bool).copy(); test = np.asarray(test, dtype=bool)
    if train.shape != (len(x),) or test.shape != train.shape or np.any(train & test) or not np.any(test):
        raise ValueError("disjoint aligned train/test masks and nonempty test required")
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("nonnegative buffer required")
    blocks = np.unique(block_coordinates(x[test],block_size_m),axis=0)
    for block in blocks:
        lower = block*block_size_m-buffer_m; upper = (block+1)*block_size_m+buffer_m
        train &= ~np.all((x >= lower) & (x < upper),axis=1)
    return train


def spatial_partition(x, *, block_size_m, test_fraction, buffer_m, seed):
    """Freeze a random set of occupied test blocks using geometry only."""
    x = _xy(x)
    if not 0 < test_fraction < .5:
        raise ValueError("test fraction must lie in (0,.5)")
    blocks, inverse = np.unique(block_coordinates(x,block_size_m),axis=0,return_inverse=True)
    if len(blocks) < 8:
        raise ValueError("at least eight occupied blocks required")
    n = int(np.ceil(len(blocks)*test_fraction))
    chosen = np.random.default_rng(seed).choice(len(blocks),n,replace=False)
    test = np.isin(inverse,chosen)
    train = buffered_training_mask(x,~test,test,block_size_m=block_size_m,buffer_m=buffer_m)
    if train.sum() < 30 or test.sum() < 10:
        raise ValueError("insufficient observations in frozen geographic partition")
    return {"development":train,"test":test,"buffer_excluded":~(train|test),
            "block_coordinates":block_coordinates(x,block_size_m),"selected_test_blocks":blocks[chosen]}


def balanced_block_folds(x, *, size, folds, seed):
    blocks,inverse,counts = np.unique(block_coordinates(x,size),axis=0,return_inverse=True,return_counts=True)
    if len(blocks)<folds or folds<2:
        raise ValueError("enough occupied blocks for >=2 folds required")
    order = sorted(np.random.default_rng(seed).permutation(len(blocks)),key=lambda i:-counts[i])
    sizes = np.zeros(folds,dtype=int); labels = np.zeros(len(blocks),dtype=int)
    for i in order:
        fold = int(np.argmin(sizes)); labels[i]=fold; sizes[fold]+=counts[i]
    return labels[inverse]


def correlation(distance, length, family):
    r = distance/length
    if family == "exponential":
        return np.exp(-r)
    if family == "matern32":
        r = np.sqrt(3.)*r
        return (1+r)*np.exp(-r)
    raise ValueError("unknown radial family")


def design_matrix(x, center, scale, trend):
    if trend == "constant":
        return np.ones((len(x),1))
    if trend == "linear":
        return np.column_stack((np.ones(len(x)),(x-center)/scale))
    raise ValueError("unknown mean model")


def reml_objective(parameters, distance, y, h, family):
    """REML negative log likelihood and exact derivatives in log parameters."""
    length, sigma, tau = np.exp(parameters)
    rho = correlation(distance,length,family)
    k = sigma*sigma*rho + tau*tau*np.eye(len(y))
    chol = cho_factor(k,lower=True,check_finite=False)
    ki = cho_solve(chol,np.eye(len(y)),check_finite=False)
    kih = ki@h
    info = h.T@kih; factor = cho_factor(info,lower=True,check_finite=False)
    beta = cho_solve(factor,h.T@(ki@y),check_finite=False)
    residual = y-h@beta; alpha = ki@residual
    p = ki-kih@cho_solve(factor,kih.T,check_finite=False)
    value = .5*(residual@alpha+2*np.log(np.diag(chol[0])).sum()
                +2*np.log(np.diag(factor[0])).sum()+(len(y)-h.shape[1])*np.log(2*np.pi))
    r = distance/length
    derivative_length = (r*np.exp(-r) if family == "exponential" else 3*r*r*np.exp(-np.sqrt(3.)*r))
    derivatives = [sigma*sigma*derivative_length,2*sigma*sigma*rho,2*tau*tau*np.eye(len(y))]
    score = p-np.outer(alpha,alpha)
    gradient = np.array([.5*np.sum(score*dk) for dk in derivatives])
    return float(value),gradient


@dataclass
class MeasurementModel:
    name: str
    x: np.ndarray
    y: np.ndarray
    parameters: dict

    def predict(self, queries):
        queries = _xy(queries); p = self.parameters
        if self.name == "iid_normal":
            mean = np.full(len(queries),p["mean_log10"])
            return mean,np.full(len(queries),p["predictive_variance_log10"]),None
        if self.name.startswith("legacy_"):
            args = dict(observation_coordinates_xy_m=self.x,observation_log10_k=self.y,
                query_coordinates_xy_m=queries,covariance_model=p["covariance_model"],
                mean_log10_k=p["mean_log10_k"],std_log10_k=p["std_log10_k"],
                length_scale_y_m=p["length_scale_y_m"],length_scale_x_m=p["length_scale_x_m"],
                structured_std_log10_k=p["structured_std_log10_k"],nugget_std_log10_k=p["nugget_std_log10_k"])
            result = exact_simple_kriging_predict_points(**args,include_query_nugget=False)
            latent = result.variance_log10_k
            return result.mean_log10_k,latent+p["nugget_std_log10_k"]**2,latent
        h = design_matrix(self.x,np.array(p["coordinate_center"]),p["coordinate_scale_m"],p["trend"])
        hq = design_matrix(queries,np.array(p["coordinate_center"]),p["coordinate_scale_m"],p["trend"])
        k = p["structured_std_log10"]**2*correlation(cdist(self.x,self.x),p["length_m"],p["family"])
        k += p["nugget_std_log10"]**2*np.eye(len(self.x))
        chol = cho_factor(k,lower=True,check_finite=False)
        kih = cho_solve(chol,h,check_finite=False)
        info = cho_factor(h.T@kih,lower=True,check_finite=False)
        beta = cho_solve(info,h.T@cho_solve(chol,self.y,check_finite=False),check_finite=False)
        cross = p["structured_std_log10"]**2*correlation(cdist(queries,self.x),p["length_m"],p["family"])
        mean = hq@beta + cross@cho_solve(chol,self.y-h@beta,check_finite=False)
        r = hq-cross@kih
        latent = (p["structured_std_log10"]**2-np.sum(cross*cho_solve(chol,cross.T,check_finite=False).T,axis=1)
                  +np.sum(r*cho_solve(info,r.T,check_finite=False).T,axis=1))
        if latent.min(initial=0.) < -1.e-8:
            raise ArithmeticError("negative universal-kriging variance")
        latent = np.maximum(latent,0.)
        return mean,latent+p["nugget_std_log10"]**2,latent


def fit_model(name,x,y,settings):
    """Fit using precisely the supplied training coordinates/readings."""
    x,y = _data(x,y)
    if name not in MODELS:
        raise ValueError("unknown measurement model")
    if name == "iid_normal":
        variance = float(np.var(y,ddof=1))
        if variance <= 0:
            raise ValueError("positive training variance required")
        return MeasurementModel(name,x,y,{"mean_log10":float(y.mean()),"sample_variance_log10":variance,
            "predictive_variance_log10":variance*(1+1/len(y)),"spatial_information":False})
    if name.startswith("legacy_"):
        model = name.removeprefix("legacy_")
        fits,_,_ = calibrate_point_covariance_candidates(x,10.**y,models=[model],**settings["variogram"])
        return MeasurementModel(name,x,y,{**fits[0].to_dict(),"mean_parameter_uncertainty_included":False,
            "covariance_parameter_uncertainty_included":False,"model_target":"new observation including fitted nugget"})
    _,family,trend = name.split("_")
    center = x.mean(axis=0); scale = 10000.
    h = design_matrix(x,center,scale,trend)
    if np.linalg.matrix_rank(h) != h.shape[1]:
        raise ValueError("training mean design lacks rank")
    distance = cdist(x,x)
    s = settings["reml"]
    bounds = np.log([s["length_bounds_m"],s["structured_std_bounds_log10"],s["nugget_std_bounds_log10"]])
    std = float(np.std(y,ddof=1)); attempts=[]; successes=[]
    for length in s["initial_lengths_m"]:
        initial = np.clip(np.log([length,max(std*.8,1.e-3),max(std*.5,1.e-3)]),bounds[:,0],bounds[:,1])
        fit = minimize(reml_objective,initial,args=(distance,y,h,family),jac=True,method="L-BFGS-B",
                       bounds=bounds,options={"maxiter":s["maxiter"],"ftol":1.e-10,"gtol":1.e-6})
        attempts.append({"initial_length_m":length,"success":bool(fit.success),"message":str(fit.message),
                         "objective":float(fit.fun),"iterations":int(fit.nit)})
        if fit.success and np.isfinite(fit.fun):
            successes.append(fit)
    if not successes:
        raise ValueError(f"no REML optimization converged: {attempts}")
    best = min(successes,key=lambda f:f.fun)
    length,sigma,tau = np.exp(best.x)
    k = sigma*sigma*correlation(distance,length,family)+tau*tau*np.eye(len(y))
    chol = cho_factor(k,lower=True); kih = cho_solve(chol,h)
    info = cho_factor(h.T@kih,lower=True)
    beta = cho_solve(info,h.T@cho_solve(chol,y))
    return MeasurementModel(name,x,y,{"family":family,"trend":trend,"length_m":float(length),
        "structured_std_log10":float(sigma),"nugget_std_log10":float(tau),
        "nugget_fraction":float(tau*tau/(sigma*sigma+tau*tau)),"beta":beta.tolist(),
        "coordinate_center":center.tolist(),"coordinate_scale_m":scale,
        "objective":float(best.fun),"optimization_attempts":attempts,
        "parameter_at_bound":bool(np.any(np.minimum(best.x-bounds[:,0],bounds[:,1]-best.x)<.01)),
        "mean_parameter_uncertainty_included":True,"covariance_parameter_uncertainty_included":False,
        "nugget_interpretation":"unresolved observation-scale residual; not identified instrument noise or 5 m geological variance"})


def spatial_scores(y,mean,std,blocks):
    blocks = np.asarray(blocks)
    if blocks.ndim == 2:
        _,blocks = np.unique(blocks,axis=0,return_inverse=True)
    rows=[]
    for b in np.unique(blocks):
        selected = blocks == b
        rows.append({"block_id":int(b),**predictive_scores(y[selected],mean[selected],std[selected])})
    return {"point_weighted":predictive_scores(y,mean,std),"blocks":rows,
            "equal_block_mean_crps_log10":float(np.mean([r["mean_crps_log10"] for r in rows])),
            "equal_block_coverage90":float(np.mean([r["reliability"]["0.9"] for r in rows]))}


def development_cv(x,y,settings,*,on_fold=None):
    """All model selection uses development data only; buffers use geometry."""
    x,y = _data(x,y); cv = settings["development"]
    assignment = balanced_block_folds(x,size=cv["block_size_m"],folds=cv["folds"],seed=cv["seed"])
    predictions = {name:{"mean":np.full(len(x),np.nan),"variance":np.full(len(x),np.nan)} for name in settings["models"]}
    folds=[]; failures={}
    for fold in range(cv["folds"]):
        test = assignment == fold
        train = buffered_training_mask(x,~test,test,block_size_m=cv["block_size_m"],buffer_m=cv["buffer_m"])
        if train.sum()<20:
            raise ValueError("insufficient buffered development training data")
        for name in settings["models"]:
            if name in failures:
                continue
            try:
                model = fit_model(name,x[train],y[train],settings)
                mean,variance,_ = model.predict(x[test])
                if np.any(variance<=0) or not np.all(np.isfinite([mean,variance])):
                    raise ArithmeticError("invalid held-out prediction")
                predictions[name]["mean"][test] = mean; predictions[name]["variance"][test] = variance
                record={"fold":fold,"model":name,"train_indices":np.flatnonzero(train).tolist(),
                    "test_indices":np.flatnonzero(test).tolist(),"buffer_excluded_indices":np.flatnonzero(~(train|test)).tolist(),
                    "parameters":model.parameters,"scores":predictive_scores(y[test],mean,np.sqrt(variance))}
                folds.append(record)
                if on_fold is not None:
                    on_fold(record)
            except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                failures[name]={"fold":fold,"error":str(exc)}
                if on_fold is not None:
                    on_fold({"fold":fold,"model":name,"failure":str(exc)})
    blocks = block_coordinates(x,cv["block_size_m"])
    scores={name:spatial_scores(y,p["mean"],np.sqrt(p["variance"]),blocks)
            for name,p in predictions.items() if name not in failures}
    eligible=[name for name in scores if name != "iid_normal"]
    legacy=[name for name in eligible if name.startswith("legacy_")]
    if not eligible or not legacy or "iid_normal" not in scores:
        raise ValueError(f"required candidate/baseline fits unavailable: {failures}")
    # The legacy procedure selects by point RMSE. The uncertainty-oriented
    # procedure selects by equal-block CRPS, before any outer test scoring.
    selected = min(eligible,key=lambda n:scores[n]["equal_block_mean_crps_log10"])
    selected_legacy = min(legacy,key=lambda n:scores[n]["point_weighted"]["rmse_log10"])
    return {"selected_model":selected,"selected_legacy":selected_legacy,"scores":scores,
            "failures":failures,"fold_records":folds,"fold_assignment":assignment.tolist(),
            "predictions":{name:{k:v.tolist() for k,v in p.items()} for name,p in predictions.items() if name not in failures}}


def paired_block_bootstrap(y,mean,std,baseline_mean,baseline_std,blocks,*,replicates,seed):
    """Exploratory paired resampling of occupied spatial blocks, not pixels."""
    blocks = np.asarray(blocks)
    _,labels = np.unique(blocks,axis=0,return_inverse=True)
    delta = gaussian_crps(mean,std,y)-gaussian_crps(baseline_mean,baseline_std,y)
    differences = np.array([delta[labels==i].mean() for i in np.unique(labels)])
    rng = np.random.default_rng(seed)
    estimates = differences[rng.integers(len(differences),size=(replicates,len(differences)))].mean(axis=1)
    return {"mean_block_crps_difference":float(differences.mean()),"ci95":np.quantile(estimates,[.025,.975]).tolist(),
            "n_blocks":len(differences),"replicates":replicates,
            "interpretation":"exploratory occupied-block bootstrap; between-block dependence and fitted-model uncertainty are not fully resolved"}
