import numpy as np
import patsy
from scipy.optimize import minimize
from joblib import Parallel, delayed

def ll_norm(betas, yX):
  y = yX[:, 0]
  x = yX[:, 1:]
  N = y.shape[0]
  mu = np.full(N, np.nan)
  for n in range(N):
    mu[n] = np.sum(x[n, :] * betas)
  d = y - mu
  return np.sum(d * d)

def optim(data, initval = None):
  if not initval:
    k = data.shape[1] - 1
    initval = np.random.normal(size = k)
  return minimize(ll_norm, initval, args=(data), method="BFGS")['x']

def boot(data, N, fun, initval):
  idx = np.random.choice(N, N)
  return fun(data[idx, :], initval)

def pbootstrap(data, R, fun, initval = None, ncpus = 1):
  N = data.shape[0]
  thetas = Parallel(ncpus) (delayed(boot) (data, N, fun, initval) for _ in range(R))
  return np.asarray(thetas)

def optim(data, initval = None):
  k = data.shape[1] - 1
  if not np.any(initval):
    initval = np.random.normal(size=k)
  return minimize(ll_norm, initval, args=(data), method="BFGS")['x']

def bootstrap(data, R, func, initval = None):
  N, k = data.shape
  k -= 1
  muhats = np.full((R, k), np.nan)
  for r in range(R):
    idx = np.random.choice(N, N, replace=True)
    muhats[r] = func(data[idx, :], initval)
  return muhats

def adjustedR2(Y, mu, k):
  N = Y.shape[0]
  error = Y - mu
  adjusted_r_square = 1 - (np.var(error) / np.var(Y) / np.var(Y)) * (N - 1) / (N - k)
  return adjusted_r_square

def SLR(df, iterations, confidence_percentage = 90, ncpus = 1):
  if df.columns.size != 2:
    return "invalid df"
  X = patsy.dmatrix(df.columns[0], data = df)
  yX = np.c_[df.iloc[:, 1], X]
  muhats = []
  if ncpus > 1:
    muhats = pbootstrap(yX, iterations, optim, ncpus = ncpus)
  else:
    muhats = bootstrap(yX, iterations, optim)
  start = (100 - confidence_percentage) / 2
  end = 100 - start
  confidence_interval = np.percentile(muhats, [start, end], axis = 0).T
  y_intercept = muhats[:, 0].mean()
  slope = muhats[:, 1].mean()
  return np.array([y_intercept, slope]), confidence_interval
