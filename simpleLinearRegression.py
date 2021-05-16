import numpy as np
import patsy
from scipy.optimize import minimize
from joblib import Parallel, delayed


def ll_norm(betas, yX):
  """
  :func ll_norm: computes the mean squared error - mean of difference between acutal y and y_hat.
    y_hat is computed with the help of betas.

  :param betas: contains intercept_hat and slope_hat
  :param yX: dataset
  :return: the mean squared difference
  """
  y = yX[:, 0]
  x = yX[:, 1:]
  N = y.shape[0]
  mu = np.full(N, np.nan)
  for n in range(N):
    mu[n] = np.sum(x[n, :] * betas)
  d = y - mu
  return np.sum(d * d)

def optim(data, initval = None):
  """
  :func optim: Computes estimate of y-intercept and slope with the help of minimize and ll_norm
    method.

  :param data: data - contains dataset
  :param initval: initial guess of y-intercept and slope can be passed - optional
  :return: estaimate of y-intercept and slope
  """
  if not initval:
    k = data.shape[1] - 1
    initval = np.random.normal(size = k)
  return minimize(ll_norm, initval, args=(data), method="BFGS")['x']

def boot(data, N, func, initval = None):
  """
  :func boot: Calls minimize and ll_norm methods to get estimates of y-intercept and slope

  :param data: data - contains dataset
  :param N: size of the dataset
  :param func: optim - function to get estimate of y-intercept and slope
  :param initval: initial guess of y-intercept and slope can be passed - optional
  :return: estimates of y-intercept and slope
  """
  idx = np.random.choice(N, N)
  return func(data[idx, :], initval)

def pbootstrap(data, R, fun, initval = None, ncpus = 1):
  """
  :func pbootstrap: Calls boot method for R iteration in parallel and gets estimates of y-intercept
    and slope

  :param data: data - contains dataset
  :param R: number of iterations
  :param func: optim - function to get estimate of y-intercept and slope
  :param initval: initial guess of y-intercept and slope can be passed - optional
  :param ncpus: number of physical cores to run the pbootstrap method - optional
  :return: estimates of y-intercept and slope
  """
  N = data.shape[0]
  thetas = Parallel(ncpus) (delayed(boot) (data, N, fun, initval) for _ in range(R))
  return np.asarray(thetas)

def bootstrap(data, R, func, initval = None):
  """
  :func bootstrap: Computes muhats (estimates of y-intercept and slope) on sample datasets.
    In bootstrapping sample datasets are created using random sampling with replecement.

  :param data: data - contains dataset
  :param R: number of iterations
  :param func: optim - function to get estimate of y-intercept and slope
  :param initval: initial guess of y-intercept and slope can be passed - optional
  :return: estimates of y-intercept and slope
  """
  N, k = data.shape
  k -= 1
  muhats = np.full((R, k), np.nan)
  for r in range(R):
    idx = np.random.choice(N, N, replace=True)
    muhats[r] = func(data[idx, :], initval)
  return muhats

def adjustedR2(Y, mu, k):
  """
  :func adjustedR2: Computes the adjusted r square

  :param Y: Y dataframe
  :param mu: estimates of Y
  :param k: shape of number of x features
  :return: adjusted r square
  """
  N = Y.shape[0]
  error = Y - mu
  adjusted_r_square = 1 - (np.var(error) / np.var(Y) / np.var(Y)) * (N - 1) / (N - k)
  return adjusted_r_square

def SLR(df, iterations, confidence_percentage = 90, ncpus = 1):
  """
  :func SLR: Main function. Calls bootstrap or pbootstrap method to get muhats back.
    Computes the confidence interval on given confidence percentage. Computes estimated y_intercept
    and slope.

  :param df: data frame containing x and y in order
  :param iterations: iterations to pass on to bootstrap or pbootstrap
  :param confidence_percentage: confidence percentage - optional
  :param ncpus: number of physical cores to run the pbootstrap method - optional
  :return: [y_intercept, slope] and confidence interval
  """
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
