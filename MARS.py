import sys

from BaseFunction import BaseFunction, HingeFunctionBaseFunction
from MARSModel import MARSModel

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

MAXVAL = sys.maxsize


def runMARS(self, X, y, n, r, maxSplits):
    model = MARSModel(computeIntercept(X))

    for M in range(1, maxSplits, 2):
        lof_ = MAXVAL
        m_ = None
        v_ = None
        t_ = None
        G = np.zeros(M+1, r)
        for m in range(1, M - 1):  # For all existing terms
            for v in range(1, n):  # For all vars
                rows = check_nonzero(model.get_component(m), X[v], r)
                for j in rows:  # For all values of current var where basis func evals positive
                    t = X[v][j]
                    G[M] = [model.get_component(m).get_function().getvalue(X) * max(0, (k - t)) for k in X[v]]
                    G[M+1][t] = [model.get_component(m).get_function().getvalue(X) * max(0, (t - k)) for k in X[v]]

        #Regress
        g: LinearRegression = LinearRegression()
        g.fit(G, y)
        #Get coeffecients and compute squared-error loss for all sub-models
        beta = g.coef_
        y_hat = G * beta
        lof = mean_squared_error(y, y_hat)
        #Add to model


        comp1 =model.get_component(m_)



def LOF(g):
    return 0

def check_nonzero(func: BaseFunction, v, r):
    indices = []
    for i in range(0, r):
        if func.getvalue(v[i]) > 0:
            indices.append(i)

    return indices


def computeIntercept(X):
    return 1.0
