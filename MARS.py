import sys

from BaseFunction import BaseFunction, HingeFunctionBaseFunction
from MARSModel import MARSModel

from sklearn.linear_model import LinearRegression
import numpy as np

MAXVAL = sys.maxsize


def runMARS(self, X, n, r, maxSplits):
    model = MARSModel(computeIntercept(X))

    for M in range(1, maxSplits, 2):
        lof_ = MAXVAL
        m_ = None
        v_ = None
        t_ = None
        for m in range(1, M - 1):  # For all existing terms
            for v in range(1, n):  # For all vars
                rows = check_nonzero(model.get_component(m), X[v], r)
                for j in rows:  # For all values of current var where basis func evals positive
                    t = X[v][j]

                    reg = LinearRegression().fit()

                    # g = 0  # Start summing
                    # for k in range(1, M):
                    #     # a_i * B_i(x)
                    #     a = model.get_component(k)
                    #     term1 = a.get_coefficient() * a.get_function().getvalue(t)
                    #     #a_M * B_m(x) * max(0, x_v - t)
                    #     b = model.get_component(M)
                    #     c = model.get_component(m)
                    #     d = HingeFunctionBaseFunction(t, v, True)
                    #     term2 = b.get_coefficient() * c.get_function().getvalue(t) * d.getvalue(t)
                    #     #a_M+1 * B_m(x) * max(0, t - x_v)
                    #     e = model.get_component(M+1)
                    #     f = HingeFunctionBaseFunction(t, v, False)
                    #     term3 = e.get_coefficient() * c.get_function().getvalue(t) * f.getvalue(t)
                    #     g += (term1 + term2 + term3)
                    # #Compute LOF
                    # lof = LOF(g)
                    # if (lof < lof_):
                    #     lof_ = lof
                    #     m_ = m
                    #     v_ = v
                    #     t_ = t

            #Add new hinge function
            comp1 = model.get_component(m_)



def LOF(g):
    return 0

def check_nonzero(func: BaseFunction, v, r):
    indices = []
    for i in range(0, r):
        if func.getvalue(v[i]) > 0:
            indices.append(i)

    return indices


def computeIntercept(X):
    return 0.0
