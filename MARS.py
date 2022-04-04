import math
import sys

import logging

from BaseFunction import BaseFunction, HingeFunctionBaseFunction, HingeFunctionProductBaseFunction
from MARSModel import MARSModel, MARSModelTerm, Operator

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.linear_model import LinearRegression

import numpy as np

MAXVAL = sys.maxsize
nonzero_store = dict()

def runMARSForward(X, y, labels, n, maxSplits):
    model = MARSModel()
    logging.warning("Starting iteration with " + str(n) + "vars..")

    for M in range(1, maxSplits, 2):
        logging.warning("Split " + str(M))
        lof_ = MAXVAL
        m_ = None
        v_ = None
        t_ = None
        model.getRegressable(X)
        for m in range(0, M):  # For all existing terms
            logging.warning("Term " + str(m))
            basefunc = model.get_component(m).get_function()
            varsnotinbasefunc = set(np.arange(n)).difference(basefunc.getVariables())
            for v in varsnotinbasefunc:  # For all suitable vars
                logging.warning("Var " + labels[v])
                if (m != 0):
                    ts = check_nonzero(basefunc, X, v)
                else:
                    ts = X[:][v]
                its = 0
                for t in ts:  # For all suitable values of vars
                    prodpos = HingeFunctionBaseFunction(t, v, labels[v], True)
                    prodneg = HingeFunctionBaseFunction(t, v, labels[v], False)
                    if m != 0:
                        prodpos = HingeFunctionProductBaseFunction([basefunc, prodpos])
                        prodneg = HingeFunctionProductBaseFunction([basefunc, prodneg])

                    logging.warning(
                        "Running regression for M=" + str(M) + ", m=" + str(m) + ", v=" + labels[v] + ", value " +
                        str(its) + "...")

                    # Evaluate using SSE
                    reg = model.getRegressableNewComponents(X, [prodpos, prodneg])
                    lr = pinv
                    lof = ((y - lr.predict(reg)**2)).sum()
                    if lof < lof_:
                        lof_ = lof
                        m_ = m
                        v_ = v
                        t_ = t
                    its += 1

        # Add new terms to model
        basefunc = model.get_component(m_).get_function()
        pos_, neg_ = generateCandidatePairs(basefunc, t_, v_, labels[v_])
        model.add_component(pos_)
        model.add_component(neg_)

    # Compute the final coefficients
    g = sm.OLS(y, model.getRegressable(X)).fit()
    model.set_coefficients(g.params)
    return model, g.ssr


def runMARSBackward(model: MARSModel, modelrss, X, y, n, maxSplits, d = 3):
    J = model
    K = model.copy()
    currentgcv = GCV(modelrss, n, model.length(), d)
    print(currentgcv)
    for M in reversed(range(1, maxSplits)):
        b = MAXVAL
        L = K.copy()
        for m in range(1, M):
            KK = L.copy()
            KK.remove_component(m)
            g = sm.OLS(y, KK.getRegressable(X)).fit()
            submodelgcv = GCV(g.ssr, n, KK.length(), d)
            print(submodelgcv)
            if submodelgcv < b:
                b = submodelgcv
                K = KK.copy()
            if submodelgcv < currentgcv:
                currentgcv = submodelgcv
                J = KK.copy()

    return J

def generateCandidatePairs(parent, t, v, label):
    prodpos = HingeFunctionBaseFunction(t, v, label, True)
    prodneg = HingeFunctionBaseFunction(t, v, label, False)
    if (parent.type > 1):
        prodpos = HingeFunctionProductBaseFunction([parent, prodpos])
        prodneg = HingeFunctionProductBaseFunction([parent, prodneg])

    posterm = MARSModelTerm(prodpos, 0.0, Operator.POS)
    negterm = MARSModelTerm(prodneg, 0.0, Operator.POS)

    return posterm, negterm

def GCV(ssr, n, M, d):
    effectiveparams = M + d * (M - 1) / 2
    t = (1 - (effectiveparams / n))
    gcv = ssr / (math.pow(t, 2))
    return gcv

def check_nonzero(func: BaseFunction, X, v):
    if (func, v) in nonzero_store:
        return nonzero_store[(func, v)]

    values = []

    varia = func.getVariables()
    for j in range(0, X.shape[0]):
        evalvalues = {}
        for var in varia:
            evalvalues[var] = X[j][var]

        if func.getvalue(evalvalues) != 0:
            values.append(X[j][v])

    nonzero_store[(func, v)] = values
    return values
