import math
import sys

import logging

from BaseFunction import BaseFunction, HingeFunctionBaseFunction, HingeFunctionProductBaseFunction
from MARSModel import MARSModel, MARSModelTerm, Operator

import statsmodels.api as sm

import numpy as np

MAXVAL = sys.maxsize
PENALTY = 3
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
        current_regressable = np.array(model.getRegressable(X))
        for m in range(0, M):  # For all existing terms
            logging.warning("Term " + str(m))
            basefunc = model.get_component(m).get_function()
            varsnotinbasefunc = set(np.arange(n)).difference(basefunc.getVariables())
            for v in varsnotinbasefunc:  # For all suitable vars
                logging.warning("Var " + labels[v])
                ts = check_nonzero(basefunc, X, v)
                its = 0
                for t in ts:  # For all suitable values of vars
                    newbasisFunctionPos = HingeFunctionBaseFunction(t, v, labels[v], True)
                    newbasisFunctionNeg = HingeFunctionBaseFunction(t, v, labels[v], False)
                    prodpos = HingeFunctionProductBaseFunction([basefunc, newbasisFunctionPos])
                    prodneg = HingeFunctionProductBaseFunction([basefunc, newbasisFunctionNeg])

                    #logging.info(
                    #    "Running regression for M=" + str(M) + ", m=" + str(m) + ", v=" + labels[v] + ", value " +
                    #    str(its) + "...")

                    # Evaluate using SSE
                    reg = np.array(model.getRegressableNewComponents(X, [prodpos, prodneg]))
                    newreg = np.concatenate((current_regressable, reg), axis=1)
                    g = sm.OLS(y, newreg).fit()
                    lof = g.ssr
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


def runMARSBackward(model: MARSModel, modelrss, X, y, n, maxSplits):
    J = model
    K = model.copy()
    currentgcv = GCV(modelrss, n, model.length())
    print(currentgcv)
    for M in reversed(range(1, maxSplits)):
        b = MAXVAL
        L = K.copy()
        for m in range(1, M):
            KK = L.copy()
            KK.remove_component(m)
            g = sm.OLS(y, KK.getRegressable(X)).fit()
            submodelgcv = GCV(g.ssr, n, KK.length())
            print(submodelgcv)
            if submodelgcv < b:
                b = submodelgcv
                K = KK.copy()
            if submodelgcv < currentgcv:
                currentgcv = submodelgcv
                J = KK.copy()

    return J

def generateCandidatePairs(parent, t, v, label):
    newbasisFunctionPos = HingeFunctionBaseFunction(t, v, label, True)
    newbasisFunctionNeg = HingeFunctionBaseFunction(t, v, label, False)
    prodpos = HingeFunctionProductBaseFunction([parent, newbasisFunctionPos])
    prodneg = HingeFunctionProductBaseFunction([parent, newbasisFunctionNeg])

    posterm = MARSModelTerm(prodpos, 0.0, Operator.POS)
    negterm = MARSModelTerm(prodneg, 0.0, Operator.POS)

    return posterm, negterm

def GCV(ssr, n, M):
    effectiveparams = M + PENALTY * (M - 1) / 2
    t = (1 - (effectiveparams / n))
    gcv = ssr / (n * math.pow(t, 2))
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
