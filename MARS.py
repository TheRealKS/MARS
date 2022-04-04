import math
import sys

import logging

from BaseFunction import BaseFunction, HingeFunctionBaseFunction, HingeFunctionProductBaseFunction
from MARSModel import MARSModel, MARSModelTerm, Operator

import statsmodels.api as sm

import numpy as np

MAXVAL = sys.maxsize
nonzero_store = dict()

def runMARSForward(X, y, labels, n, maxSplits):
    model = MARSModel()
    logging.warning("Starting iteration with " + str(n) + "vars..")

    lof_ = MAXVAL
    for M in range(1, maxSplits, 2):
        logging.warning("Split " + str(M))
        m_ = None
        v_ = None
        t_ = None
        model.getRegressable(X)
        for m in range(0, M):  # For all existing terms
            logging.warning("Term " + str(m))
            basefunc = model.get_component(m).get_function()
            varsnotinbasefunc = set(np.arange(n)).difference(basefunc.getVariables())
            for v in varsnotinbasefunc:  # For all suitable vars
                #logging.warning("Var " + labels[v])
                if m != 0:
                    ts = check_nonzero(basefunc, X, v)
                else:
                    ts = X[:, v]
                its = 0
                for t in ts:  # For all suitable values of vars
                    prodpos = HingeFunctionBaseFunction(t, v, labels[v], True)
                    prodneg = HingeFunctionBaseFunction(t, v, labels[v], False)
                    if m != 0:
                        prodpos = HingeFunctionProductBaseFunction([basefunc, prodpos])
                        prodneg = HingeFunctionProductBaseFunction([basefunc, prodneg])

                    #logging.warning(
                    #    "Running regression for M=" + str(M) + ", m=" + str(m) + ", v=" + labels[v] + ", value " +
                    #    str(its) + "...")

                    # Evaluate using SSE
                    reg = model.getRegressableNewComponents(X, [prodpos, prodneg])
                    lof, _ = ssr(reg, y)
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
    reg = model.getRegressable(X)
    _, b = ssr(reg, y)
    model.set_coefficients(b)
    return model, lof_


def runMARSBackward(model: MARSModel, modelrss, X, y, n, maxSplits, d = 3):
    logging.warning("Running backward MARS now...")
    J = model
    K = model.copy()
    currentgcv = GCV(modelrss, n, model.length(), d)
    print(currentgcv)
    for M in reversed(range(1, maxSplits)):
        lof_ = MAXVAL
        L = K.copy()
        for m in range(1, M):
            KK = L.copy()
            KK.remove_component(m)
            lof, b = ssr(KK.getRegressable(X), y)
            submodelgcv = GCV(lof, n, KK.length(), d)
            if submodelgcv < lof_:
                lof_ = submodelgcv
                K = KK.copy()
            if submodelgcv < currentgcv:
                print(submodelgcv)
                currentgcv = submodelgcv
                J = KK.copy()
                J.set_coefficients(b)

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

def ssr(reg, y):
    b = np.linalg.pinv(reg).dot(y)
    b[0] = 1.0
    pred = reg.dot(b)
    lof = np.sum(np.power(pred - y, 2))
    return lof, b

def GCV(ssr, n, M, d):
    effectiveparams = M + d * (M - 1)
    t = (1 - (effectiveparams / n))
    gcv = (ssr / n) / math.pow(t, 2)
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
