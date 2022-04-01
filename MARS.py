import sys

import logging

from BaseFunction import BaseFunction, HingeFunctionBaseFunction, HingeFunctionProductBaseFunction
from MARSModel import MARSModel, MARSModelTerm, Operator

import statsmodels.api as sm

import numpy as np

MAXVAL = sys.maxsize


def runMARSForward(X, y, labels, n, maxSplits):
    model = MARSModel(1.0)
    logging.info("Starting iteration with " + str(n) + "vars..")

    for M in range(1, maxSplits + 1, 2):
        lof_ = MAXVAL
        m_ = None
        v_ = None
        t_ = None
        for m in range(0, M):  # For all existing terms
            basefunc = model.get_component(m).get_function()
            varsinbasefunc = basefunc.getVariables()
            allvars = set(np.arange(n))
            allvars = allvars.difference(varsinbasefunc)
            for v in allvars:  # For all suitable vars
                ts = check_nonzero(basefunc, X, v)
                its = 0
                for t in ts:  # For all suitable values of vars
                    # Generate candidate pairs and candidate model
                    p, n = generateCandidatePairs(basefunc, t, v, labels)
                    newmodel = model.copy()
                    newmodel.add_component(p)
                    newmodel.add_component(n)

                    logging.info(
                        "Running regression for M=" + str(M) + ", m=" + str(m) + ", v=" + labels[v] + ", value " +
                        str(its) + "...")

                    # Evaluate using SSE
                    reg = newmodel.getRegressable(X)
                    g = sm.OLS(y, reg).fit()
                    if g.ssr < lof_:
                        lof_ = g.ssr
                        m_ = m
                        v_ = v
                        t_ = t
                    its += 1

        # Add new terms to model
        basefunc = model.get_component(m_).get_function()
        p_, n_ = generateCandidatePairs(basefunc, t_, v_, labels[v_])
        model.add_component(p_)
        model.add_component(n_)

    # Compute the final coefficients
    g = sm.OLS(y, model.getRegressable(X)).fit()
    model.set_coefficients(g.params)
    return model


def runMARSBackward(model: MARSModel, X, y, n):
    raise NotImplementedError()


def generateCandidatePairs(parent, t, v, labels):
    newbasisFunctionPos = HingeFunctionBaseFunction(t, v, labels[v], True)
    newbasisFunctionNeg = HingeFunctionBaseFunction(t, v, labels[v], False)
    prodpos = HingeFunctionProductBaseFunction([parent, newbasisFunctionPos])
    prodneg = HingeFunctionProductBaseFunction([parent, newbasisFunctionNeg])

    posterm = MARSModelTerm(prodpos, 0.0, Operator.POS)
    negterm = MARSModelTerm(prodneg, 0.0, Operator.POS)

    return posterm, negterm


def check_nonzero(func: BaseFunction, X, v):
    values = []

    varia = func.getVariables()
    for j in range(0, X.shape[0]):
        evalvalues = {}
        for var in varia:
            evalvalues[var] = X[j][var]

        if func.getvalue(evalvalues) > 0:
            values.append(X[j][v])

    return values
