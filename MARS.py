import sys

from BaseFunction import BaseFunction, HingeFunctionBaseFunction, HingeFunctionProductBaseFunction
from MARSModel import MARSModel, MARSModelTerm, Operator

import statsmodels.api as sm

import numpy as np

MAXVAL = sys.maxsize


def runMARS(self, X, y, labels, n, maxSplits):
    model = MARSModel(1.0)

    for M in range(2, maxSplits, 2):
        lof_ = MAXVAL
        m_ = None
        v_ = None
        t_ = None
        for m in range(0, M):  # For all existing terms
            basefunc = model.get_component(m).get_function()
            varsinbasefunc = basefunc.getVariables()
            allvars = np.arange(n)
            allvars -= varsinbasefunc
            for v in allvars:  # For all suitable vars
                ts = check_nonzero(basefunc, X, n, v)
                for t in ts:  # For all suitable values of vars
                    newbasisFunctionPos = HingeFunctionBaseFunction(t, v, labels[v], True)
                    newbasisFunctionNeg = HingeFunctionBaseFunction(t, v, labels[v], False)
                    prodpos = HingeFunctionProductBaseFunction([basefunc, newbasisFunctionPos])
                    prodneg = HingeFunctionProductBaseFunction([basefunc, newbasisFunctionNeg])

                    posterm = MARSModelTerm(prodpos, 0.0, Operator.POS)
                    negterm = MARSModelTerm(prodneg, 0.0, Operator.NEG)
                    newmodel = model.copy()
                    newmodel.add_component(posterm)
                    newmodel.add_component(negterm)

                    g = sm.OLS(y, newmodel.getRegressable(X)).fit()
                    if g.ssr < lof_:
                        lof_ = g.ssr
                        m_ = m
                        v_ = v
                        t_ = t

        # Add new terms to model
        basefunc = model.get_component(m_).get_function()
        newbasisFunctionPos = HingeFunctionBaseFunction(t_, v_, labels[v_], True)
        newbasisFunctionNeg = HingeFunctionBaseFunction(t_, v_, labels[v_], False)
        prodpos = HingeFunctionProductBaseFunction([basefunc, newbasisFunctionPos])
        prodneg = HingeFunctionProductBaseFunction([basefunc, newbasisFunctionNeg])
        posterm = MARSModelTerm(prodpos, 0.0, Operator.POS)
        negterm = MARSModelTerm(prodneg, 0.0, Operator.NEG)
        model.add_component(posterm)
        model.add_component(negterm)

    return model


def check_nonzero(func: BaseFunction, X, n, v):
    values = []

    evalvalues = []
    varia = func.getVariables()
    for j in range(0, len(X[0])):
        for var in varia:
            evalvalues.append(X[var][j])

        if func.getvalue(evalvalues) > 0:
            values.append(X[v][j])

    return values
