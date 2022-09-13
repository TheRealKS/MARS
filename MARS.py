import math
import sys

import logging
import copy

from MARSModel import *
from sklearn.linear_model import LinearRegression

import numpy as np

MAXVAL = sys.maxsize


def runMARSForward(X, y, labels, maxSplits):
    model = generateEmptyMarsModel()
    n = len(labels)
    logging.warning("Starting iteration with " + str(n) + "vars..")

    for M in range(1, maxSplits, 2):
        logging.warning("Split " + str(M))
        m_ = None
        v_ = None
        t_ = None
        lof_ = MAXVAL
        for m in range(0, M):  # For all existing terms
            logging.warning("Term " + str(m))
            currentterm = model[m]
            varsinbasefunc = getVarsInBaseFunc(currentterm)
            varsnotinbasefunc = set(labels).difference(np.array(varsinbasefunc).flatten())
            for v in varsnotinbasefunc:  # For all suitable vars
                if m != 0:
                    ts = checkWhereBaseFuncNonZero(currentterm, labels.index(v), X, labels)
                else:
                    ts = X[:, labels.index(v)]
                its = 0
                for t in ts:  # For all suitable values of vars
                    pos = generateHingeFunction(v, t, True)
                    neg = generateHingeFunction(v, t, False)
                    if not currentterm[0] == 1:
                        prodpos = [currentterm, pos]
                        prodneg = [currentterm, neg]
                    else:
                        prodpos = pos
                        prodneg = neg

                    # Evaluate using SSE
                    tempmodel = model.copy()
                    tempmodel.append(prodpos)
                    tempmodel.append(prodneg)
                    evaluatedModelWithCoefs = evalModelWithCoefs(tempmodel, labels, X, y)
                    lof = ssr(evaluatedModelWithCoefs - y)
                    if lof < lof_:
                        lof_ = lof
                        m_ = m
                        v_ = v
                        t_ = t
                    its += 1
                    # print(its)

        # Add new terms to model
        basefunc = model[m_]
        pos = generateHingeFunction(v_, t_, True)
        neg = generateHingeFunction(v_, t_, False)
        if not basefunc[0] == 1:
            prodpos = [basefunc, pos]
            prodneg = [basefunc, neg]
        else:
            prodpos = pos
            prodneg = neg
        model.append(prodpos)
        model.append(prodneg)

    # Compute the final coefficients
    evaluatedModelWithCoefs = evalModelWithCoefs(model, labels, X, y)
    lof = ssr(evaluatedModelWithCoefs - y)
    return model, lof


def runMARSBackward(model, modelrss, X, labels, y, n, maxSplits, d=2):
    logging.warning("Running backward MARS now...")
    J = model
    K = copy.deepcopy(model)
    modellen = getModelLength(model)
    currentgcv = GCV(modelrss, n, modellen, d)
    print(currentgcv)
    for M in reversed(range(1, maxSplits)):
        lof_ = MAXVAL
        L = copy.deepcopy(K)
        for m in range(1, getModelLength(L)):
            KK = copy.deepcopy(L)
            KKdeleted = removeHingeFromModel(KK, m)[0]
            ran = evalModelWithCoefs(KKdeleted, labels, X, y)
            lof = ssr(ran - y)
            submodelgcv = GCV(lof, n, getModelLength(KKdeleted), d)
            if submodelgcv < lof_:
                lof_ = submodelgcv
                K = copy.deepcopy(KKdeleted)
            if submodelgcv < currentgcv:
                currentgcv = submodelgcv
                J = copy.deepcopy(KKdeleted)

    return J, getModelCoefs(J, labels, X, y)


def getModelCoefs(model, labels, X, y):
    evaluatedModel = generateRegressableFunction(model, labels)(X)
    res = LinearRegression(fit_intercept=True).fit(evaluatedModel, y)
    return res.coef_


def evalModelWithCoefs(model, labels, X, y):
    evaluatedModel = generateRegressableFunction(model, labels)(X)
    res = LinearRegression(fit_intercept=True).fit(evaluatedModel, y)
    pred = evaluateModel(model, labels, res.coef_, X)
    return pred


def ssr(pred):
    lof = np.sum(np.power(pred, 2))
    return lof


def GCV(ssr, n, M, d):
    cmhat = M + (d * (M - 1))
    meansq = ssr / n
    penalty = math.pow((1 - (cmhat / n)), 2)
    return meansq / penalty
