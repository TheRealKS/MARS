import numpy as np
import collections.abc


def generateEmptyMarsModel(intercept=1):
    """Generate an empty MARS model with chosen intercept value (default = 1)"""
    # Empty root node
    n = [intercept, "", 0, True]
    return [n]


def generateHingeFunction(var, c, side):
    """Generate a new hinge function"""
    return [0, var, c, side]


def getModelLength(model, topdepth=None):
    """Get length of the model counting intercept + all hinge functions. This is not the same as the number of basic functions"""
    if topdepth is None:
        topdepth = len(model)
    else:
        topdepth += 1
    l = 0
    for i in range(0, topdepth):
        term = model[i]
        if isinstance(term[0], int):
            l += 1
        elif isinstance(term[0], list):
            # This is a hinge func product
            l += getModelLengthRec(term)

    return l


def getModelLengthRec(term, l=0):
    for subterm in term:
        if isinstance(subterm[0], int):
            l += 1
        elif isinstance(subterm[0], list):
            l += getModelLengthRec(subterm)
    return l


def removeHingeFromModel(model, index, c=0, top=True):
    """Remove one hinge function from the model (which may be part of a product basic function)"""
    if index == c:
        return model, c
    for m in range(0, len(model)):
        if isinstance(model[m][0], list):
            new, cc = removeHingeFromModel(model[m], index, c, False)
            model[m] = new
            c = cc
        else:
            if c == index:
                del model[m]
                if top:
                    return model, c
                else:
                    return model[0], c
            c += 1

    return model, c


def getVarsInBaseFunc(func):
    """Get all vars in a basic function"""
    vars = []
    if isinstance(func[0], collections.abc.Sequence):
        # This is a hinge func product
        for subfunc in func:
            vars = getVarsInBaseFunc(subfunc).flatten()
    else:
        vars.append(func[1])

    return np.array(vars)


def evaluateNonProdBaseFunc(func, valdict):
    """Get the value for a non-product basic function (intercept or hinge function)"""
    if func[1] == "":
        # Root
        return func[0]
    else:
        if func[3]:
            return max(0, valdict[func[1]] - func[2])
        else:
            return max(0, func[2] - valdict[func[1]])


def evaluateBaseFunc(func, valdict):
    """Get the value for a basic function"""
    if isinstance(func[0], collections.abc.Sequence):
        # This is a hinge func product
        val = 1
        for subfunc in func:
            val *= evaluateBaseFunc(subfunc, valdict)
        return val

    return evaluateNonProdBaseFunc(func, valdict)


def checkWhereBaseFuncNonZero(func, currentv, X, varlabels):
    """Find all values where a certain basic function does not evaluate to 0"""
    ts = []
    for j in range(0, X.shape[0]):
        row = X[j]
        d = {varlabels[i]: row[i] for i in range(len(row))}
        val = evaluateBaseFunc(func, d)

        if val > 0:
            ts.append(row[currentv])

    return ts


def generateRegressableFunction(model, varlabels):
    """Get a function which generates a data matrix that can be used to fit a regression model (to find the
    coefficients for the MARS model) """
    def regfunc(X):
        newarr = []
        for sample in X:
            roww = [model[0][0]]
            for i in range(0, len(model) - 1):
                d = {varlabels[j]: sample[j] for j in range(len(sample))}
                term = model[i + 1]
                roww.append(evaluateBaseFunc(term, d))

            newarr.append(roww)

        return np.array(newarr)

    return regfunc


def generateEvalFunction(model, varlabels):
    """Get a function which can be used to evaluate the model for each sample in a dataset"""
    def evalfunc(coefs, X):
        newarr = []
        for sample in X:
            sum = model[0][0]
            for i in range(0, len(model) - 1):
                d = {varlabels[j]: sample[j] for j in range(len(sample))}
                term = model[i + 1]
                c = coefs[i]
                sum += c * evaluateBaseFunc(term, d)

            newarr.append(sum)

        return np.array(newarr)

    return evalfunc


def evaluateModel(model, varlabels, coefs, X):
    """Evaluate the model for a given dataset, using given coefficients"""
    ref = generateEvalFunction(model, varlabels)
    return ref(coefs, X)
