import enum
from typing import List
import numpy as np

from BaseFunction import BaseFunction, ConstantBaseFunction


class Operator(enum.Enum):
    POS = 1
    NEG = -1


class MARSModelTerm:
    def __init__(self, basefunction: BaseFunction, coefficient: float, operator: Operator):
        self.func = basefunction
        self.op = operator
        self.coef = coefficient

    def eval(self, x):
        return self.op.value * (self.coef * self.func.getvalue(x))

    def get_function(self):
        return self.func

    def __str__(self):
        if self.func.type == 1:
            return self.func.getvalue([])
        if self.op == Operator.POS:
            return "+ " + str(self.coef) + str(self.func)
        else:
            coefstr = str(self.coef)[1:]
            return "- " + coefstr + str(self.func)


class MARSModel:
    components: List[MARSModelTerm] = []

    def __init__(self, intercept: float = 1.0, empty = False):
        if not empty:
            self.components.append(MARSModelTerm(ConstantBaseFunction(), intercept, Operator.POS))

    def add_component(self, component: MARSModelTerm):
        self.components.append(component)

    def get_component(self, i):
        return self.components[i]

    def set_coefficients(self, coef):
        for i in range(1, len(self.components)):
            self.components[i].coef = coef[i]
            if coef[i] < 0:
                self.components[i].op = Operator.NEG

    def getRegressable(self, X: np.ndarray):
        regressable = []
        row = []
        for j in range(0, X.shape[0]):
            for i in range(0, len(self.components)):
                func = self.components[i].get_function()
                evalvalues = {}
                varia = func.getVariables()
                for var in varia:
                    evalvalues[var] = X[j][var]
                fval = func.getvalue(evalvalues)
                row.append(fval)
            regressable.append(row)
            row = []

        return regressable

    def copy(self):
        newmodel = MARSModel(empty=True)
        newmodel.components = self.components.copy()
        return newmodel

    def __str__(self):
        stringbuilder = str(self.components[0].eval(0)) + " "
        for i in range(1, len(self.components)):
            stringbuilder += str(self.components[i])
            stringbuilder += " "

        return stringbuilder
