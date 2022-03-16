import enum
from typing import List

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

    def get_coefficient(self):
        return self.coef

    def get_function(self):
        return self.func

    def __str__(self):
        if self.op == Operator.POS:
            return "+ " + str(self.coef) + str(self.func)
        else:
            return "- " + str(self.coef) + str(self.func)


class MARSModel:
    components: List[MARSModelTerm] = []

    def __init__(self, intercept: float = 1.0):
        self.components.append(MARSModelTerm(ConstantBaseFunction(), intercept, Operator.POS))

    def add_component(self, component: MARSModelTerm):
        self.components.append(component)

    def set_component(self, i, newcomponent: MARSModelTerm):
        self.components[i] = newcomponent

    def get_component(self, i):
        return self.components[i]

    def set_coefficients(self, coef):
        for i in range(0, len(self.components)):
            self.components[i].coef = coef[i]

    def getRegressable(self, X):
        regressable = []
        row = []
        for j in range(0, len(X[0])):
            for i in range(0, len(self.components)):
                func = self.components[i].get_function()
                evalvalues = []
                varia = func.getVariables()
                for var in varia:
                    evalvalues.append(X[var][j])
                row.append(func.getvalue(evalvalues))
            regressable.append(row)
            row = []

        return regressable

    def copy(self):
        c = MARSModel()
        c.components = self.components.copy()
        return c

    def __str__(self):
        stringbuilder = str(self.components[0].eval(0)) + " "
        for i in range(1, len(self.components)):
            stringbuilder += str(self.components[i])
            stringbuilder += " "

        return stringbuilder
