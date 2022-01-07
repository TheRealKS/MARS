import enum
from typing import List

from BaseFunction import BaseFunction, ConstantBaseFunction


class Operator(enum.Enum):
    POS = 1
    NEG = -1


class MARSModelComponent:
    def __init__(self, basefunc: BaseFunction, coefficient: float, operator: Operator):
        self.func = basefunc
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
    components: List[MARSModelComponent] = []

    def __init__(self, intercept: float):
        self.components.append(MARSModelComponent(ConstantBaseFunction(), intercept, Operator.POS))

    def eval(self, x):
        b = 0
        for c in self.components:
            b += c.eval(x)

        return b

    def add_component(self, component: MARSModelComponent):
        self.components.append(component)

    def get_component(self, i):
        return self.components[i]

    def __str__(self):
        stringbuilder = str(self.components[0].eval()) + " "
        for i in range(1, len(self.components)):
            stringbuilder += str(self.components[i])
            stringbuilder += " "

        return stringbuilder