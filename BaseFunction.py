# type 1 = constant 1
# type 2 = single hinge function
# type 3 = product of two or more hinge functions
from typing import List


class BaseFunction:
    type = 0

    def __init__(self):
        self.type = 0

    def getvalue(self, x):
        raise "BaseFunction should not be called"

    def getVariables(self):
        raise "BaseFunction should not be called"


class ConstantBaseFunction(BaseFunction):
    type = 1

    def __init__(self):
        super().__init__()
        self.value = 1.0

    def getvalue(self, x):
        return self.value

    def __str__(self):
        return "1"

    def getVariables(self):
        return []


class HingeFunctionBaseFunction(BaseFunction):
    type = 2

    def __init__(self, c: int, x: int, label: str, t: bool):
        super().__init__()
        self.c = c
        self.x = x
        self.label = label
        self.t = t  # true = x - constant

    def getvalue(self, x):
        if self.t:
            return max(0, x - self.c)
        else:
            return max(0, self.c - x)

    def getVariables(self):
        return [self.x]

    def __str__(self):
        if self.t:
            return "max(0," + self.label + " - " + str(self.c) + ")"
        else:
            return "max(0," + str(self.c) + " - " + self.label + ")"


class HingeFunctionProductBaseFunction(BaseFunction):
    type = 3

    def __init__(self, h: list):
        super().__init__()
        self.hinges = h
        if len(h) < 2:
            raise "Hinge list cannot have size < 2"

    def getvalue(self, x: List[int]):
        if len(x) != len(self.hinges):
            raise "Number of X values should be equal to number of hinge functions in product"

        v = 1
        for i in range(0, len(self.hinges)):
            v *= self.hinges[i].getvalue(x[i])

        return v

    def getVariables(self):
        l = []
        for h in self.hinges:
            l += h.getVariables()

        return l

    def __str__(self):
        stringbuilder = ""
        for h in self.hinges:
            stringbuilder += str(h)

        return stringbuilder
