# type 0 = base class, should not be used
# type 1 = constant 1
# type 2 = single hinge function
# type 3 = product of two or more hinge functions
from typing import Dict


class BaseFunction:
    def __init__(self):
        self.type = 0

    def getvalue(self, x):
        raise SyntaxError("BaseFunction should not be called")

    def getVariables(self):
        raise SyntaxError("BaseFunction should not be called")


class ConstantBaseFunction(BaseFunction):
    def __init__(self, value = 1.0):
        super().__init__()
        self.value = value
        self.type = 1

    def getvalue(self, x):
        return self.value

    def getVariables(self):
        return set()

    def __str__(self):
        return "1"


class HingeFunctionBaseFunction(BaseFunction):

    def __init__(self, c: int, x: int, label: str, t: bool):
        super().__init__()
        self.c = c
        self.x = x
        self.label = label
        self.t = t  # true = x - constant
        self.type = 2

    def getvalue(self, x):
        v = x[self.x]
        if self.t:
            return max(0, v - self.c)
        else:
            return max(0, self.c - v)

    def getVariables(self):
        return {self.x}

    def __str__(self):
        if self.t:
            return "max(0," + self.label + " - " + str(self.c) + ")"
        else:
            return "max(0," + str(self.c) + " - " + self.label + ")"


class HingeFunctionProductBaseFunction(BaseFunction):
    def __init__(self, h: list):
        super().__init__()
        self.hinges = h
        if len(h) < 2:
            raise IndexError("Hinge list cannot have size < 2")
        self.type = 3

    def getvalue(self, x: Dict[int, int]):
        v = 1
        for hinge in self.hinges:
            if hinge.type == 2:
                v *= hinge.getvalue(x)
            elif hinge.type == 3:
                v *= hinge.getvalue(x)
            else:
                v *= hinge.getvalue(None)

        return v

    def getVariables(self):
        l = set()
        for h in self.hinges:
            l = l.union(h.getVariables())

        return l

    def __str__(self):
        stringbuilder = ""
        for h in self.hinges:
            stringbuilder += str(h)

        return stringbuilder
