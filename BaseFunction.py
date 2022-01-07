# type 1 = constant 1
# type 2 = single hinge function
# type 3 = product of two or more hinge functions

class BaseFunction:
    type = 0

    def __init__(self):
        #"do nothing"

    def getvalue(self, x):
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


class HingeFunctionBaseFunction(BaseFunction):
    type = 2

    def __init__(self, c, x, t: bool):
        super().__init__()
        self.c = c
        self.x = x
        self.t = t  # true = x - constant

    def getvalue(self, x):
        if self.t:
            return max(0, x - self.c)
        else:
            return max(0, self.c - x)

    def __str__(self):
        if self.t:
            return "max(0,x - " + str(self.c) + ")"
        else:
            return "max(0," + str(self.c) + " - x)"


class HingeFunctionProductBaseFunction(BaseFunction):
    type = 3

    def __init__(self, h: list):
        super().__init__()
        self.hinges = h
        if len(h) < 2:
            raise "Hinge list cannot have size < 2"

    def getvalue(self, x):
        v = 1
        for h in self.hinges:
            v = v * h.getvalue(x)

        return v

    def __str__(self):
        stringbuilder = ""
        for h in self.hinges:
            stringbuilder += str(h)

        return stringbuilder
