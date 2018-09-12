from dsl.impl import HIGHER_ORDER_FUNCTIONS, FIRST_ORDER_FUNCTIONS, LAMBDAS


def build_operator_space():
    operators = []
    for func in HIGHER_ORDER_FUNCTIONS:
        for lambd in LAMBDAS:
            if lambd.type == func.input_type[0]:
                operators.append(Operator(func, lambd))
    operators += [Operator(func) for func in FIRST_ORDER_FUNCTIONS]
    return operators


class Operator(object):
    """
    Represents a combination of function + lambda (or just function if the function does not receive a lambda).
    This type is needed for the "function head" of the network.
    """
    def __init__(self, function, lambd=None):
        self.function = function
        self.lambd = lambd

    @staticmethod
    def from_statement(statement):
        if isinstance(statement.args[0], int):
            return Operator(statement.function)
        else:
            return Operator(statement.function, statement.args[0])

    def __repr__(self):
        if self.lambd:
            return "<Operator: %s %s>" % (self.function, self.lambd)
        else:
            return "<Operator: %s>" % self.function

    def __eq__(self, other):
        if not isinstance(other, Operator):
            return False
        return self.function == other.function and self.lambd == other.lambd

    def __hash__(self):
        return hash(str(self))


operator_space = build_operator_space()
num_operators = len(operator_space)
operator_to_index = dict([(func, indx) for indx, func in enumerate(operator_space)])
