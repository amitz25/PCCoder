from dsl.impl import ALL_FUNCTIONS, LAMBDAS
from dsl.types import FunctionType, INT, LIST

import params
import itertools


def build_statement_space():
    statements = []
    for func in ALL_FUNCTIONS:
        input_type = func.input_type
        if not isinstance(input_type, tuple):
            input_type = (input_type,)

        argslists = []
        for type in input_type:
            if type in [LIST, INT]:
                argslists.append(range(params.max_program_vars))
            elif isinstance(type, FunctionType):
                argslists.append([x for x in LAMBDAS if x.type == type])
            else:
                raise ValueError("Invalid input type encountered!")
        statements += [Statement(func, x) for x in list(itertools.product(*argslists))]

    return statements


class Statement(object):
    def __init__(self, function, args):
        self.function = function
        self.args = tuple(args)

        self.input_types = function.input_type
        self.output_type = self.function.output_type

        if not isinstance(self.input_types, tuple):
            self.input_types = (self.input_types,)

    def __repr__(self):
        return "<Statement: %s %s>" % (self.function, self.args)

    def __eq__(self, other):
        if not isinstance(other, Statement):
            return False
        return self.function == other.function and self.args == other.args

    def __hash__(self):
        return hash(str(self))


statement_space = build_statement_space()
num_statements = len(statement_space)
index_to_statement = dict([(indx, statement) for indx, statement in enumerate(statement_space)])
statement_to_index = {v: k for k, v in index_to_statement.items()}
