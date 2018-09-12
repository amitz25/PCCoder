import params
import numpy as np

from dsl.value import NULLVALUE
from dsl.function import Function, OutputOutOfRangeError, NullInputError
from env.statement import Statement

class ProgramState(object):
    """
    Represents an execution of a program on a single example
    """
    def __init__(self, example=None):
        if example:
            self._vars = example.inputs
            self.output = example.output
            self.num_inputs = len(self._vars)
            self.output_idx = self.num_inputs - 1

    @property
    def vars(self):
        return self._vars[:]

    def copy(self):
        new_env = ProgramState()
        new_env._vars = self._vars[:]
        new_env.output = self.output
        new_env.num_inputs = self.num_inputs
        new_env.output_idx = self.output_idx
        return new_env

    def step(self, statement, out_idx=None):
        f = statement.function
        args = list(statement.args)
        for i in range(len(args)):
            if isinstance(args[i], int):
                args[i] = self._vars[args[i]]

        res = f(*args)
        if out_idx is None:
            assert len(self._vars) < params.max_program_vars, "Too many statements added without dropping!"
            self.output_idx = len(self._vars)
            self._vars.append(res)
        else:
            self._vars[out_idx] = res
            self.output_idx = out_idx

    def get_encoding(self):
        encoded_vars = [var.encoded for var in self._vars]
        if len(encoded_vars) < params.max_program_vars:
            encoded_vars.extend([NULLVALUE.encoded] * (params.max_program_vars - len(self._vars)))
        return np.array(encoded_vars + [self.output.encoded])

    def is_solution(self):
        return self._vars[self.output_idx] == self.output

    def __repr__(self):
        return "<Vars: %s Output: %s>" % (str(self._vars), str(self.output))


class ProgramEnv(object):
    """
    Represents multiple parallel runs of a program, each one on a different example.
    """
    def __init__(self, examples=[]):
        self.states = [ProgramState(example) for example in examples]
        self.var_types = []
        if examples:
            self.var_types = [var.type for var in self.states[0].vars]
        self.num_vars = len(self.var_types)
        self.total_num_vars = self.num_vars
        self.real_var_idxs = list(range(self.num_vars))

    def is_valid(self, statement):
        for i in range(len(statement.args)):
            arg = statement.args[i]
            if isinstance(arg, Function):
                if arg.type != statement.input_types[i]:
                    return False
            else:
                if arg >= self.num_vars or self.var_types[arg] != statement.input_types[i]:
                    return False
        return True

    def copy(self):
        new_env = ProgramEnv()
        new_env.states = [state.copy() for state in self.states]
        new_env.var_types = self.var_types[:]
        new_env.real_var_idxs = self.real_var_idxs[:]
        new_env.num_vars = self.num_vars
        new_env.total_num_vars = self.total_num_vars

        return new_env

    def step(self, statement, out_idx=None):
        if out_idx is None:
            self.var_types.append(statement.output_type)
            self.real_var_idxs.append(self.num_vars)
            self.num_vars += 1
        else:
            self.var_types[out_idx] = statement.output_type
            self.real_var_idxs[out_idx] = self.total_num_vars

        self.total_num_vars += 1

        for state in self.states:
            state.step(statement, out_idx)

    def step_safe(self, statement, out_idx=None):
        """
        Copies the env before performing the statement, ensuring that the original env is untouched.
        Furthermore, in case the statement is invalid the function returns None instead of throwing.
        This function is less efficient than step(), and is meant specifically for the search component
        that needs to preserve the state of the original env and might also try an invalid statement.
        """
        if not self.is_valid(statement):
            return None

        new_env = self.copy()
        try:
            new_env.step(statement, out_idx)
        except (NullInputError, OutputOutOfRangeError):
            return None

        return new_env

    def get_encoding(self):
        """
        Returns the encoding of the environment.
        Unlike other classes, the encoding is not cached and is calculated dynamically on purpose.
        This is done both for efficiency, and because the encoding may change.
        """
        return np.array([state.get_encoding() for state in self.states])

    def is_solution(self):
        for state in self.states:
            if not state.is_solution():
                return False
        return True

    def statement_to_real_idxs(self, statement):
        f, args = statement.function, list(statement.args)
        for i, arg in enumerate(args):
            if isinstance(arg, int):
                args[i] = self.real_var_idxs[arg]
        return Statement(f, args)

    def __repr__(self):
        return "\n".join(["State %d: %s" % (i, state) for i, state in enumerate(self.states)])
