from __future__ import print_function

import params
from dsl.impl import FUNCTIONS_AND_LAMBDAS, ALL_FUNCTIONS
from dsl.types import LIST, INT
from dsl.program import Program
from env.statement import Statement

START_PROGRAM_TOKEN = '<'
END_PROGRAM_TOKEN = '>'
PADDING_TOKEN = 'PAD'

START_LIST_TOKEN = '['
END_LIST_TOKEN = ']'
LIST_TYPE_TOKEN = 'L'
INT_TYPE_TOKEN = 'I'


# Generate a mapping from function/arg to tokens
def generate_program_vocab():
    vocab = {PADDING_TOKEN: 0, START_PROGRAM_TOKEN: 1, END_PROGRAM_TOKEN: 2}
    for i, func in enumerate(FUNCTIONS_AND_LAMBDAS):
        vocab[func] = i + 3
    size = len(vocab)

    for i, num in enumerate(range(params.max_program_vars)):
        vocab[num] = size + i

    return vocab


# Generate a vocabulary for the I/O samples
def generate_io_vocab():
    vocab = {PADDING_TOKEN: 0, START_LIST_TOKEN: 1, END_LIST_TOKEN: 2, LIST_TYPE_TOKEN: 3, INT_TYPE_TOKEN: 4}
    numbers = range(params.integer_min, params.integer_max + 1)
    size = len(vocab)

    for i, num in enumerate(numbers):
        vocab[num] = size + i

    return vocab


program_vocab = generate_program_vocab()
reverse_program_vocab = dict([(v, k) for k, v in program_vocab.items()])
program_vocab_size = len(program_vocab)

# Each statement can be at most 4 tokens - Function + 3 args. The additional 2 tokens are the start and end tokens.
program_max_seq_len = 4 * params.max_program_len + 2

io_vocab = generate_io_vocab()
reverse_io_vocab = dict([(v, k) for k, v in io_vocab.items()])
io_vocab_size = len(io_vocab)
io_max_seq_len = params.num_inputs * (params.max_list_len + 3)


def tokens_to_program(seq, input_types):
    tokens = [reverse_program_vocab[token] for token in seq]
    if tokens[0] == START_PROGRAM_TOKEN:
        tokens = tokens[1:]

    indx = 0
    statements = []
    while indx < len(tokens) and tokens[indx] != END_PROGRAM_TOKEN:
        token = tokens[indx]
        if not token in ALL_FUNCTIONS:
            return None

        if isinstance(token.input_type, tuple):
            num_args = len(token.input_type)
        else:
            num_args = 1
        args = tokens[indx + 1 : indx + 1 + num_args]
        statements.append(Statement(token, args))
        indx = indx + 1 + num_args

    return Program(input_types, statements)


def statement_to_tokens(statement):
    return [program_vocab[statement.function]] + [program_vocab[arg] for arg in statement.args]


def program_to_tokens(program):
    res = []
    for statement in program.statements:
        res += statement_to_tokens(statement)
    return res + [program_vocab[END_PROGRAM_TOKEN]]


def var_to_tokens(var):
    out = [io_vocab[START_LIST_TOKEN]]
    if var.type == INT:
        out += [io_vocab[INT_TYPE_TOKEN], int_to_token(var.val)]
    elif var.type == LIST:
        out += [io_vocab[LIST_TYPE_TOKEN]] + [int_to_token(x) for x in var.val]
    else:
        raise ValueError("Unknown var type: %s" % var.type)
    return out + [io_vocab[END_LIST_TOKEN]]


def int_to_token(num):
    return io_vocab[num]
