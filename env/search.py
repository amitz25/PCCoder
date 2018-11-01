import time

import params
from env.statement import index_to_statement

import torch

FAILED_SOLUTION = [None]

def dfs(env, max_depth, model, width, timeout):
    """
    Perform a DFS tree-search where the nodes are program environments and the edges are statements.
    A limited width is explored (number of statements) to aid with larger lengths.
    """
    start_time = time.time()
    state = {'num_steps': 0, 'num_invalid': 0}
    state['end_time'] = start_time + timeout

    def helper(env, statements, state):
        state['num_steps'] += 1
        if 'end_time' in state and time.time() >= state['end_time']:
            return FAILED_SOLUTION

        if env.is_solution():
            return statements

        depth = len(statements)
        if depth >= max_depth:
            return False

        statement_pred, statement_probs, drop_indx = model.predict(torch.LongTensor(env.get_encoding()).unsqueeze(0))

        statement_pred = statement_pred[0]
        drop_indx = drop_indx[0]

        if env.num_vars == params.max_program_vars:
            to_drop = drop_indx
        else:
            to_drop = None

        num_tries = 0
        for statement_index in reversed(statement_pred[-width:]):
            statement = index_to_statement[statement_index]

            new_env = env.step_safe(statement, to_drop)
            if new_env is None:
                state['num_invalid'] += 1
                continue

            res = helper(new_env, statements + [statement], state)
            if res:
                if res != FAILED_SOLUTION:
                    res[depth] = env.statement_to_real_idxs(res[depth])
                return res
            num_tries += 1

        return False

    res = helper(env, [], state)
    if res == FAILED_SOLUTION:
        res = False
    return {'result': res, 'num_steps': state['num_steps'], 'time': time.time() - start_time,
            'num_invalid': state['num_invalid']}


def cab(env, max_depth, model, beam_size, width, width_growth, timeout, max_beam_size=6553600):
    """
    Performs a CAB search. Each iteration, beam_search is called with an increased beam size and
    width. We increase the beam size exponentially each iteration, to ensure that the majority
    of paths explored are new. This prevents the need of caching which slows things down.

    max_beam_size is provided as a safety precaution
    """
    start_time = time.time()
    state = {'num_invalid': 0, 'num_steps': 0, 'end_time': start_time + timeout}

    res = False
    while time.time() < state['end_time']:
        res = beam_search(env, max_depth, model, beam_size, width, state)
        if res is not False or beam_size >= max_beam_size:
            break
        beam_size *= 2
        width += width_growth

    ret = {'result': res, 'num_steps': state['num_steps'], 'time': time.time() - start_time,
           'beam_size': beam_size, 'num_invalid': state['num_invalid'], 'width': width}
    return ret


def beam_search(env, max_depth, model, beam_size, expansion_size, state):
    """
    Performs a beam search where the nodes are program environments and the edges are possible statements.
    For now, the DSL does not have a vectorized implementation. Thus, only the network prediction is done
    in parallel while the environments are updated in sequential manner.
    """
    def helper(beams, state):
        if time.time() >= state['end_time']:
            return FAILED_SOLUTION

        for env, statements, _ in beams:
            if env.is_solution():
                return statements

        assert len(beams) > 0, "Empty beam list received!"
        depth = len(beams[0][1])
        if depth >= max_depth:
            return FAILED_SOLUTION

        new_beams = []

        env_encodings = [beam[0].get_encoding() for beam in beams]
        env_encodings = torch.LongTensor(env_encodings)
        statement_pred, statement_probs, drop_indx = model.predict(env_encodings)

        for beam_num, (env, statements, prob) in enumerate(beams):
            if time.time() >= state['end_time']:
                return FAILED_SOLUTION

            if env.num_vars == params.max_program_vars:
                to_drop = drop_indx[beam_num]
            else:
                to_drop = None

            for statement_index in reversed(statement_pred[beam_num, -expansion_size:]):
                statement = index_to_statement[statement_index]
                new_env = env.step_safe(statement, to_drop)
                if new_env is None:
                    state['num_invalid'] += 1
                    continue

                state['num_steps'] += 1
                new_beams.append((new_env, statements + [env.statement_to_real_idxs(statement)],
                                  prob * statement_probs[beam_num, statement_index]))

        if len(new_beams) == 0:
            return FAILED_SOLUTION

        new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]
        return helper(new_beams, state)

    res = helper([(env, [], 1)], state)
    if res == FAILED_SOLUTION:
        res = False

    return res