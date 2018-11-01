import argparse
import json
import torch
import multiprocessing
import time

from baseline.robustfill.model import RobustFill
from baseline.robustfill.train import load_data
from env.env import ProgramEnv
from dsl.example import Example
from dsl.program import Program


def init_worker(*args):
    global counter, fail_counter, model, program_len, timeout
    counter, fail_counter, model, program_len, timeout = args


def robustfill_cab(env, max_depth, model, beam_size, width, timeout, input, input_lens, output,
        output_lens, input_masks, output_masks):

    start_time = time.time()
    state = {'num_steps': 0, 'end_time': start_time + timeout}

    res = False
    while time.time() < state['end_time']:
        res = model.beam_search(env, max_depth, input, input_lens, output, output_lens, input_masks,
                                output_masks, beam_size, width, state)
        if res is not None:
            break
        beam_size *= 2

    ret = {'result': res, 'num_steps': state['num_steps'], 'time': time.time() - start_time,
           'beam_size': beam_size, 'width': width}
    return ret


def solve_problem_worker(args):
    line, input, input_lens, output, output_lens, input_masks, output_masks = args
    examples = Example.from_line(line)
    sol = Program.parse(line['program'])
    env = ProgramEnv(examples)
    res = robustfill_cab(env, program_len, model, 100, 48, timeout, input, input_lens, output, output_lens,
                         input_masks, output_masks)

    counter.value += 1
    print("\rSolving problems... %d (failed: %d)" % (counter.value, fail_counter.value), end="")

    if res['result'] is None:
        res['result'] = "Failed"
        fail_counter.value += 1
        return res
    else:
        res['result'] = str(Program(sol.input_types, res['result']))
        return res


def solve_problems(input_path, program_len, model, num_workers, timeout):
    # Prevents deadlocks due to torch's problems with GPUs on multi processes.
    # This line is here for convenience, but it is recommended to solve problems on CPU since the overhead
    # in this case is minimal.
    torch.set_num_threads(1)

    with open(input_path, 'r') as f:
        lines = f.read().splitlines()
        data = load_data(lines, num_workers)
        filtered_data = dict()

        for k in ['input', 'input_lens', 'output', 'output_lens']:
            filtered_data[k] = torch.LongTensor(data[k])

        for k in ['input_padding_mask', 'output_padding_mask']:
            filtered_data[k] = torch.FloatTensor(data[k])

        worker_data = []
        for i, line in enumerate(lines):
            line_data = {}
            for k, v in filtered_data.items():
                line_data[k] = v[i].unsqueeze(0)

            worker_data.append((json.loads(lines[i]), line_data['input'], line_data['input_lens'],
                                line_data['output'], line_data['output_lens'], line_data['input_padding_mask'],
                                line_data['output_padding_mask']))

    counter = multiprocessing.Value('i', 0)
    fail_counter = multiprocessing.Value('i', 0)

    if num_workers is None or num_workers > 1:
        pool = multiprocessing.Pool(processes=num_workers, initializer=init_worker,
                                    initargs=(counter, fail_counter, model, program_len, timeout))
        return pool.map(solve_problem_worker, worker_data)
    else:
        # Don't run in pool to enable debugging
        init_worker(counter, fail_counter, model, program_len, timeout)
        return [solve_problem_worker(data) for data in worker_data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('timeout', type=int)
    parser.add_argument('max_program_len', type=int)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--max_beam_size', type=int, default=6400)

    args = parser.parse_args()

    model = RobustFill()
    model.load(args.model_path)

    model.eval()

    res = solve_problems(args.input_path, args.max_program_len, model, args.num_workers, args.timeout)
    print('')

    solved = len([x for x in res if x['result'] != 'Failed'])
    print("Solved: %d\\%d:" % (solved, len(res)), str(100.0 * solved / len(res)) + '%')

    open(args.output_path, 'w').write('\n'.join([json.dumps(str(x)) for x in res]))


if __name__ == '__main__':
    main()
