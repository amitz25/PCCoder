import argparse
import json


def avg(l):
    return sum(l) / float(len(l))


def analyze(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    res = [json.loads(x) for x in lines]

    succeeded = [x for x in res if x['result'] != 'Failed']
    times = sorted([x['time'] for x in res])

    for ratio in [0.05, 0.1, 0.2,  0.4, 0.6, 0.7, 0.8, 0.9, 0.99]:
        print("%f: %f" % (ratio, times[int(ratio * len(times)) - 1]))

    print("Total solved: %d\\%d - %f%%" % (len(succeeded), len(res), len(succeeded) / len(res) * 100.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    args = parser.parse_args()
    analyze(args.input_path)


if __name__ == '__main__':
    main()
