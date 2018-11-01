from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
import random
import torch
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils import clip_grad_norm_

from cuda import use_cuda
from dsl.program import Program
from dsl.example import Example
from baseline.robustfill.model import RobustFill, program_max_seq_len, reverse_program_vocab, END_PROGRAM_TOKEN
from baseline.robustfill.env import var_to_tokens, program_to_tokens, io_max_seq_len

learn_rate = 0.01
batch_size = 128
num_epochs = 40

max_grad_norm = 1.0

class DictDataset(Dataset):
    def __init__(self, data, idxs):
        self.data = dict()
        self.len = len(idxs)
        for k, v in data.items():
            self.data[k] = v[idxs]

    def __getitem__(self, index):
        res = dict()
        for k, v in self.data.items():
            res[k] = v[index]
        return res

    def __len__(self):
        return self.len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to data')
    parser.add_argument('output_path', type=str, help='Output path of trained model')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--max_len', type=int, default=None)
    args = parser.parse_args()
    train(args)


def pad_seq(seq, size):
    orig_size = len(seq)
    if len(seq) < size:
        seq += [0] * (size - len(seq))
    return orig_size


def generate_prog_data(line):
    data = json.loads(line.rstrip())
    examples = Example.from_line(data)
    program = Program.parse(data['program'])

    prog_data = dict(input=[], input_lens=[], output=[], output_lens=[])
    input_padding_mask = np.zeros((len(examples), io_max_seq_len))
    output_padding_mask = np.zeros((len(examples), io_max_seq_len))
    for i, example in enumerate(examples):
        example_input = []
        for inp in example.inputs:
            example_input += var_to_tokens(inp)

        input_len = pad_seq(example_input, io_max_seq_len)
        prog_data['input_lens'].append(input_len)
        input_padding_mask[i, : input_len] = 1
        prog_data['input'].append(np.array(example_input))

        example_output = var_to_tokens(example.output)
        output_len = pad_seq(example_output, io_max_seq_len)
        prog_data['output_lens'].append(output_len)
        prog_data['output'].append(example_output)
        output_padding_mask[i, : output_len] = 1

    prog_data['target'] = program_to_tokens(program)
    prog_data['target_len'] = pad_seq(prog_data['target'], program_max_seq_len)
    dec_padding_mask = np.zeros(program_max_seq_len)
    for i in range(prog_data['target_len']):
        dec_padding_mask[i] = 1

    for k, v in prog_data.items():
        prog_data[k] = np.array(v)

    prog_data['input_padding_mask'] = input_padding_mask
    prog_data['output_padding_mask'] = output_padding_mask
    prog_data['dec_padding_mask'] = dec_padding_mask

    return prog_data


def load_data(lines, num_workers):
    data = defaultdict(list)

    pool = Pool(processes=num_workers)
    res = list(tqdm(pool.imap(generate_prog_data, lines), total=len(lines)))
    for prog_data in res:
        for k, v in prog_data.items():
            data[k].append(v)

    for k, v in data.items():
        data[k] = np.stack(v)

    return data


def train(args):
    with open(args.input_path, 'r') as f:
        lines = f.read().splitlines()
        if args.max_len:
            lines = lines[:args.max_len]
        data = load_data(lines, args.num_workers)

    model = RobustFill()

    if use_cuda:
        model.cuda()

    model = nn.DataParallel(model)

    for k in ['input', 'input_lens', 'output', 'output_lens', 'target']:
        data[k] = torch.LongTensor(data[k])

    for k in ['target_len', 'input_padding_mask', 'output_padding_mask', 'dec_padding_mask']:
        data[k] = torch.FloatTensor(data[k])

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    dataset_size = data['input'].shape[0]
    train_dataset_size = int(0.9 * dataset_size)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_indices = indices[:train_dataset_size]
    test_indices = indices[train_dataset_size:]

    train_dataset = DictDataset(data, train_indices)
    test_dataset = DictDataset(data, test_indices)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        print("Epoch %d" % epoch)
        lr_sched.step()

        train_losses = []
        grads = []

        for i_batch, batch in enumerate(train_data_loader):
            print("\rBatch %d\\%d" % (i_batch, len(train_data_loader)), end="")

            optimizer.zero_grad()

            for k, v in batch.items():
                batch[k] = v.cuda()

            loss = model(batch['input'], batch['input_lens'], batch['output'], batch['output_lens'],
                         batch['input_padding_mask'], batch['output_padding_mask'], batch['dec_padding_mask'],
                         batch['target'], batch['target_len']).sum()
            loss.backward()

            train_losses.append(loss.data)

            grads.append(clip_grad_norm_(model.parameters(), max_grad_norm))

            optimizer.step()

        train_loss = np.array(train_losses).mean()
        print(" Train loss: %f" % train_loss, end="")

        model.eval()

        with torch.no_grad():
            test_losses = []
            for i_batch, batch in enumerate(test_data_loader):
                for k, v in batch.items():
                    batch[k] = v.cuda()

                loss = model(batch['input'], batch['input_lens'], batch['output'], batch['output_lens'],
                             batch['input_padding_mask'], batch['output_padding_mask'], batch['dec_padding_mask'],
                             batch['target'], batch['target_len']).sum()
                test_losses.append(loss.data)

            test_loss = np.array(test_losses).mean()
            print(" Test loss: %f" % test_loss)

            res = model.module.greedy_decode(batch['input'], batch['input_lens'], batch['output'], batch['output_lens'],
                                             batch['input_padding_mask'], batch['output_padding_mask'])
            for i in range(min(len(res), 10)):
                tokens = [reverse_program_vocab[token.item()] for token in res[i]]
                if END_PROGRAM_TOKEN in tokens:
                    tokens = tokens[:tokens.index(END_PROGRAM_TOKEN)]
                print("Decoding output: %s" % " ".join([str(x) for x in tokens]))

        model.module.save(args.output_path + ".%d" % epoch)


if __name__ == '__main__':
    main()
