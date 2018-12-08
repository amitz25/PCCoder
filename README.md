# PCCoder
The official implementation of the paper "Automatic Program Synthesis of Long Programs with a Learned Garbage Collector":
https://arxiv.org/abs/1809.04682

## Requirements
- Python 3.6 
- Pytorch >= 0.4.0 
- pathos 
- tqdm

## Generating a dataset
scripts/gen_programs.py allows the generation of a dataset either from scratch, or by continuing from an existing database. For example, to generate a dataset similar to experiment 1 from the paper:
```
python3.6 -m scripts.gen_programs --num_train=100000 --num_test=500 --train_output_path=train_dataset --test_output_path=test_dataset --max_train_len=12 --test_lengths="5" --num_workers=8
```

1. You can (and should) have virtually all programs with lengths <= 3 in the dataset to ensure that longer programs are meaningful. Generation for smaller lengths is slower since the number of possible programs is small. Therefore, it is recommended to generate a dataset for length 3 once, and then use it as a cache with --cache.
2. test_lengths accepts a list of test lengths separated by a space.

## Training the network
scripts/train.py expects just the input dataset and the output path of the model. A model is saved for each epoch.
```
python3.6 -m scripts.train dataset model
```

CUDA is detected automatically and uses the GPU if it's available.

## Solving a test dataset
scripts/solve_problems expects a list of I/O sample sets and a network and solves them in multiple processes concurrently.
For now, CUDA is not used when solving problems. We advise you to not use this at test-time as its effect on speed is minimal (at test-time) and it may cause PyTorch some synchronization problems.
```
python3.6 -m scripts.solve_problems dataset result model 60 5 --num_workers=8
```

1. max_program_len dictates the maximum depth of the search.
2. The result file has a json dictionary line for each program predicted. The dictionary contains the predicted program and some details about the search, like the amount of time the search took and the final beam size.
3. Use --search_method to change the method from the default CAB search to DFS.

## Changing parameters
params.py contains all of the global "constants". This includes the program's memory size (which is calculated as params.num_inputs + params.max_program_len which are both changeable), number of exampes, DSL int range and max array size, and more.

## Program representation
As in https://github.com/dkamm/deepcoder, each program is represented in a compact string:
1. '|' delimites each statement
2. ',' delimits a function call and its arguments

Specifically, this is the general format:
```
INPUT_0_TYPE|...|INPUT_K_TYPE|FUNCTION_CALL_0,PARAMS_0|...|FUNCTION_CALL_N,PARAMS_N
```

For example, the program:
```
a <- [int]
b <- FILTER (%2==0) a
c <- MAP (/2) b
d <- SORT c
e <- LAST d
```

will be represented as:
```
LIST|FILTER,EVEN,0|MAP,/2,1|SORT,2|LAST,3
```

## RobustFill
The full implementation of RobustFill's attention-B variant (https://arxiv.org/abs/1703.07469) for this DSL is inside baseline/robustfill. Since our problem (and DSL) is significantly different from RobustFill's original paper, some alterations were made:
1. For attention, we use the "concat" variant whereas RobustFill used "general" in their paper (https://arxiv.org/pdf/1508.04025.pdf).
2. For evaluation, we use an altered version of beam-search that detects the prediction of invalid statements prematurely and significantly improves results. Furthermore, we use CAB instead of a vanilla beam-search with constant-size. 
3. In order to give the I/O samples as input to the LSTMs (I and O), we encode them similarly to how it is done for PCCoder. Concretely, for each variable, we pass as input '\[', then '0' or '1' (type of var - list of int), then the values of the list number-by-number, and then ']'. The decoder LSTM (P) outputs a program token-by-token, where each token can be either a parameter (number), lambda, or function.

In order to run RobustFill, use robustfill/train.py and robustfill/solve_problems.py similarly to how they're used for PCCoder.

## Special thanks
Since there is no public implementation of DeepCoder, this code is based on a reimplementation by dkamm (https://github.com/dkamm/deepcoder). Most of the code was since heavily changed, but some parts still remain, most notably the implementation of the DSL.

Furthermore, several parts of the reimplementation of the RobustFill baseline were taken from atulkum's PyTorch implementation of Pointer-Generator networks: https://github.com/atulkum/pointer_summarizer (and were since heavily edited).
