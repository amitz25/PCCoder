import torch

use_cuda = torch.cuda.is_available()

if use_cuda:
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor
    torch.backends.cudnn.benchmark = True
else:
    LongTensor = torch.LongTensor
    FloatTensor = torch.FloatTensor
