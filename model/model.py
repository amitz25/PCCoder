import numpy as np
import torch.nn.functional as F
from torch import nn

import params
import torch
from cuda import use_cuda
from env.statement import num_statements
from env.operator import num_operators
from model.encoder import DenseEncoder


class BaseModel(nn.Module):
    def load(self, path):
        if use_cuda:
            params = torch.load(path)
        else:
            params = torch.load(path, map_location=lambda storage, loc: storage)

        state = self.state_dict()
        for name, val in params.items():
            if name in state:
                assert state[name].shape == val.shape, "%s size has changed from %s to %s" % \
                                                       (name, state[name].shape, val.shape)
                state[name].copy_(val)
            else:
                print("WARNING: %s not in model during model loading!" % name)

    def save(self, path):
        torch.save(self.state_dict(), path)


class PCCoder(BaseModel):
    def __init__(self):
        super(PCCoder, self).__init__()
        self.encoder = DenseEncoder()
        self.statement_head = nn.Linear(params.dense_output_size, num_statements)
        self.drop_head = nn.Linear(params.dense_output_size, params.max_program_vars)
        self.operator_head = nn.Linear(params.dense_output_size, num_operators)

    def forward(self, x, get_operator_head=True):
        x = self.encoder(x)
        if get_operator_head:
            return self.statement_head(x), torch.sigmoid(self.drop_head(x)), self.operator_head(x)
        else:
            return self.statement_head(x), torch.sigmoid(self.drop_head(x))

    def predict(self, x):
        statement_pred, drop_pred, _ = self.forward(x)
        statement_probs = F.softmax(statement_pred, dim=1).data
        drop_indx = np.argmax(drop_pred.data.cpu().numpy(), axis=-1)
        return np.argsort(statement_probs.cpu().numpy()), statement_probs, drop_indx
