from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import params
from cuda import LongTensor
from model.model import BaseModel
from dsl.impl import ALL_FUNCTIONS, LAMBDAS
from dsl.types import LIST, INT, FunctionType
from env.statement import Statement
from baseline.robustfill.env import program_vocab, program_max_seq_len, io_vocab_size, reverse_program_vocab, \
    program_vocab_size, START_PROGRAM_TOKEN, END_PROGRAM_TOKEN

embedding_dim = 128
hidden_size = 512

rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)


class RobustFill(BaseModel):
    def __init__(self):
        super(RobustFill, self).__init__()
        self.input_encoder = IOEncoder()
        self.output_encoder = IOEncoder(attention=True)
        self.decoder = Decoder()

    def encode(self, input, input_lens, output, output_lens, input_padding_mask):
        input_encoder_output, input_encoder_hidden = self.input_encoder(input, input_lens)

        output_encoder_output, output_encoder_hidden = self.output_encoder(output, output_lens,
                                                                           hidden_state=input_encoder_hidden,
                                                                           attention_outputs=input_encoder_output,
                                                                           attention_padding_mask=input_padding_mask)

        return input_encoder_output, input_encoder_hidden, output_encoder_output, output_encoder_hidden

    def forward(self, input, input_lens, output, output_lens, input_padding_mask, output_padding_mask,
                dec_padding_mask, target, target_lens):
        input_encoder_output, input_encoder_hidden, output_encoder_output, output_encoder_hidden = \
            self.encode(input, input_lens, output, output_lens, input_padding_mask)

        step_losses = []
        hidden = output_encoder_hidden

        start_token = program_vocab[START_PROGRAM_TOKEN]
        decoder_input = torch.cat((LongTensor([start_token]).repeat(target.shape[0], 1), target), dim=1)

        max_len = int(target_lens.max().item())
        context = torch.zeros((input.shape[0] * params.num_examples, hidden_size), device=input.device)

        for i in range(min(program_max_seq_len, max_len)):
            prev_y = decoder_input[:, i].unsqueeze(1).repeat(1, params.num_examples).view(-1)
            out_dist, hidden, context = self.decoder(prev_y, hidden, context, input_encoder_output,
                                                     input_padding_mask, output_encoder_output, output_padding_mask)

            seq_target = target[:, i]
            gold_log_probs = torch.gather(out_dist, 1, seq_target.unsqueeze(1)).squeeze()
            step_loss = -gold_log_probs
            step_mask = dec_padding_mask[:, i]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        normalized_loss = sum_losses / target_lens
        loss = torch.mean(normalized_loss)
        return loss

    def greedy_decode(self, input, input_lens, output, output_lens, input_padding_mask, output_padding_mask):
        batch_size = input.shape[0]

        input_encoder_output, input_encoder_hidden, output_encoder_output, output_encoder_hidden = \
            self.encode(input, input_lens, output, output_lens, input_padding_mask)

        start_token = program_vocab[START_PROGRAM_TOKEN]
        prev_y = LongTensor([start_token]).repeat(batch_size * params.num_examples)

        hidden = output_encoder_hidden
        output = torch.zeros(batch_size, program_max_seq_len, dtype=torch.int)
        context = torch.zeros((input.shape[0] * params.num_examples, hidden_size), device=input.device)
        for i in range(program_max_seq_len):
            out_dist, hidden, context = self.decoder(prev_y, hidden, context, input_encoder_output,
                                                     input_padding_mask, output_encoder_output, output_padding_mask)
            prev_y = out_dist.max(dim=1)[1]
            output[:, i] = prev_y

            # Duplicate for num of examples
            prev_y = prev_y.unsqueeze(1).repeat(1, params.num_examples).view(-1)

        return output

    # An implementation of beam search, with some optimization improvements for the case of RobustFill
    def beam_search(self, env, max_program_len, input, input_lens, output, output_lens,
                    input_padding_mask, output_padding_mask, beam_size, beam_width, state):
        def sort_beams(beams):
            return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

        if time.time() >= state['end_time']:
            return None

        input_encoder_output, input_encoder_hidden, output_encoder_output, output_encoder_hidden = \
            self.encode(input, input_lens, output, output_lens, input_padding_mask)

        c_t_0 = torch.zeros((input.shape[0] * params.num_examples, hidden_size), device=input.device)

        beams = [Beam(tokens=[program_vocab[START_PROGRAM_TOKEN]],
                      log_probs=[0.0],
                      state=output_encoder_hidden,
                      context=c_t_0,
                      env=env)]

        results = []
        steps = 0
        while steps < program_max_seq_len and len(results) < beam_size:
            if len(beams) == 0:
                break

            if time.time() >= state['end_time']:
                return None

            latest_tokens = [h.latest_token for h in beams]
            prev_y = LongTensor(latest_tokens)

            # Duplicate for num of examples
            prev_y = prev_y.unsqueeze(1).repeat(1, params.num_examples).view(-1)

            all_state_h = []
            all_state_c = []
            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            hidden = (torch.cat(all_state_h, dim=1), torch.cat(all_state_c, dim=1))
            context = torch.cat(all_context, 0)

            out_dist, hidden, context = self.decoder(prev_y, hidden, context,
                                                     input_encoder_output.repeat(len(beams), 1, 1),
                                                     input_padding_mask.repeat(len(beams), 1, 1),
                                                     output_encoder_output.repeat(len(beams), 1, 1),
                                                     output_padding_mask.repeat(len(beams), 1, 1))

            expansion_size = min(beam_width, out_dist.shape[-1])
            topk_log_probs, topk_ids = torch.topk(out_dist, expansion_size)

            dec_h, dec_c = hidden

            all_beams = []
            num_orig_beams = len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[:, i: i + params.num_examples], dec_c[:, i: i + params.num_examples])
                context_i = context[i: i + params.num_examples, :]

                for j in range(expansion_size):
                    if len(h.statements) >= max_program_len:
                        return None

                    # We check for this with len(statements) explicitly
                    if topk_ids[i, j].item() == program_vocab[END_PROGRAM_TOKEN]:
                        continue

                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i)
                    if new_beam:
                        if new_beam.env.is_solution():
                            state['num_steps'] += 1
                            return new_beam.statements

                        all_beams.append(new_beam)

            beams = sort_beams(all_beams)[:beam_size]
            steps += 1

        return None


class IOEncoder(nn.Module):
    def __init__(self, attention=False):
        super(IOEncoder, self).__init__()
        self.embedding = nn.Embedding(io_vocab_size, embedding_dim)

        attention_size = hidden_size if attention else 0
        self.lstm = nn.LSTM(embedding_dim + attention_size, hidden_size, num_layers=1,
                            batch_first=True, bidirectional=False)

        if attention:
            self.attention = Attention()

        init_wt_normal(self.embedding.weight)
        init_lstm_wt(self.lstm)

    def forward(self, seq, seq_lens, hidden_state=None, attention_outputs=None, attention_padding_mask=None):
        embedded = self.embedding(seq)

        embedded_flat = embedded.view(embedded.shape[0] * embedded.shape[1], embedded.shape[2], embedded.shape[3])

        # Sort the input to the LSTM by sequence length
        sorted_seq_lens, sorted_seq_lens_ind = seq_lens.view(-1).sort(descending=True)
        input_to_encoder = embedded_flat.clone()[sorted_seq_lens_ind, :, :]

        if attention_outputs is not None:
            assert hidden_state is not None and attention_padding_mask is not None, "Received invalid params!"

            sorted_attention_outputs = attention_outputs.clone()[sorted_seq_lens_ind, :, :]
            sorted_attention_masks = attention_padding_mask.view(-1, attention_padding_mask.shape[2]).clone()[sorted_seq_lens_ind, :]

            # Since we use attention, we have to pass the input through the LSTM token-by-token.
            # We use seq_lens to efficiently pass through the batch
            indx = len(sorted_seq_lens) - 1
            output = torch.zeros((input_to_encoder.shape[0], sorted_seq_lens[0], hidden_size),
                                 device=input_to_encoder.device)
            context = torch.zeros((input_to_encoder.shape[0], 1, hidden_size), device=input_to_encoder.device)
            for step in range(sorted_seq_lens[0].item()):
                while step + 1 > sorted_seq_lens[indx]:
                    indx -= 1

                h = (hidden_state[0][:, :indx+1, :], hidden_state[1][:, :indx+1, :])
                step_input = torch.cat((input_to_encoder[:indx + 1, step: step+1, :], context[:indx+1, :, :]), dim=-1)

                self.lstm.flatten_parameters()
                o, h = self.lstm(step_input, h)

                context = self.attention(torch.cat((h[0], h[1]), 2).view(1, h[0].shape[1], -1).squeeze(0),
                                         sorted_attention_outputs[:indx + 1, :, :],
                                         sorted_attention_masks[:indx + 1, :])
                context = context.unsqueeze(1)

                # Update output and hidden_state with the values returned by the LSTM (only for the relevant indices)
                new_hidden_state = hidden_state[0].clone(), hidden_state[1].clone()
                new_hidden_state[0][:, :indx+1, :] = h[0]
                new_hidden_state[1][:, :indx+1, :] = h[1]
                hidden_state = new_hidden_state
                output[: indx+1, step: step+1, :] = o

        else:
            packed = pack_padded_sequence(input_to_encoder, sorted_seq_lens, batch_first=True)

            self.lstm.flatten_parameters()
            output, hidden_state = self.lstm(packed, hidden_state)

            output, _ = pad_packed_sequence(output, batch_first=True)

        # Reverse the sorting
        indxs_for_output = sorted_seq_lens_ind.unsqueeze(-1).unsqueeze(-1).expand(-1, output.shape[1],
                                                                                  output.shape[2])
        unsorted_output = torch.zeros_like(output)
        unsorted_output.scatter_(0, indxs_for_output, output)

        indxs_for_h = sorted_seq_lens_ind.unsqueeze(0).unsqueeze(-1).expand(hidden_state[0].shape[0], -1,
                                                                            hidden_state[0].shape[2])
        unsorted_h = (torch.zeros_like(hidden_state[0]), torch.zeros_like(hidden_state[1]))
        unsorted_h[0].scatter_(1, indxs_for_h, hidden_state[0])
        unsorted_h[1].scatter_(1, indxs_for_h, hidden_state[1])

        return unsorted_output, unsorted_h


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(program_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers=1, batch_first=True,
                            bidirectional=False)
        self.linear = nn.Linear(hidden_size * 2, program_vocab_size)
        self.input_attention = Attention(hidden_size * 3)
        self.output_attention = Attention()

        init_wt_normal(self.embedding.weight)
        init_lstm_wt(self.lstm)
        init_linear_wt(self.linear)

    def forward(self, prev_y, hidden, context, i_enc_outputs, i_enc_mask, o_enc_outputs, o_enc_mask):
        i_enc_mask = i_enc_mask.view(-1, i_enc_mask.shape[2])
        o_enc_mask = o_enc_mask.view(-1, o_enc_mask.shape[2])

        prev_y_embedded = self.embedding(prev_y)
        input_to_lstm = torch.cat((prev_y_embedded, context), dim=-1).unsqueeze(1)
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(input_to_lstm, hidden)

        hidden_concat = torch.cat((hidden[0], hidden[1]), 2).squeeze(0)
        output_context = self.output_attention(hidden_concat, o_enc_outputs, o_enc_mask)

        input_attn_input = torch.cat((hidden_concat, output_context), dim=-1)
        final_context = self.input_attention(input_attn_input, i_enc_outputs, i_enc_mask)

        out_concat = torch.cat((out, final_context.unsqueeze(1)), dim=-1)
        out_unflatten = out_concat.view(-1, params.num_examples, out_concat.shape[2])
        out_fc = self.linear(out_unflatten)
        out_pooled = out_fc.max(dim=1)[0]
        return F.log_softmax(out_pooled, dim=1), hidden, final_context


class Attention(nn.Module):
    def __init__(self, hidden_dim=hidden_size*2):
        super(Attention, self).__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.decode_proj = nn.Linear(hidden_dim, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

        init_linear_wt(self.W_h)
        init_linear_wt(self.decode_proj)
        init_linear_wt(self.v)

    def forward(self, hidden, output, enc_padding_mask):
        b, t_k, n = list(output.size())
        output = output.view(-1, n)
        encoder_feature = self.W_h(output)

        dec_fea = self.decode_proj(hidden)
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()
        dec_fea_expanded = dec_fea_expanded.view(-1, n)

        att_features = encoder_feature + dec_fea_expanded

        e = torch.tanh(att_features)
        scores = self.v(e)
        scores = scores.view(-1, t_k)

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask[:, :t_k]
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)
        output = output.view(-1, t_k, n)
        c_t = torch.bmm(attn_dist, output)
        c_t = c_t.squeeze(1)

        return c_t

# Contains the necessary information for a beam during beam-search. For further efficiency, we also
# keep the last tokens predicted. When trying to extend the beam with a new token, we ensure that
# the combination of tokens is legal and can result in a valid statement in the DSL.
class Beam(object):
    def __init__(self, tokens, log_probs, state, context, env, last_statement=None, statements=[]):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.env = env
        self.last_statement = last_statement
        self.statements = statements

    def extend(self, token, log_prob, state, context):
        new_beam = Beam(tokens=self.tokens + [token],
                        log_probs=self.log_probs + [log_prob],
                        state=state,
                        context=context,
                        env=self.env.copy(),
                        last_statement=self.last_statement,
                        statements=self.statements[:])
        if not new_beam.extend_last_statement(token):
            return None
        else:
            return new_beam

    def extend_last_statement(self, token):
        token = reverse_program_vocab[token]
        if token == END_PROGRAM_TOKEN:
            return True
        elif self.last_statement is None:
            if token == START_PROGRAM_TOKEN:
                return False
            elif token in ALL_FUNCTIONS:
                self.last_statement = [token]
                return True
            else:
                return False
        else:
            func = self.last_statement[0]
            num_inputs = len(func.input_type) if isinstance(func.input_type, tuple) else 1
            if len(self.last_statement[1:]) == num_inputs:
                if token in ALL_FUNCTIONS:
                    statement = Statement(func, self.last_statement[1:])
                    self.env = self.env.step_safe(statement)
                    self.statements.append(statement)
                    if self.env is None:
                        return False
                    self.last_statement = [token]
                    return True
                else:
                    return False
            else:
                if isinstance(func.input_type, tuple):
                    input_type = func.input_type[len(self.last_statement[1:])]
                else:
                    input_type = func.input_type
                if isinstance(token, int) and (input_type == LIST or input_type == INT):
                    self.last_statement = self.last_statement + [token]
                    return True
                elif token in LAMBDAS and isinstance(input_type, FunctionType):
                    self.last_statement = self.last_statement + [token]
                else:
                    return False


    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)