# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from apex import amp
import torch
import torch.nn as nn
from parts.features import FeatureFactory
from helpers import Optimization
import random


jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


class AudioPreprocessing(nn.Module):
    """GPU accelerated audio preprocessing
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)    # For PyTorch API
        self.optim_level = kwargs.get('optimization_level', Optimization.nothing)
        self.featurizer = FeatureFactory.from_config(kwargs)

    def forward(self, x):
        input_signal, length = x
        length.requires_grad_(False)
        if self.optim_level not in  [Optimization.nothing, Optimization.mxprO0, Optimization.mxprO3]:
            with amp.disable_casts():
                processed_signal = self.featurizer(x)
                processed_length = self.featurizer.get_seq_len(length)
        else:
                processed_signal = self.featurizer(x)
                processed_length = self.featurizer.get_seq_len(length)
        return processed_signal, processed_length

class SpectrogramAugmentation(nn.Module):
    """Spectrogram augmentation
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.spec_cutout_regions = SpecCutoutRegions(kwargs)
        self.spec_augment = SpecAugment(kwargs)

    @torch.no_grad()
    def forward(self, input_spec):
        augmented_spec = self.spec_cutout_regions(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec

class SpecAugment(nn.Module):
    """Spec augment. refer to https://arxiv.org/abs/1904.08779
    """
    def __init__(self, cfg):
        super(SpecAugment, self).__init__()
        self.cutout_x_regions = cfg.get('cutout_x_regions', 0)
        self.cutout_y_regions = cfg.get('cutout_y_regions', 0)

        self.cutout_x_width = cfg.get('cutout_x_width', 10)
        self.cutout_y_width = cfg.get('cutout_y_width', 10)

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape).byte()
        for idx in range(sh[0]):
            for _ in range(self.cutout_x_regions):
                cutout_x_left = int(random.uniform(0, sh[1] - self.cutout_x_width))

                mask[idx, cutout_x_left:cutout_x_left + self.cutout_x_width, :] = 1

            for _ in range(self.cutout_y_regions):
                cutout_y_left = int(random.uniform(0, sh[2] - self.cutout_y_width))

                mask[idx, :, cutout_y_left:cutout_y_left + self.cutout_y_width] = 1

        x = x.masked_fill(mask.to(device=x.device), 0)

        return x

class SpecCutoutRegions(nn.Module):
    """Cutout. refer to https://arxiv.org/pdf/1708.04552.pdf
    """
    def __init__(self, cfg):
        super(SpecCutoutRegions, self).__init__()

        self.cutout_rect_regions = cfg.get('cutout_rect_regions', 0)
        self.cutout_rect_time = cfg.get('cutout_rect_time', 5)
        self.cutout_rect_freq = cfg.get('cutout_rect_freq', 20)

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape).byte()

        for idx in range(sh[0]):
            for i in range(self.cutout_rect_regions):
                cutout_rect_x = int(random.uniform(
                        0, sh[1] - self.cutout_rect_freq))
                cutout_rect_y = int(random.uniform(
                        0, sh[2] - self.cutout_rect_time))

                mask[idx, cutout_rect_x:cutout_rect_x + self.cutout_rect_freq,
                         cutout_rect_y:cutout_rect_y + self.cutout_rect_time] = 1

        x = x.masked_fill(mask.to(device=x.device), 0)

        return x


SUPPORTED_RNNS = {
    'gru': nn.GRU,
    'lstm': nn.LSTM,
    'rnn': nn.RNN
}


class OverLastDim(nn.Module):
    """Collapses a tensor to 2D, applies a module, and (re-)expands the tensor.

    An n-dimensional tensor of shape (s_1, s_2, ..., s_n) is first collapsed to
    a tensor with shape (s_1*s_2*...*s_n-1, s_n). The module is called with
    this as input producing (s_1*s_2*...*s_n-1, s_n') --- note that the final
    dimension can change. This is expanded to (s_1, s_2, ..., s_n-1, s_n') and
    returned.

    Args:
        module (nn.Module): Module to apply. Must accept a 2D tensor as input
            and produce a 2D tensor as output, optionally changing the size of
            the last dimension.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        *dims, input_size = x.size()

        reduced_dims = 1
        for dim in dims:
            reduced_dims *= dim

        x = x.view(reduced_dims, -1)
        x = self.module(x)
        x = x.view(*dims, -1)
        return x



class Lambda(nn.Module):
    """A `torch.nn.Module` wrapper for a Python lambda."""
    def __init__(self, lambda_fn, lambda_fn_desc=""):
        super().__init__()
        self.lambda_fn = lambda_fn
        self.lambda_fn_desc = lambda_fn_desc

    def forward(self, x):
        return self.lambda_fn(x)

    def extra_repr(self):
        if self.lambda_fn_desc:
            return f"lambda_fn={self.lambda_fn_desc}"
        return ""


def lstm(input_size, hidden_size, num_layers, batch_first, dropout,
         forget_gate_bias, bidirectional=False):
    """Returns an LSTM with forget gate bias init to `forget_gate_bias`.

    Args:
        input_size: See `torch.nn.LSTM`.
        hidden_size: See `torch.nn.LSTM`.
        num_layers: See `torch.nn.LSTM`.
        batch_first: See `torch.nn.LSTM`.
        dropout: See `torch.nn.LSTM`.
        forget_gate_bias: For each layer and each direction, the total value of
            to initialise the forget gate bias to.
        bidirectional: See `torch.nn.LSTM`.

    Returns:
        A `torch.nn.LSTM`.
    """
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional
    )
    if forget_gate_bias is not None:
        for name, v in lstm.named_parameters():
            if "bias_ih" in name:
                bias = getattr(lstm, name)
                bias.data[hidden_size:2*hidden_size].fill_(forget_gate_bias)
            if "bias_hh" in name:
                bias = getattr(lstm, name)
                bias.data[hidden_size:2*hidden_size].fill_(0)

    return lstm


class RNNLayer(nn.Module):
    """A single RNNLayer with optional batch norm and bidir summation."""
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM,
                 bidirectional=False, lookahead_context=None, batch_norm=True,
                 batch_first=False, forget_gate_bias=1.0, relu_clip=20.0):
        super().__init__()
        self.bidirectional = bidirectional

        if batch_norm:
            self.bn = OverLastDim(nn.BatchNorm1d(input_size))

        if isinstance(rnn_type, nn.LSTM) and not batch_norm:
            # batch_norm will apply bias, no need to add a second to LSTM
            self.rnn = lstm(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=batch_first,
                            forget_gate_bias=forget_gate_bias,
                            bidirectional=bidirectional)
        else:
            self.rnn = rnn_type(input_size=input_size,
                                hidden_size=hidden_size,
                                bidirectional=bidirectional,
                                batch_first=batch_first,
                                bias=not batch_norm)

        if lookahead_context is not None and bidirectional is False:
            raise NotImplementedError()

    def forward(self, x, hx=None):
        if hasattr(self, 'bn'):
            x = self.bn(x)
        x, h = self.rnn(x, hx=hx)
        if self.bidirectional:
            # TxNx(H*2) -> TxNxH by sum.
            seq_len, batch_size, _ = x.size()
            x = x.view(seq_len, batch_size, 2, -1) \
                 .sum(dim=2) \
                 .view(seq_len, batch_size, -1)
        return x, h

    def _flatten_parameters(self):
        self.rnn.flatten_parameters()

class BNRNNSum(nn.Module):
    """RNN wrapper with optional batch norm and bidir summation.

    Instantiates an RNN. If it is an LSTM it initialises the forget gate
    bias =`lstm_gate_bias`. Optionally applies a batch normalisation layer to
    the input with the statistics computed over all time steps. If the RNN is
    bidirectional, the output from the forward and backward units is summed
    before return. If dropout > 0 then it is applied to all layer outputs
    except the last.
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM,
                 bidirectional=False, lookahead_context=None,
                 rnn_layers=1, batch_norm=True, batch_first=False,
                 dropout=0.0, forget_gate_bias=1.0, relu_clip=20.0):
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn_layers = rnn_layers

        self.layers = torch.nn.ModuleList()
        for i in range(rnn_layers):
            final_layer = (rnn_layers - 1) == i

            self.layers.append(
                RNNLayer(
                    input_size,
                    hidden_size,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                    lookahead_context=lookahead_context,
                    batch_norm=batch_norm and i > 0,
                    batch_first=batch_first,
                    forget_gate_bias=forget_gate_bias,
                    relu_clip=relu_clip
                )
            )

            if dropout > 0.0 and not final_layer:
                self.layers.append(torch.nn.Dropout(dropout))

            input_size = hidden_size

    def forward(self, x, hx=None):
        if hx is not None and self.bidirectional:
            raise NotImplementedError(
                "hidden state not implemented for bidirectional RNNs"
            )

        hx = self._parse_hidden_state(hx)

        hs = []
        cs = []
        rnn_idx = 0
        for layer in self.layers:
            if isinstance(layer, torch.nn.Dropout):
                x = layer(x)
            else:
                x, h_out = layer(x, hx=hx[rnn_idx])
                hs.append(h_out[0])
                cs.append(h_out[1])
                rnn_idx += 1

        h_0 = torch.stack(hs, dim=0)
        c_0 = torch.stack(cs, dim=0)
        return x, (h_0, c_0)

    def _parse_hidden_state(self, hx):
        """
        Dealing w. hidden state:
        Typically in pytorch: (h_0, c_0)
            h_0 = ``[num_layers * num_directions, batch, hidden_size]``
            c_0 = ``[num_layers * num_directions, batch, hidden_size]``
        """
        if hx is None:
            return [None] * self.rnn_layers
        else:
            h_0, c_0 = hx
            assert h_0.shape[0] == self.rnn_layers
            return [(h_0[i], c_0[i]) for i in range(h_0.shape[0])]

    def _flatten_parameters(self):
        for layer in self.layers:
            if isinstance(layer, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
                layer._flatten_parameters()


def label_collate(labels):
    """Collates the label inputs for the rnn-t prediction network.

    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(
            f"`labels` should be a list or tensor not {type(labels)}"
        )

    batch_size = len(labels)
    max_len = max(len(l) for l in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, :len(l)] = l
    labels = torch.LongTensor(cat_labels)

    return labels


class RNNT(torch.nn.Module):
    """A Recurrent Neural Network Transducer (RNN-T).

    Args:
        in_features: Number of input features per step per batch.
        vocab_size: Number of output symbols (not including blank).
        relu_clip: ReLU clamp value: `min(max(0, x), relu_clip)`.
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
        drop_prob: Dropout probability in the encoder and decoder. Must be in
            [0. 1.].
        batch_norm: Use batch normalization in encoder and prediction network
            if true.
        encoder_n_hidden: Internal hidden unit size of the encoder.
        encoder_rnn_layers: Encoder number of layers.
        pred_n_hidden:  Internal hidden unit size of the prediction network.
        pred_rnn_layers: Prediction network number of layers.
        joint_n_hidden: Internal hidden unit size of the joint network.
        rnn_type: string. Type of rnn in SUPPORTED_RNNS.

    Examples:
        >>> Network(in_features=320, vocab_size=28)
        Network(
          (encoder): Sequential(
            (0): Linear(in_features=320, out_features=1152, bias=True)
            (1): Hardtanh(min_val=0.0, max_val=20.0)
            (2): Dropout(p=0.25)
            (3): Linear(in_features=1152, out_features=1152, bias=True)
            (4): Hardtanh(min_val=0.0, max_val=20.0)
            (5): Dropout(p=0.25)
            (6): BNRNNSum(
              (layers): ModuleList(
                (0): RNNLayer(
                  (rnn): LSTM(1152, 1152)
                )
                (1): Dropout(p=0.25)
                (2): RNNLayer(
                  (rnn): LSTM(1152, 1152)
                )
              )
            )
            (7): Lambda(lambda_fn=Access RNN output)
            (8): Linear(in_features=1152, out_features=1152, bias=True)
            (9): Hardtanh(min_val=0.0, max_val=20.0)
            (10): Dropout(p=0.25)
            (11): Linear(in_features=1152, out_features=512, bias=True)
            (12): Hardtanh(min_val=0.0, max_val=20.0)
            (13): Dropout(p=0.25)
          )
          (prediction): ModuleDict(
            (dec_rnn): BNRNNSum(
              (layers): ModuleList(
                (0): RNNLayer(
                  (rnn): LSTM(256, 256, batch_first=True)
                )
                (1): Dropout(p=0.25)
                (2): RNNLayer(
                  (rnn): LSTM(256, 256, batch_first=True)
                )
              )
            )
            (embed): Embedding(28, 256)
          )
          (joint_net): Sequential(
            (0): Linear(in_features=768, out_features=512, bias=True)
            (1): Hardtanh(min_val=0.0, max_val=20.0)
            (2): Dropout(p=0.25)
            (3): Linear(in_features=512, out_features=29, bias=True)
          )
        )

    """
    def __init__(self, rnnt=None, num_classes=1, **kwargs):
        super().__init__()
        if kwargs.get("no_featurizer", False):
            self.audio_preprocessor = None
            in_features = kwargs.get("in_features")
        else:
            feat_config = kwargs.get("feature_config")
            self.audio_preprocessor = AudioPreprocessing(**feat_config)
            in_features = feat_config['features'] * feat_config.get("frame_splicing", 1)
        self.data_spectr_augmentation = SpectrogramAugmentation(**kwargs.get("feature_config"))

        self._pred_n_hidden = rnnt['pred_n_hidden']
        self.encoder = self._encoder(
            in_features,
            rnnt["encoder_n_hidden"],
            rnnt["encoder_rnn_layers"],
            rnnt["joint_n_hidden"],
            rnnt["forget_gate_bias"],
            rnnt["drop_prob"],
            rnnt["batch_norm"],
            rnnt["rnn_type"],
            rnnt["relu_clip"]
        )

        self.prediction = self._predict(
            num_classes,
            rnnt["pred_n_hidden"],
            rnnt["pred_rnn_layers"],
            rnnt["forget_gate_bias"],
            rnnt["drop_prob"],
            rnnt["batch_norm"],
            rnnt["rnn_type"],
        )

        self.joint_net = self._joint_net(
            num_classes, rnnt["pred_n_hidden"], rnnt["joint_n_hidden"], rnnt["drop_prob"], rnnt["relu_clip"]
        )

    def _encoder(self, in_features, encoder_n_hidden, encoder_rnn_layers,
                 joint_n_hidden, forget_gate_bias, drop_prob, batch_norm,
                 rnn_type, relu_clip):
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, encoder_n_hidden),
            torch.nn.Hardtanh(min_val=0.0, max_val=relu_clip),
            torch.nn.Dropout(p=drop_prob),
            torch.nn.Linear(encoder_n_hidden, encoder_n_hidden),
            torch.nn.Hardtanh(min_val=0.0, max_val=relu_clip),
            torch.nn.Dropout(p=drop_prob),
            BNRNNSum(
                input_size=encoder_n_hidden,
                hidden_size=encoder_n_hidden,
                rnn_type=SUPPORTED_RNNS[rnn_type],
                bidirectional=False,
                lookahead_context=None,
                rnn_layers=encoder_rnn_layers,
                batch_norm=batch_norm,
                batch_first=False,
                dropout=drop_prob,
                forget_gate_bias=forget_gate_bias,
            ),
            Lambda(lambda x: x[0], lambda_fn_desc="Access RNN output"),
            torch.nn.Linear(encoder_n_hidden, encoder_n_hidden),
            torch.nn.Hardtanh(min_val=0.0, max_val=relu_clip),
            torch.nn.Dropout(p=drop_prob),
            torch.nn.Linear(encoder_n_hidden, joint_n_hidden),
            torch.nn.Hardtanh(min_val=0.0, max_val=relu_clip),
            torch.nn.Dropout(p=drop_prob),
        )

    def _predict(self, vocab_size, pred_n_hidden, pred_rnn_layers,
                 forget_gate_bias, drop_prob, batch_norm, rnn_type):
        return torch.nn.ModuleDict({
            "embed": torch.nn.Embedding(vocab_size, pred_n_hidden),
            "dec_rnn": BNRNNSum(
                input_size=pred_n_hidden,
                hidden_size=pred_n_hidden,
                rnn_type=SUPPORTED_RNNS[rnn_type],
                bidirectional=False,
                lookahead_context=None,
                rnn_layers=pred_rnn_layers,
                batch_norm=batch_norm,
                batch_first=False,
                dropout=drop_prob,
                forget_gate_bias=forget_gate_bias
            )
        })

    def _joint_net(self, vocab_size, pred_n_hidden, joint_n_hidden, drop_prob,
                   relu_clip):
        return torch.nn.Sequential(
            torch.nn.Linear(pred_n_hidden + joint_n_hidden, joint_n_hidden),
            torch.nn.Hardtanh(min_val=0.0, max_val=relu_clip),
            torch.nn.Dropout(p=drop_prob),
            torch.nn.Linear(joint_n_hidden, vocab_size + 1)
        )

    def forward(self, batch, state=None):
        # batch: ((x, y), (x_lens, y_lens))

        # x: (B, channels, features, seq_len)
        (x, y), (x_lens, y_lens) = batch
        y = label_collate(y)

        # Apply optional preprocessing
        if self.audio_preprocessor is not None:
            x, x_len = self.audio_preprocessor(x)
        # Apply optional spectral augmentation
        if self.training:
            x = self.data_spectr_augmentation(input_spec=x)

        batch, channels, features, seq_len = x.shape
        x = x.view(batch, channels*features, seq_len).permute(2, 0, 1)

        f = self.encode(x)
        g, _ = self.predict(y, state)
        out = self.joint(f, g)

        return out, (x_lens, y_lens)

    def encode(self, x):
        """
        Args:
            x: (T, B, I)

        Returns:
            f: (B, T, H)
        """
        return self.encoder(x).transpose(0, 1)

    def predict(self, y, state=None, add_sos=True):
        """
        B - batch size
        U - label length
        H - Hidden dimension size
        L - Number of decoder layers = 2

        Args:
            y: (B, U)

        Returns:
            Tuple (g, hid) where:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is
                    the final cell state:
                        h (tensor), shape (L, B, H)
                        c (tensor), shape (L, B, H)
        """
        if y is not None:
            y = self.prediction["embed"](y)
        else:
            B = 1 if state is None else state[0].size(1)
            y = torch.zeros((1, B, self._pred_n_hidden)).to(device=self.encoder[0].weight.device, dtype=self.encoder[0].weight.dtype)

        # preprend blank "start of sequence" symbol
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H)).to(device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()   # (B, U + 1, H)
        else:
            start = None   # makes del call later easier

        y = y.transpose(0, 1).contiguous()   # (U + 1, B, H)
        self.prediction["dec_rnn"]._flatten_parameters()
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1).contiguous()   # (B, U + 1, H)
        del y, start, state
        return g, hid

    def joint(self, f, g):
        """
        f should be shape (B, T, H)
        g should be shape (B, U + 1, H)

        returns:
            logits of shape (B, T, U, K + 1)
        """
        # Combine the input states and the output states
        B, T, H = f.shape
        B, U_, H2 = g.shape

        f = f.unsqueeze(dim=2)   # (B, T, 1, H)
        f = f.expand((B, T, U_, H))

        g = g.unsqueeze(dim=1)   # (B, 1, U + 1, H)
        g = g.expand((B, T, U_, H2))

        inp = torch.cat([f, g], dim=3)   # (B, T, U, 2H)
        res = self.joint_net(inp)
        del f, g, inp
        return res
