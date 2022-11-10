import torch
from torch import nn
import copy
import logging
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
# helpers
from vit_pytorch import lossZoo
import numpy as np

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


import ml_collections
config = ml_collections.ConfigDict()
config.patches = ml_collections.ConfigDict({'size': (16, 16)})
config.hidden_size = 128


config.transformer = ml_collections.ConfigDict()
config.transformer.mlp_dim = 200
config.transformer.num_heads = 8
config.transformer.num_layers = 6
config.transformer.attention_dropout_rate = 0.0
config.transformer.dropout_rate = 0.1
config.classifier = 'token'
config.representation_size = None


msa_layer=6






class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, posi_emb=None, ad_net=None, is_source=False):
        # print('Attention==========')
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        if posi_emb is not None:
            eps = 1e-10
            batch_size = key_layer.size(0)
            patch = key_layer
            # print(patch[:, :, 1:].shape)
            ad_out, loss_ad = lossZoo.adv_local(patch[:, :, 1:], ad_net)
            entropy = - ad_out * torch.log2(ad_out + eps) - (1.0 - ad_out) * torch.log2(1.0 - ad_out + eps)
            entropy = torch.cat(
                (torch.ones(batch_size, self.num_attention_heads, 1).to(hidden_states.device).float(), entropy), 2)
            trans_ability = entropy
            entropy = entropy.view(batch_size, self.num_attention_heads, 1, -1)
            attention_probs = torch.cat(
                (attention_probs[:, :, 0, :].unsqueeze(2) * entropy, attention_probs[:, :, 1:, :]), 2)

        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        if posi_emb is not None:
            return attention_output, loss_ad, weights, trans_ability
        else:
            return attention_output, weights










class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        # print('Mlp==========')
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x













class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)                                      ##Attention

    def forward(self, x, posi_emb=None, ad_net=None, is_source=False):
        # print('Block==========')
        h = x
        x = self.attention_norm(x)
        if posi_emb is not None:
            x, loss_ad, weights, tran_weights = self.attn(x, posi_emb, ad_net, is_source)
        else:
            x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        if posi_emb is not None:
            return x, loss_ad, weights, tran_weights
        else:
            return x, weights












class Encoder(nn.Module):
    def __init__(self, config, msa_layer=msa_layer):
        super(Encoder, self).__init__()

        self.msa_layer = msa_layer
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, posi_emb, ad_net, is_source=False):
        # print('Encoder==========')

        for i, layer_block in enumerate(self.layer):
            if i == (self.msa_layer - 1):

                hidden_states, loss_ad, weights, tran_weights = layer_block(hidden_states, posi_emb, ad_net, is_source)
            else:

                hidden_states, weights = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        # print('encoded:',encoded.shape)
        # print('loss_ad:', loss_ad.shape)
        return encoded, loss_ad, tran_weights








class ViT(nn.Module):
    def __init__(self, *,pool = 'cls', num_patches, dim, emb_dropout = 0.):
        super().__init__()


        num_patches = num_patches
        dim = dim
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)


        self.encoder = Encoder(config, msa_layer)


        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, 100),
            nn.BatchNorm1d(100),

        )

    def forward(self, img_s, img_t, ad_net_local):

        x = img_s.view(-1,16,128)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embedding_x = self.pos_embedding[:, :(n + 1)]
        x += pos_embedding_x
        x = self.dropout(x)

        y = img_t.view(-1,16,128)
        b, n, _ = y.shape
        cls_tokens_y = repeat(self.cls_token, '() n d -> b n d', b = b)
        y = torch.cat((cls_tokens_y, y), dim=1)
        pos_embedding_y = self.pos_embedding[:, :(n + 1)]
        y += pos_embedding_y
        y = self.dropout(y)


        encoded_s, loss_ad_s, tran_weights_s = self.encoder(x, pos_embedding_x, ad_net=ad_net_local)
        encoded_t, loss_ad_t, tran_weights_y = self.encoder(y, pos_embedding_y, ad_net=ad_net_local)
        encoded_s =encoded_s[:, 0,:]
        encoded_t = encoded_t[:, 0, :]

        return encoded_s, encoded_t, (loss_ad_s + loss_ad_t) / 2.0







def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1




class AdversarialNetwork(nn.Module):
    def __init__(self, input_size_co, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(input_size_co, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        if self.training and x.requires_grad:
            x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]





class FC(nn.Module):

    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(128, 50)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(50, 16)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)

        return x




class AdversarialNetwork_consistency(nn.Module):
    def __init__(self, in_feature_co, hidden_size_co):
        super(AdversarialNetwork_consistency, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature_co, hidden_size_co)
        self.ad_layer2 = nn.Linear(hidden_size_co, hidden_size_co)
        self.ad_layer3 = nn.Linear(hidden_size_co, 3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        if self.training and x.requires_grad:
            x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]
