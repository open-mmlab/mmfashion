from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from ..registry import TYPESPECIFICNET


class ListModule(nn.Module):

    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


@TYPESPECIFICNET.register_module
class TypeSpecificNet(nn.Module):

    def __init__(self,
                 learned,
                 n_conditions,
                 rand_typespaces=False,
                 use_fc=True,
                 l2_embed=False,
                 dim_embed=256,
                 prein=False):
        """init

        Args:
            learned: boolean, indicating whether masks are learned or fixed
            n_conditions: Integer defining number of different similarity
                notions
            use_fc: When true a fully connected layer is learned to transform
                the general embedding to the type specific embedding
            l2_embed: When true we l2 normalize the output type specific
                embeddings
            prein: boolean, indicating whether masks are initialized in equally
                sized disjoint sections or random otherwise
        """
        super(TypeSpecificNet, self).__init__()
        assert (
            (learned == True and use_fc == False)  # noqa
            or (learned == False and use_fc == True)  # noqa
        ), "learn a metric or use fc layer to transform the general embeddings, only one can be true."  # noqa
        self.learnedmask = learned

        # Indicates that there isn't a 1:1 relationship between type specific
        # spaces and pairs of items categories
        if rand_typespaces:
            n_conditions = int(
                np.ceil(n_conditions / float(args.num_rand_embed)))  # noqa

        self.fc_masks = use_fc
        self.l2_norm = l2_embed

        if self.fc_masks:
            # learn a fully connected layer rather than a mask to project the
            # general embedding into the type specific space
            masks = []
            for i in range(n_conditions):
                masks.append(nn.Linear(dim_embed, dim_embed))
            self.masks = ListModule(*masks)
        else:
            # create the mask
            if self.learnedmask:
                if prein:
                    self.masks = nn.Embedding(n_conditions, dim_embed)
                    mask_array = np.zeros([n_conditions, dim_embed])
                    mask_array.fill(0.1)
                    mask_len = int(dim_embed / n_conditions)
                    for i in range(n_conditions):
                        mask_array[i, i * mask_len:(i + 1) * mask_len] = 1
                    # no gradients for the masks
                    self.masks.weight = nn.Parameter(
                        torch.Tensor(mask_array), requires_grad=True)
                else:
                    # define masks with gradients
                    self.masks = nn.Embedding(n_conditions, dim_embed)
                    # initialize weights
                    self.masks.weight.data.normal_(0.9, 0.7)  # 0.1, 0.005
            else:
                # define masks
                self.masks = nn.Embedding(n_conditions, dim_embed)
                # initialize masks
                mask_array = np.zeros([n_conditions, dim_embed])
                mask_len = int(dim_embed / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i * mask_len:(i + 1) * mask_len] = 1
                # no gradients for the masks
                self.masks.weight = nn.Parameter(
                    torch.Tensor(mask_array), requires_grad=False)

    def forward_test(self, embed_x):
        if self.fc_masks:
            masked_embedding = []
            for mask in self.masks:
                masked_embedding.append(mask(embed_x).unsqueeze(1))

            masked_embedding = torch.cat(masked_embedding, 1)
            embedded_x = embed_x.unsqueeze(1)
        else:
            masks = Variable(self.masks.weight.data)
            masks = masks.unsqueeze(0).repeat(embed_x.size(0), 1, 1)
            embedded_x = embed_x.unsqueeze(1)
            masked_embedding = embedded_x.expand_as(masks) * masks

        if self.l2_norm:
            norm = torch.norm(masked_embedding, p=2, dim=2) + 1e-10
            norm.unsqueeze_(2)
            masked_embedding = masked_embedding / norm.expand_as(
                masked_embedding)

        return torch.cat((masked_embedding, embedded_x), 1)

    def forward_train(self, embed_x, c=None):
        """forward_train.

        Args:
            embed_x: feature embeddings.
            c: type specific embedding to compute for the images, returns all
                embeddings when None including the general embedding
                concatenated onto the end.
        """
        if self.fc_masks:
            mask_norm = 0.
            masked_embedding = []
            for embed, condition in zip(embed_x, c):
                mask = self.masks[condition]
                masked_embedding.append(mask(embed.unsqueeze(0)))
                mask_norm += mask.weight.norm(1)

            masked_embedding = torch.cat(masked_embedding)
        else:
            self.mask = self.masks(c)
            if self.learnedmask:
                self.mask = torch.nn.functional.relu(self.mask)

            masked_embedding = embed_x * self.mask
            mask_norm = self.mask.norm(1)

        embed_norm = embed_x.norm(2)
        if self.l2_norm:
            norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
            norm.unsqueeze_(1)
            masked_embedding = masked_embedding / norm

        return masked_embedding, mask_norm, embed_norm

    def forward(self, embed_x, c=None, return_loss=True):
        if return_loss:
            return self.forward_train(embed_x, c)
        else:
            return self.forward_test(embed_x)

    def init_weights(self):
        if isinstance(self.masks, nn.Sequential):
            for m in self.masks:
                if type(m) == nn.Linear:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.01)
        elif isinstance(self.masks, nn.Module):
            for m in self.masks.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.01)
