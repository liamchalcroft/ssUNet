#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from itertools import chain

import torch
from ssunet.training.network_training.ContrastivePreTrainer import ContrastivePreTrainer, GC_ContrastivePreTrainer
from ssunet.training.network_training.custom_layer import BatchNormDimSwap, NOBS
from batchgenerators.utilities.file_and_folder_operations import *

from grad_cache.functional import cat_input_tensor


def arb_loss_func(z1, z2, shuffle=False, groups=1):
    """
    Loss function for 'Align Representations with Base'. Adapted from https://github.com/Sherrylone/Align-Representation-with-Base
    """
    B = z1.size(0)

    z1 = z1.transpose(0,1)
    z2 = z2.transpose(0,1)

    if shuffle:
        idx = torch.randperm(z1.size(0))
        z1 = z1[idx, :]
        z2 = z2[idx, :]
    
    nobsnet = NOBS(groups=groups)

    z1 = (z1 - z1.mean(dim=-1, keepdim=True)) / (z1.std(dim=-1, keepdim=True) + 1e-5)
    z2 = (z2 - z2.mean(dim=-1, keepdim=True)) / (z2.std(dim=-1, keepdim=True) + 1e-5)

    z1_group = nobsnet(z1).detach()
    z2_group = nobsnet(z2).detach()

    c = (z1 * z2_group).sum(dim=-1)
    c.div_(B)
    loss1 = c.add_(-1).pow_(2).sum()

    c = (z2 * z1_group).sum(dim=-1)
    c.div_(B)
    loss2 = c.add_(-1).pow_(2).sum()

    loss = loss1 + loss2
    return loss



class ARBTrainer(ContrastivePreTrainer):
    """
    """

    def __init__(self, plans_file, output_folder=None, dataset_directory=None,
                 unpack_data=True, deterministic=True, fp16=False,
                 freeze_encoder=False, freeze_decoder=True, extractor=True,
                 proj_output_dim=2048, proj_hidden_dim=2048,
                 shuffle=True, groups=1,
                 detcon=False):
        super().__init__(plans_file, output_folder, dataset_directory, unpack_data,
                         deterministic, fp16, freeze_encoder, freeze_decoder, extractor)

        self.load_plans_file()
        self.process_plans(self.plans)
        self.detcon = detcon

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(320 if self.threeD else 480, proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.shuffle = shuffle 
        self.groups = groups

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(chain(self.network.parameters(), self.projector.parameters()),
                                            self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None

    def loss(self, view1, view2, mask1=None, mask2=None):
        if self.detcon: # pool by multiplying images with masks
            view1, view2 = self.detcon_views(view1, view2, mask1, mask2)
        else:
            view1 = view1.view(view1.size(0), view1.size(1), -1).mean(dim=2)
            view2 = view2.view(view2.size(0), view2.size(1), -1).mean(dim=2)

        z1 = self.projector(view1)
        z2 = self.projector(view2)

        if self.detcon=='intra': # treat each class as batch item - separate classes in same image will be treated as separate images
            z1 = z1.view(z1.size(0)*z1.size(1), -1)
            z2 = z2.view(z2.size(0)*z2.size(1), -1)
        elif self.detcon=='inter': # treat each class as batch and original batch as features - same class if diff images treated as same image
            z1 = z1.permute(1,0,2).reshape(z1.size(1), -1)
            z2 = z2.permute(1,0,2).reshape(z2.size(1), -1)

        arb_loss = arb_loss_func(z1,z2,self.shuffle,self.groups)

        del z1, z2, view1, view2

        return arb_loss


class GC_BarlowTrainer(GC_ContrastivePreTrainer):
    """
    """

    def __init__(self, plans_file, output_folder=None, dataset_directory=None,
                 unpack_data=True, deterministic=True, fp16=False,
                 freeze_encoder=False, freeze_decoder=True, extractor=True,
                 proj_output_dim=2048, proj_hidden_dim=2048,
                 shuffle=True, groups=1, metabatch=8,
                 detcon=False):
        super().__init__(plans_file, output_folder, dataset_directory, unpack_data,
                         deterministic, fp16, freeze_encoder, freeze_decoder, extractor)

        self.load_plans_file()
        self.process_plans(self.plans)
        self.metabatch = metabatch
        self.detcon = detcon

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(320 if self.threeD else 480, proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.shuffle = shuffle 
        self.groups = groups

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(chain(self.network.parameters(), self.projector.parameters()),
                                            self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None

    @cat_input_tensor
    def loss(self, view1, view2, mask1=None, mask2=None):
        if self.detcon: # pool by multiplying images with masks
            view1, view2 = self.detcon_views(view1, view2, mask1, mask2)
        else:
            view1 = view1.view(view1.size(0), view1.size(1), -1).mean(dim=2)
            view2 = view2.view(view2.size(0), view2.size(1), -1).mean(dim=2)

        z1 = self.projector(view1)
        z2 = self.projector(view2)

        if self.detcon=='intra': # treat each class as batch item - separate classes in same image will be treated as separate images
            z1 = z1.view(z1.size(0)*z1.size(1), -1)
            z2 = z2.view(z2.size(0)*z2.size(1), -1)
        elif self.detcon=='inter': # treat each class as batch and original batch as features - same class if diff images treated as same image
            z1 = z1.permute(1,0,2).reshape(z1.size(1), -1)
            z2 = z2.permute(1,0,2).reshape(z2.size(1), -1)

        arb_loss = arb_loss_func(z1,z2,self.shuffle,self.groups)

        del z1, z2, view1, view2

        return arb_loss
