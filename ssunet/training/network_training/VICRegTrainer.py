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


from collections import OrderedDict
from typing import Tuple

from itertools import chain

import numpy as np
import torch
from ssunet.training.data_augmentation.data_augmentation_contrastive import get_contrastive_augmentation
from ssunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from ssunet.utilities.to_torch import maybe_to_torch, to_cuda
from ssunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet
from ssunet.network_architecture.initialization import InitWeights_He
from ssunet.network_architecture.neural_network import SegmentationNetwork
from ssunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from ssunet.training.dataloading.dataset_loading import unpack_dataset
from ssunet.training.network_training.ContrastivePreTrainer import ContrastivePreTrainer
from ssunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from ssunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *

import sys
assert 'solo' in sys.modules, "solo-learn module not installed! Please go to https://github.com/vturrisi/solo-learn and follow the package installation instructions."
from solo.losses.vicreg import vicreg_loss_func


class VICRegTrainer(ContrastivePreTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold='all', output_folder=None, dataset_directory=None,
                 unpack_data=True, deterministic=True, fp16=False,
                 freeze_encoder=False, freeze_decoder=True, extractor=True,
                 proj_output_dim=2048, proj_hidden_dim=2048,
                 sim_loss_weight=25.0, var_loss_weight=25.0, cov_loss_weight=1.0):
        super().__init__(plans_file, fold, output_folder, dataset_directory, unpack_data,
                         deterministic, fp16, freeze_encoder, freeze_decoder, extractor)

        self.load_plans_file()
        self.process_plans(self.plans)

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(320 if self.threeD else 480, proj_hidden_dim),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_hidden_dim),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(chain(self.network.parameters(), self.projector.parameters()),
                                            self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None

    def loss(self, view1, view2):
        # pool both views
        view1 = torch.mean(view1.view(view1.size(0), view1.size(1), -1), dim=2)
        view2 = torch.mean(view2.view(view2.size(0), view2.size(1), -1), dim=2)

        z1 = self.projector(view1)
        z2 = self.projector(view2)

        vic_loss = vicreg_loss_func(z1,z2,self.sim_loss_weight,self.var_loss_weight,self.cov_loss_weight)

        del z1, z2

        return vic_loss
