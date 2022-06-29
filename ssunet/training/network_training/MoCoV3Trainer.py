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
from ssunet.training.network_training.MomentumPreTrainer import MomentumPreTrainer, GC_MomentumPreTrainer
from ssunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from ssunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *

from solo.utils.momentum import initialize_momentum_params
from gradcache.functional import cat_input_tensor


class MoCoV3Trainer(MomentumPreTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, output_folder=None, dataset_directory=None,
                 unpack_data=True, deterministic=True, fp16=False,
                 freeze_encoder=False, freeze_decoder=True, extractor=True,
                 proj_output_dim=256, proj_hidden_dim=4096, pred_hidden_dim=4096, temperature=0.2):
        super().__init__(plans_file, output_folder, dataset_directory, unpack_data,
                         deterministic, fp16, freeze_encoder, freeze_decoder, extractor)

        self.load_plans_file()
        self.process_plans(self.plans)

        self.projector = nn.Sequential(
            torch.nn.Linear(320 if self.threeD else 480, proj_hidden_dim),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.momentum_projector = nn.Sequential(
            torch.nn.Linear(320 if self.threeD else 480, proj_hidden_dim),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)
        self.momentum_pairs.append((self.projector, self.momentum_projector))

        self.predictor = nn.Sequential(
            torch.nn.Linear(proj_output_dim, pred_hidden_dim),
            torch.nn.BatchNorm1d(pred_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        self.temperature = temperature

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(chain(self.network.parameters(), 
                                                self.projector.parameters(),
                                                self.predictor.parameters()),
                                            self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None

    def loss(self, view1, view2):
        # pool both views
        view1 = torch.mean(view1.view(view1.size(0), view1.size(1), -1), dim=2)
        view2 = torch.mean(view2.view(view2.size(0), view2.size(1), -1), dim=2)

        z1 = self.projector(view1)
        z1 = self.predictor(z1)
        z2 = self.momentum_projector(view2)

        byol_loss = self.mocov3_loss_func(z1,z2,self.temperature)

        del z1, z2, view1, view2

        return byol_loss

    def mocov3_loss_func(self, query, key, temperature=0.2) -> torch.Tensor:
        """Computes MoCo V3's loss given a batch of queries from view 1, a batch of keys from view 2 and a
        queue of past elements.
        Args:
            query (torch.Tensor): NxD Tensor containing the queries from view 1.
            key (torch.Tensor): NxD Tensor containing the keys from view 2.
            temperature (float, optional): temperature of the softmax in the contrastive
                loss. Defaults to 0.2.
        Returns:
            torch.Tensor: MoCo loss.
        """

        device = query.device

        query = torch.nn.functional.normalize(query, dim=1)
        key = torch.nn.functional.normalize(key, dim=1)

        logits = torch.einsum("nc,mc->nm", [query, key]) / temperature
        labels = torch.arange(query.size(0), dtype=torch.long, device=device)

        return torch.nn.functional.cross_entropy(logits, labels) * (2 * temperature)

class GC_MoCoV3Trainer(GC_MomentumPreTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, output_folder=None, dataset_directory=None,
                 unpack_data=True, deterministic=True, fp16=False,
                 freeze_encoder=False, freeze_decoder=True, extractor=True,
                 proj_output_dim=256, proj_hidden_dim=4096, pred_hidden_dim=4096, temperature=0.2, metabatch=8):
        super().__init__(plans_file, output_folder, dataset_directory, unpack_data,
                         deterministic, fp16, freeze_encoder, freeze_decoder, extractor)

        self.load_plans_file()
        self.process_plans(self.plans)
        self.metabatch = metabatch

        self.projector = nn.Sequential(
            torch.nn.Linear(320 if self.threeD else 480, proj_hidden_dim),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.momentum_projector = nn.Sequential(
            torch.nn.Linear(320 if self.threeD else 480, proj_hidden_dim),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)
        self.momentum_pairs.append((self.projector, self.momentum_projector))

        self.predictor = nn.Sequential(
            torch.nn.Linear(proj_output_dim, pred_hidden_dim),
            torch.nn.BatchNorm1d(pred_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        self.temperature = temperature

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(chain(self.network.parameters(), 
                                                self.projector.parameters(),
                                                self.predictor.parameters()),
                                            self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None

    @cat_input_tensor
    def loss(self, view1, view2):
        # pool both views
        view1 = torch.mean(view1.view(view1.size(0), view1.size(1), -1), dim=2)
        view2 = torch.mean(view2.view(view2.size(0), view2.size(1), -1), dim=2)

        z1 = self.projector(view1)
        z1 = self.predictor(z1)
        z2 = self.momentum_projector(view2)

        byol_loss = self.mocov3_loss_func(z1,z2,self.temperature)

        del z1, z2, view1, view2

        return byol_loss

    def mocov3_loss_func(self, query, key, temperature=0.2) -> torch.Tensor:
        """Computes MoCo V3's loss given a batch of queries from view 1, a batch of keys from view 2 and a
        queue of past elements.
        Args:
            query (torch.Tensor): NxD Tensor containing the queries from view 1.
            key (torch.Tensor): NxD Tensor containing the keys from view 2.
            temperature (float, optional): temperature of the softmax in the contrastive
                loss. Defaults to 0.2.
        Returns:
            torch.Tensor: MoCo loss.
        """

        device = query.device

        query = torch.nn.functional.normalize(query, dim=1)
        key = torch.nn.functional.normalize(key, dim=1)

        logits = torch.einsum("nc,mc->nm", [query, key]) / temperature
        labels = torch.arange(query.size(0), dtype=torch.long, device=device)

        return torch.nn.functional.cross_entropy(logits, labels) * (2 * temperature)
