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
from ssunet.training.network_training.DenoisingPreTrainer import DenoisingPreTrainer
from batchgenerators.utilities.file_and_folder_operations import *


class MSEDenoiser(DenoisingPreTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, output_folder=None, dataset_directory=None,
                 unpack_data=True, deterministic=True, fp16=False,
                 freeze_encoder=False, freeze_decoder=True, extractor=True, 
                 noisevec=False, skip=True, deep_sup=False):
        super().__init__(plans_file, output_folder, dataset_directory, unpack_data,
                         deterministic, fp16, freeze_encoder, freeze_decoder, 
                         extractor, noisevec, skip, deep_sup)

        self.load_plans_file()
        self.process_plans(self.plans)

        self.mse = torch.nn.MSELoss()

    def loss(self, input, target):
        l = self.mse(input, target)
        return l
