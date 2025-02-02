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
from typing import Tuple, List
from time import time, sleep
from tqdm import trange

import numpy as np
import torch
from ssunet.training.data_augmentation.data_augmentation_contrastive import (
    get_contrastive_augmentation,
)
from ssunet.utilities.to_torch import maybe_to_torch, to_cuda
from ssunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet
from ssunet.network_architecture.initialization import InitWeights_He
from ssunet.network_architecture.neural_network import SegmentationNetwork
from ssunet.training.data_augmentation.default_data_augmentation import (
    default_2D_augmentation_params,
    get_patch_size,
    default_3D_augmentation_params,
)
from ssunet.training.dataloading.dataset_loading import (
    load_dataset,
    DataLoader3D,
    DataLoader2D,
    unpack_dataset,
)
from ssunet.training.network_training.network_pretrainer import NetworkPreTrainer
from ssunet.training.network_training.gradcache_pretrainer import GradCachePreTrainer
from ssunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from ssunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *

import matplotlib

matplotlib.use("agg")

from grad_cache.functional import cached, cat_input_tensor
from solo.utils.momentum import initialize_momentum_params, MomentumUpdater


class MomentumPreTrainer(NetworkPreTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(
        self,
        plans_file,
        output_folder=None,
        dataset_directory=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
        clip_grad=12,
        freeze_encoder=False,
        freeze_decoder=True,
        extractor=True,
        base_tau=0.99,
        final_tau=1.0,
    ):

        super().__init__(deterministic, fp16)

        self.max_num_epochs = 500
        self.initial_lr = 1e-2

        self.pin_memory = True
        self.unpack_data = unpack_data
        self.init_args = (
            plans_file,
            output_folder,
            dataset_directory,
            unpack_data,
            deterministic,
            fp16,
        )

        # set through arguments from init
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.output_folder_base = self.output_folder
        self.clip_grad = clip_grad
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.extractor = extractor
        self.base_tau = base_tau
        self.final_tau = final_tau
        self.detcon = None

        self.momentum_pairs = []

        self.plans = None

        self.last_step = 0

        self.folder_with_preprocessed_data = None

        # set in self.initialize()

        self.dl_tr = self.dl_val = None
        self.num_input_channels = (
            self.num_classes
        ) = (
            self.net_pool_per_axis
        ) = (
            self.patch_size
        ) = (
            self.batch_size
        ) = (
            self.threeD
        ) = (
            self.base_num_features
        ) = (
            self.intensity_properties
        ) = (
            self.normalization_schemes
        ) = (
            self.net_num_pool_op_kernel_sizes
        ) = self.net_conv_kernel_sizes = None  # loaded automatically from plans_file
        self.basic_generator_patch_size = (
            self.data_aug_params
        ) = self.transpose_forward = self.transpose_backward = None

        self.classes = (
            self.do_dummy_2D_aug
        ) = (
            self.use_mask_for_norm
        ) = (
            self.only_keep_largest_connected_component
        ) = self.min_region_size_per_class = self.min_size_per_class = None

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {"constant_values": 0}

        self.pad_all_sides = None

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5

        self.oversample_foreground_percent = 0.33

        self.conv_per_stage = None
        self.regions_class_order = None

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.folder_with_preprocessed_data = join(
                self.plans["preprocessed_data_folder"], self.plans["data_identifier"]
            )
            if training:
                self.dl_tr = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!"
                    )

                self.tr_gen = get_contrastive_augmentation(
                    self.dl_tr,
                    self.data_aug_params["patch_size_for_spatialtransform"],
                    self.data_aug_params,
                    pin_memory=self.pin_memory,
                    detcon=self.detcon,
                )
                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % (str(self.dataset.keys())),
                    also_print_to_console=False,
                )
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )
        self.was_initialized = True

    def initialize_network(self):
        """
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = False (decoder not used here at all...)
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        dropout_op_kwargs = {"p": 0, "inplace": True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        self.network = Generic_UNet(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage,
            2,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            net_nonlin,
            net_nonlin_kwargs,
            False,
            False,
            lambda x: x,
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,
            True,
            True,
            None,
            ConvDropoutNormNonlin,
            False,
            self.freeze_encoder,
            self.freeze_decoder,
            self.extractor,
        )
        self.momentum_network = Generic_UNet(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage,
            2,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            net_nonlin,
            net_nonlin_kwargs,
            False,
            False,
            lambda x: x,
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,
            True,
            True,
            None,
            ConvDropoutNormNonlin,
            False,
            self.freeze_encoder,
            self.freeze_decoder,
            self.extractor,
        )
        if torch.cuda.is_available():
            self.network.cuda()
            self.momentum_network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        initialize_momentum_params(self.network, self.momentum_network)

        self.momentum_updater = MomentumUpdater(self.base_tau, self.final_tau)

        self.momentum_pairs.append((self.network, self.momentum_network))

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay
        )
        self.lr_scheduler = None

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params["rotation_x"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_y"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_z"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params[
                    "elastic_deform_alpha"
                ] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params[
                    "elastic_deform_sigma"
                ] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params[
                    "rotation_x"
                ]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params["rotation_x"] = (
                    -15.0 / 360 * 2.0 * np.pi,
                    15.0 / 360 * 2.0 * np.pi,
                )
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size[1:],
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )
            self.basic_generator_patch_size = np.array(
                [self.patch_size[0]] + list(self.basic_generator_patch_size)
            )
        else:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size,
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params["selected_seg_channels"] = [0]
        self.data_aug_params["patch_size_for_spatialtransform"] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if self.epoch is None:
            ep = self.epoch + 1
        else:
            ep = self.epoch
        self.optimizer.param_groups[0]["lr"] = poly_lr(
            ep, self.max_num_epochs, self.initial_lr, 0.9
        )
        self.print_to_log_file(
            "lr:", np.round(self.optimizer.param_groups[0]["lr"], decimals=6)
        )

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        self.update_momentum()
        continue_training = self.epoch < self.max_num_epochs
        return continue_training

    def update_momentum(self):
        momentum_pairs = self.momentum_pairs
        for mp in momentum_pairs:
            self.momentum_updater.update(*mp)
        self.momentum_updater.update_tau(
            cur_step=self.epoch, max_steps=self.max_num_epochs
        )

    def run_iteration(self, data_generator, do_backprop=True):
        data_dict = next(data_generator)
        data1 = data_dict["data1"]
        data2 = data_dict["data2"]
        data1 = maybe_to_torch(data1)
        data2 = maybe_to_torch(data2)
        if torch.cuda.is_available():
            data1 = to_cuda(data1)
            data2 = to_cuda(data2)
        if self.detcon:
            mask1 = data_dict["mask1"]
            mask2 = data_dict["mask2"]
            mask1 = maybe_to_torch(mask1)
            mask2 = maybe_to_torch(mask2)
            if torch.cuda.is_available():
                mask1 = to_cuda(mask1)
                mask2 = to_cuda(mask2)

        # import matplotlib.pyplot as plt
        # plt.subplot(221)
        # plt.imshow(data1[0,0,50], cmap='gray')
        # plt.axis('off')
        # plt.subplot(222)
        # plt.imshow(data2[0,0,50], cmap='gray')
        # plt.axis('off')
        # if self.detcon:
        #     plt.subplot(223)
        #     plt.imshow(mask1[0,0,50], cmap='jet', vmax=5)
        #     plt.axis('off')
        #     plt.subplot(224)
        #     plt.imshow(mask2[0,0,50], cmap='jet', vmax=5)
        #     plt.axis('off')
        # plt.savefig('/Users/liamchalcroft/Desktop/MRES/ssunet-test/test.png')
        # plt.close()

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output1 = self.network(data1)
                output2 = self.momentum_network(data2)
                del data1, data2
                l = (
                    self.loss(output1, output2, mask1, mask2)
                    if self.detcon
                    else self.loss(output1, output2)
                )

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.clip_grad
                    )
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output1 = self.network(data1)
            output2 = self.momentum_network(data2)
            del data1, data2
            l = (
                self.loss(output1, output2, mask1, mask2)
                if self.detcon
                else self.loss(output1, output2)
            )

            if do_backprop:
                l.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.clip_grad
                    )
                self.optimizer.step()

        if self.extractor:
            self.run_online_knn(output1, output2)

        return l.detach().cpu().numpy()

    def plot_network_architecture(self):
        try:
            from batchgenerators.utilities.file_and_folder_operations import join
            import hiddenlayer as hl

            if torch.cuda.is_available():
                g = hl.build_graph(
                    self.network,
                    torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                    transforms=None,
                )
            else:
                g = hl.build_graph(
                    self.network,
                    torch.rand((1, self.num_input_channels, *self.patch_size)),
                    transforms=None,
                )
            g.save(join(self.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            self.print_to_log_file("Unable to plot network architecture:")
            self.print_to_log_file(e)

            self.print_to_log_file("\nprinting the network instead:\n")
            self.print_to_log_file(self.network)
            self.print_to_log_file("\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_debug_information(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct["plans"]
        del dct["intensity_properties"]
        del dct["dataset"]
        save_json(dct, join(self.output_folder, "debug.json"))

        import shutil

        shutil.copy(self.plans_file, join(self.output_folder_base, "plans.pkl"))

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)

    def process_plans(self, plans):
        assert len(list(plans["plans_per_stage"].keys())) == 1, (
            "If self.stage is None then there can be only one stage in the plans file. That seems to not be the "
            "case. Please specify which stage of the cascade must be trained"
        )
        self.stage = list(plans["plans_per_stage"].keys())[0]
        self.plans = plans

        stage_plans = self.plans["plans_per_stage"][self.stage]
        self.batch_size = stage_plans["batch_size"]
        self.net_pool_per_axis = stage_plans["num_pool_per_axis"]
        self.patch_size = np.array(stage_plans["patch_size"]).astype(int)
        self.do_dummy_2D_aug = stage_plans["do_dummy_2D_data_aug"]

        if "pool_op_kernel_sizes" not in stage_plans.keys():
            assert "num_pool_per_axis" in stage_plans.keys()
            self.print_to_log_file(
                "WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it..."
            )
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans["pool_op_kernel_sizes"]

        if "conv_kernel_sizes" not in stage_plans.keys():
            self.print_to_log_file(
                "WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it..."
            )
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (
                max(self.net_pool_per_axis) + 1
            )
        else:
            self.net_conv_kernel_sizes = stage_plans["conv_kernel_sizes"]

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans["dataset_properties"]["intensityproperties"]
        self.normalization_schemes = plans["normalization_schemes"]
        self.base_num_features = plans["base_num_features"]
        self.num_input_channels = plans["num_modalities"]
        self.num_classes = (
            plans["num_classes"] + 1
        )  # background is no longer in num_classes
        self.classes = plans["all_classes"]
        self.use_mask_for_norm = plans["use_mask_for_norm"]
        self.only_keep_largest_connected_component = plans["keep_only_largest_region"]
        self.min_region_size_per_class = plans["min_region_size_per_class"]
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if (
            plans.get("transpose_forward") is None
            or plans.get("transpose_backward") is None
        ):
            print(
                "WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!"
            )
            plans["transpose_forward"] = [0, 1, 2]
            plans["transpose_backward"] = [0, 1, 2]
        self.transpose_forward = plans["transpose_forward"]
        self.transpose_backward = plans["transpose_backward"]

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError(
                "invalid patch size in plans file: %s" % str(self.patch_size)
            )

        if (
            "conv_per_stage" in plans.keys()
        ):  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans["conv_per_stage"]
        else:
            self.conv_per_stage = 2

    def load_dataset(self):
        print(self.folder_with_preprocessed_data)
        self.dataset = load_dataset(self.folder_with_preprocessed_data)

    def get_basic_generators(self):
        self.load_dataset()

        if self.threeD:
            dl_tr = DataLoader3D(
                self.dataset,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
        else:
            dl_tr = DataLoader2D(
                self.dataset,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
        return dl_tr

    def save_checkpoint(self, fname, save_optimizer=True):
        super().save_checkpoint(fname, save_optimizer)
        info = OrderedDict()
        info["init"] = self.init_args
        info["name"] = self.__class__.__name__
        info["class"] = str(self.__class__)
        info["plans"] = self.plans

        write_pickle(info, fname + ".pkl")

    def detcon_views(self, view1, view2, mask1, mask2):
        if self.threeD:
            pool = torch.nn.functional.adaptive_avg_pool3d
        else:
            pool = torch.nn.functional.adaptive_avg_pool2d

        shape1 = view1.shape[2:]
        shape2 = view2.shape[2:]

        view1 = view1.view(view1.size(0), view1.size(1), -1)
        view2 = view2.view(view2.size(0), view2.size(1), -1)

        if mask1.size(1) == 1 and mask2.size(1) == 1:
            mask1 = torch.nn.functional.one_hot(mask1.long()).float()
            mask2 = torch.nn.functional.one_hot(mask2.long()).float()
            if self.threeD:
                mask1 = mask1.permute(0, 5, 2, 3, 4, 1)[..., 0]
                mask2 = mask2.permute(0, 5, 2, 3, 4, 1)[..., 0]
            else:
                mask1 = mask1.permute(0, 4, 2, 3, 1)[..., 0]
                mask2 = mask2.permute(0, 4, 2, 3, 1)[..., 0]

        ch = mask1.size(1)

        mask1 = pool(mask1, output_size=shape1)
        mask2 = pool(mask2, output_size=shape2)

        mask1 = mask1.view(mask1.size(0), mask1.size(1), -1)
        mask2 = mask2.view(mask2.size(0), mask2.size(1), -1)

        mask1 = mask1.argmax(dim=1)
        mask2 = mask2.argmax(dim=1)

        mask1 = torch.eye(ch, dtype=view1.dtype, device=view1.device)[mask1]
        mask2 = torch.eye(ch, dtype=view2.dtype, device=view2.device)[mask2]

        view1 = mask1.permute(0, 2, 1) @ view1.permute(0, 2, 1)
        view2 = mask2.permute(0, 2, 1) @ view2.permute(0, 2, 1)
        # now have pooled views of shape (B, MASK_CHANS, FEATURE_CHANS)
        # instead of (B, FEATURE_CHANS) for non-detcon method

        return view1, view2


class GC_MomentumPreTrainer(GradCachePreTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(
        self,
        plans_file,
        output_folder=None,
        dataset_directory=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
        clip_grad=12,
        freeze_encoder=False,
        freeze_decoder=True,
        extractor=True,
        base_tau=0.99,
        final_tau=1.0,
    ):

        super().__init__(deterministic, fp16)

        self.max_num_epochs = 500
        self.initial_lr = 1e-2

        self.pin_memory = True
        self.unpack_data = unpack_data
        self.init_args = (
            plans_file,
            output_folder,
            dataset_directory,
            unpack_data,
            deterministic,
            fp16,
        )

        # set through arguments from init
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.output_folder_base = self.output_folder
        self.clip_grad = clip_grad
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.extractor = extractor
        self.base_tau = base_tau
        self.final_tau = final_tau
        self.detcon = None

        self.momentum_pairs = []

        self.plans = None

        self.last_step = 0

        self.folder_with_preprocessed_data = None

        # set in self.initialize()

        self.dl_tr = self.dl_val = None
        self.num_input_channels = (
            self.num_classes
        ) = (
            self.net_pool_per_axis
        ) = (
            self.patch_size
        ) = (
            self.batch_size
        ) = (
            self.threeD
        ) = (
            self.base_num_features
        ) = (
            self.intensity_properties
        ) = (
            self.normalization_schemes
        ) = (
            self.net_num_pool_op_kernel_sizes
        ) = self.net_conv_kernel_sizes = None  # loaded automatically from plans_file
        self.basic_generator_patch_size = (
            self.data_aug_params
        ) = self.transpose_forward = self.transpose_backward = None

        self.classes = (
            self.do_dummy_2D_aug
        ) = (
            self.use_mask_for_norm
        ) = (
            self.only_keep_largest_connected_component
        ) = self.min_region_size_per_class = self.min_size_per_class = None

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {"constant_values": 0}

        self.pad_all_sides = None

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5

        self.oversample_foreground_percent = 0.33

        self.conv_per_stage = None
        self.regions_class_order = None

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.folder_with_preprocessed_data = join(
                self.plans["preprocessed_data_folder"], self.plans["data_identifier"]
            )
            if training:
                self.dl_tr = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!"
                    )

                self.tr_gen = get_contrastive_augmentation(
                    self.dl_tr,
                    self.data_aug_params["patch_size_for_spatialtransform"],
                    self.data_aug_params,
                    pin_memory=self.pin_memory,
                    detcon=self.detcon,
                )
                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % (str(self.dataset.keys())),
                    also_print_to_console=False,
                )
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )
        self.was_initialized = True

    def initialize_network(self):
        """
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = False (decoder not used here at all...)
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        dropout_op_kwargs = {"p": 0, "inplace": True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        self.network = Generic_UNet(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage,
            2,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            net_nonlin,
            net_nonlin_kwargs,
            False,
            False,
            lambda x: x,
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,
            True,
            True,
            None,
            ConvDropoutNormNonlin,
            False,
            self.freeze_encoder,
            self.freeze_decoder,
            self.extractor,
        )
        self.momentum_network = Generic_UNet(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage,
            2,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            net_nonlin,
            net_nonlin_kwargs,
            False,
            False,
            lambda x: x,
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,
            True,
            True,
            None,
            ConvDropoutNormNonlin,
            False,
            self.freeze_encoder,
            self.freeze_decoder,
            self.extractor,
        )
        if torch.cuda.is_available():
            self.network.cuda()
            self.momentum_network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        initialize_momentum_params(self.network, self.momentum_network)

        self.momentum_updater = MomentumUpdater(self.base_tau, self.final_tau)

        self.momentum_pairs.append((self.network, self.momentum_network))

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay
        )
        self.lr_scheduler = None

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params["rotation_x"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_y"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            self.data_aug_params["rotation_z"] = (
                -30.0 / 360 * 2.0 * np.pi,
                30.0 / 360 * 2.0 * np.pi,
            )
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params[
                    "elastic_deform_alpha"
                ] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params[
                    "elastic_deform_sigma"
                ] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params[
                    "rotation_x"
                ]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params["rotation_x"] = (
                    -15.0 / 360 * 2.0 * np.pi,
                    15.0 / 360 * 2.0 * np.pi,
                )
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size[1:],
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )
            self.basic_generator_patch_size = np.array(
                [self.patch_size[0]] + list(self.basic_generator_patch_size)
            )
        else:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size,
                self.data_aug_params["rotation_x"],
                self.data_aug_params["rotation_y"],
                self.data_aug_params["rotation_z"],
                self.data_aug_params["scale_range"],
            )

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params["selected_seg_channels"] = [0]
        self.data_aug_params["patch_size_for_spatialtransform"] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if self.epoch is None:
            ep = self.epoch + 1
        else:
            ep = self.epoch
        self.optimizer.param_groups[0]["lr"] = poly_lr(
            ep, self.max_num_epochs, self.initial_lr, 0.9
        )
        self.print_to_log_file(
            "lr:", np.round(self.optimizer.param_groups[0]["lr"], decimals=6)
        )

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        self.update_momentum()
        continue_training = self.epoch < self.max_num_epochs
        return continue_training

    def update_momentum(self):
        momentum_pairs = self.momentum_pairs
        for mp in momentum_pairs:
            self.momentum_updater.update(*mp)
        self.momentum_updater.update_tau(
            cur_step=self.epoch, max_steps=self.max_num_epochs
        )

    def run_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file(
                "WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)

        if cudnn.benchmark and cudnn.deterministic:
            warn(
                "torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                "If you want deterministic then set benchmark=False"
            )

        if not self.was_initialized:
            self.initialize(True)
        self.plot_network_architecture()

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            outcache1 = []
            outcache2 = []
            if self.detcon:
                maskcache1 = []
                maskcache2 = []

            outcache1 = []
            outcache2 = []
            outfunc1 = []
            outfunc2 = []
            if self.detcon:
                maskcache1 = []
                maskcache2 = []

            trg = (
                trange(self.num_batches_per_epoch)
                if self.use_progress_bar
                else range(self.num_batches_per_epoch)
            )
            with trg as tbar:
                for step, b in enumerate(tbar):
                    if self.use_progress_bar:
                        tbar.set_description(
                            "Epoch {}/{}".format(self.epoch + 1, self.max_num_epochs)
                        )

                    data_dict = next(self.tr_gen)

                    data1 = data_dict["data1"]
                    data2 = data_dict["data2"]
                    data1 = maybe_to_torch(data1)
                    data2 = maybe_to_torch(data2)
                    if torch.cuda.is_available():
                        data1 = to_cuda(data1)
                        data2 = to_cuda(data2)
                    if self.detcon:
                        mask1 = data_dict["mask1"]
                        mask2 = data_dict["mask2"]
                        mask1 = maybe_to_torch(mask1)
                        mask2 = maybe_to_torch(mask2)
                        if torch.cuda.is_available():
                            mask1 = to_cuda(mask1)
                            mask2 = to_cuda(mask2)

                    self.optimizer.zero_grad()

                    if self.detcon:
                        maskcache1.append(mask1)
                        maskcache2.append(mask2)

                    if self.fp16:
                        with autocast():
                            outc1, outf1 = self.call_model(self.network, data1)
                            outc2, outf2 = self.call_model(self.momentum_network, data2)
                            outcache1.append(outc1)
                            outcache2.append(outc2)
                            outfunc1.append(outf1)
                            outfunc2.append(outf2)

                            if (step + 1) % self.metabatch == 0:
                                l = (
                                    self.loss(
                                        outcache1, outcache2, maskcache1, maskcache2
                                    )
                                    if self.detcon
                                    else self.loss(outcache1, outcache2)
                                )
                                self.amp_grad_scaler.scale(l).backward()
                                for f, c in zip(outfunc1, outcache1):
                                    f(c)
                                for f, c in zip(outfunc2, outcache2):
                                    f(c)
                                if self.clip_grad is not None:
                                    torch.nn.utils.clip_grad_norm_(
                                        self.network.parameters(), self.clip_grad
                                    )
                                self.amp_grad_scaler.step(self.optimizer)
                                self.amp_grad_scaler.update()
                    else:
                        outc1, outf1 = self.call_model(self.network, data1)
                        outc2, outf2 = self.call_model(self.momentum_network, data2)
                        outcache1.append(outc1)
                        outcache2.append(outc2)
                        outfunc1.append(outf1)
                        outfunc2.append(outf2)
                        if (step + 1) % self.metabatch == 0:
                            l = (
                                self.loss(outcache1, outcache2, maskcache1, maskcache2)
                                if self.detcon
                                else self.loss(outcache1, outcache2)
                            )
                            l.backward()
                            if self.clip_grad is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.network.parameters(), self.clip_grad
                                )
                            for f, c in zip(outfunc1, outcache1):
                                f(c)
                            for f, c in zip(outfunc2, outcache2):
                                f(c)
                            self.optimizer.step()

                    if (step + 1) % self.metabatch == 0:
                        self.run_online_knn(
                            torch.cat(outcache1, dim=0), torch.cat(outcache2, dim=0)
                        )
                        del outcache1, outcache2, outfunc1, outfunc2, data1, data2
                        outcache1 = []
                        outcache2 = []
                        outfunc1 = []
                        outfunc2 = []
                        if self.detcon:
                            del maskcache1, maskcache2
                            maskcache1 = []
                            maskcache2 = []
                        if self.use_progress_bar:
                            tbar.set_postfix(loss=l.detach().cpu().numpy())
                        train_losses_epoch.append(l.detach().cpu().numpy())

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()
            if self.freeze_decoder and self.extractor:
                self.print_to_log_file("kNN loss: %.4f" % self.knn_acc[-1])

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file(
                "This epoch took %f s\n" % (epoch_end_time - epoch_start_time)
            )

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint:
            self.save_checkpoint(
                join(self.output_folder, "model_final_checkpoint.model")
            )
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    def plot_network_architecture(self):
        try:
            from batchgenerators.utilities.file_and_folder_operations import join
            import hiddenlayer as hl

            if torch.cuda.is_available():
                g = hl.build_graph(
                    self.network,
                    torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                    transforms=None,
                )
            else:
                g = hl.build_graph(
                    self.network,
                    torch.rand((1, self.num_input_channels, *self.patch_size)),
                    transforms=None,
                )
            g.save(join(self.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            self.print_to_log_file("Unable to plot network architecture:")
            self.print_to_log_file(e)

            self.print_to_log_file("\nprinting the network instead:\n")
            self.print_to_log_file(self.network)
            self.print_to_log_file("\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_debug_information(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct["plans"]
        del dct["intensity_properties"]
        del dct["dataset"]
        save_json(dct, join(self.output_folder, "debug.json"))

        import shutil

        shutil.copy(self.plans_file, join(self.output_folder_base, "plans.pkl"))

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)

    def process_plans(self, plans):
        assert len(list(plans["plans_per_stage"].keys())) == 1, (
            "If self.stage is None then there can be only one stage in the plans file. That seems to not be the "
            "case. Please specify which stage of the cascade must be trained"
        )
        self.stage = list(plans["plans_per_stage"].keys())[0]
        self.plans = plans

        stage_plans = self.plans["plans_per_stage"][self.stage]
        self.batch_size = stage_plans["batch_size"]
        self.net_pool_per_axis = stage_plans["num_pool_per_axis"]
        self.patch_size = np.array(stage_plans["patch_size"]).astype(int)
        self.do_dummy_2D_aug = stage_plans["do_dummy_2D_data_aug"]

        if "pool_op_kernel_sizes" not in stage_plans.keys():
            assert "num_pool_per_axis" in stage_plans.keys()
            self.print_to_log_file(
                "WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it..."
            )
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans["pool_op_kernel_sizes"]

        if "conv_kernel_sizes" not in stage_plans.keys():
            self.print_to_log_file(
                "WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it..."
            )
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (
                max(self.net_pool_per_axis) + 1
            )
        else:
            self.net_conv_kernel_sizes = stage_plans["conv_kernel_sizes"]

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans["dataset_properties"]["intensityproperties"]
        self.normalization_schemes = plans["normalization_schemes"]
        self.base_num_features = plans["base_num_features"]
        self.num_input_channels = plans["num_modalities"]
        self.num_classes = (
            plans["num_classes"] + 1
        )  # background is no longer in num_classes
        self.classes = plans["all_classes"]
        self.use_mask_for_norm = plans["use_mask_for_norm"]
        self.only_keep_largest_connected_component = plans["keep_only_largest_region"]
        self.min_region_size_per_class = plans["min_region_size_per_class"]
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if (
            plans.get("transpose_forward") is None
            or plans.get("transpose_backward") is None
        ):
            print(
                "WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!"
            )
            plans["transpose_forward"] = [0, 1, 2]
            plans["transpose_backward"] = [0, 1, 2]
        self.transpose_forward = plans["transpose_forward"]
        self.transpose_backward = plans["transpose_backward"]

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError(
                "invalid patch size in plans file: %s" % str(self.patch_size)
            )

        if (
            "conv_per_stage" in plans.keys()
        ):  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans["conv_per_stage"]
        else:
            self.conv_per_stage = 2

    def load_dataset(self):
        print(self.folder_with_preprocessed_data)
        self.dataset = load_dataset(self.folder_with_preprocessed_data)

    def get_basic_generators(self):
        self.load_dataset()

        if self.threeD:
            dl_tr = DataLoader3D(
                self.dataset,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
        else:
            dl_tr = DataLoader2D(
                self.dataset,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
        return dl_tr

    def save_checkpoint(self, fname, save_optimizer=True):
        super().save_checkpoint(fname, save_optimizer)
        info = OrderedDict()
        info["init"] = self.init_args
        info["name"] = self.__class__.__name__
        info["class"] = str(self.__class__)
        info["plans"] = self.plans

        write_pickle(info, fname + ".pkl")

    def detcon_views(self, view1, view2, mask1, mask2):
        if self.threeD:
            pool = torch.nn.functional.adaptive_avg_pool3d
        else:
            pool = torch.nn.functional.adaptive_avg_pool2d

        shape1 = view1.shape[2:]
        shape2 = view2.shape[2:]

        view1 = view1.view(view1.size(0), view1.size(1), -1)
        view2 = view2.view(view2.size(0), view2.size(1), -1)

        if mask1.size(1) == 1 and mask2.size(1) == 1:
            mask1 = torch.nn.functional.one_hot(mask1.long()).float()
            mask2 = torch.nn.functional.one_hot(mask2.long()).float()
            if self.threeD:
                mask1 = mask1.permute(0, 5, 2, 3, 4, 1)[..., 0]
                mask2 = mask2.permute(0, 5, 2, 3, 4, 1)[..., 0]
            else:
                mask1 = mask1.permute(0, 4, 2, 3, 1)[..., 0]
                mask2 = mask2.permute(0, 4, 2, 3, 1)[..., 0]

        ch = mask1.size(1)

        mask1 = pool(mask1, output_size=shape1)
        mask2 = pool(mask2, output_size=shape2)

        mask1 = mask1.view(mask1.size(0), mask1.size(1), -1)
        mask2 = mask2.view(mask2.size(0), mask2.size(1), -1)

        mask1 = mask1.argmax(dim=1)
        mask2 = mask2.argmax(dim=1)

        mask1 = torch.eye(ch, dtype=view1.dtype, device=view1.device)[mask1]
        mask2 = torch.eye(ch, dtype=view2.dtype, device=view2.device)[mask2]

        view1 = mask1.permute(0, 2, 1) @ view1.permute(0, 2, 1)
        view2 = mask2.permute(0, 2, 1) @ view2.permute(0, 2, 1)
        # now have pooled views of shape (B, MASK_CHANS, FEATURE_CHANS)
        # instead of (B, FEATURE_CHANS) for non-detcon method

        return view1, view2
