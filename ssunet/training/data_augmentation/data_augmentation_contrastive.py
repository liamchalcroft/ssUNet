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

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RenameTransform, NumpyToTensor
from ssunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform
from ssunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params


def get_augs(tr_transforms=[], target_key='data',
            patch_size_spatial=None, params=default_3D_augmentation_params,
            border_val_seg=-1, ignore_axes=False, order_seg=1, order_data=3):
    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis"),
        p_independent_scale_per_axis=params.get("p_independent_scale_per_axis"), data_key=target_key))

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.15, data_key=target_key))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.5), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5, data_key=target_key))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.70, 1.3), p_per_sample=0.15, data_key=target_key))
    tr_transforms.append(ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.15, data_key=target_key))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes, data_key=target_key))
    tr_transforms.append(
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.15, data_key=target_key))  # inverted gamma

    if params.get("do_additive_brightness"):
        tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                 params.get("additive_brightness_sigma"),
                                                 True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                 p_per_channel=params.get("additive_brightness_p_per_channel"), 
                                                 data_key=target_key))

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"], data_key=target_key))

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes"), data_key=target_key))

    return tr_transforms


def get_contrastive_augmentation(dataloader_train, patch_size, params=default_3D_augmentation_params,
                              seeds_train=None, pin_memory=True):

    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    tr_transforms.append(RenameTransform('data', 'data1', False))
    tr_transforms.append(RenameTransform('data', 'data2', True))

    tr_transforms = get_augs(tr_transforms, target_key='data1', patch_size_spatial=patch_size_spatial, ignore_axes=ignore_axes)
    tr_transforms = get_augs(tr_transforms, target_key='data2', patch_size_spatial=patch_size_spatial, ignore_axes=ignore_axes)

    tr_transforms.append(NumpyToTensor(['data1', 'data2'], 'float'))
    tr_transforms = Compose(tr_transforms)

    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                  params.get("num_cached_per_thread"),
                                                  seeds=seeds_train, pin_memory=pin_memory)

    return batchgenerator_train
    