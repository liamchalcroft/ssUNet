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

import os
from copy import deepcopy

import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import (
    DataChannelSelectionTransform,
)
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform,
    MirrorTransform,
)
from batchgenerators.transforms.utility_transforms import RenameTransform, NumpyToTensor

from ssunet.training.data_augmentation.custom_transforms import (
    Convert3DTo2DTransform,
    Convert2DTo3DTransform,
    ScaledNoiseTransform,
)


default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,
    "do_elastic": True,
    "elastic_deform_alpha": (0.0, 900.0),
    "elastic_deform_sigma": (9.0, 13.0),
    "p_eldef": 0.2,
    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,
    "do_rotation": True,
    "rotation_x": (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi),
    "rotation_y": (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi),
    "rotation_z": (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
    "do_mirror": True,
    "mirror_axes": (0, 1, 2),
    "dummy_2D": False,
    "mask_was_used_for_normalization": None,
    "border_mode_data": "constant",
    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,
    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,
    "num_threads": 12
    if "nnUNet_n_proc_DA" not in os.environ
    else int(os.environ["nnUNet_n_proc_DA"]),
    "num_cached_per_thread": 1,
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params["elastic_deform_alpha"] = (0.0, 200.0)
default_2D_augmentation_params["elastic_deform_sigma"] = (9.0, 13.0)
default_2D_augmentation_params["rotation_x"] = (
    -180.0 / 360 * 2.0 * np.pi,
    180.0 / 360 * 2.0 * np.pi,
)
default_2D_augmentation_params["rotation_y"] = (
    -0.0 / 360 * 2.0 * np.pi,
    0.0 / 360 * 2.0 * np.pi,
)
default_2D_augmentation_params["rotation_z"] = (
    -0.0 / 360 * 2.0 * np.pi,
    0.0 / 360 * 2.0 * np.pi,
)

# sometimes you have 3d data and a 3d net but cannot augment them properly in 3d due to anisotropy (which is currently
# not supported in batchgenerators). In that case you can 'cheat' and transfer your 3d data into 2d data and
# transform them back after augmentation
default_2D_augmentation_params["dummy_2D"] = False
default_2D_augmentation_params["mirror_axes"] = (
    0,
    1,
)  # this can be (0, 1, 2) if dummy_2D=True


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2.0 * np.pi, rot_x)
    rot_y = min(90 / 360 * 2.0 * np.pi, rot_y)
    rot_z = min(90 / 360 * 2.0 * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d

    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0
        )
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0
        )
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0
        )
    elif len(coords) == 2:
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0
        )
    final_shape /= min(scale_range)
    return final_shape.astype(int)


def get_denoising_augmentation(
    dataloader_train,
    patch_size,
    params=default_3D_augmentation_params,
    border_val_seg=-1,
    pin_memory=True,
    seeds_train=None,
    noisevec=False,
):
    assert (
        params.get("mirror") is None
    ), "old version of params, use new keyword do_mirror"
    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(
            DataChannelSelectionTransform(params.get("selected_data_channels"))
        )

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size

    tr_transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=None,
            do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"),
            angle_x=params.get("rotation_x"),
            angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            do_scale=params.get("do_scaling"),
            scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"),
            border_cval_data=0,
            order_data=3,
            border_mode_seg="constant",
            border_cval_seg=border_val_seg,
            order_seg=1,
            random_crop=params.get("random_crop"),
            p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"),
            p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get(
                "independent_scale_factor_for_each_axis"
            ),
        )
    )
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                False,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=params["p_gamma"],
            )
        )

    if params.get("do_mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    # tr_transforms.append(RenameTransform('data', 'target', False))

    tr_transforms.append(ScaledNoiseTransform(return_noise_vec=noisevec))

    tr_transforms.append(
        NumpyToTensor(
            ["data", "data_target", "data_noisevec"]
            if noisevec
            else ["data", "data_target"],
            "float",
        )
    )

    tr_transforms = Compose(tr_transforms)

    batchgenerator_train = MultiThreadedAugmenter(
        dataloader_train,
        tr_transforms,
        params.get("num_threads"),
        params.get("num_cached_per_thread"),
        seeds=seeds_train,
        pin_memory=pin_memory,
    )

    return batchgenerator_train
