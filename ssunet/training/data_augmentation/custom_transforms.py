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

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class RemoveKeyTransform(AbstractTransform):
    def __init__(self, key_to_remove):
        self.key_to_remove = key_to_remove

    def __call__(self, **data_dict):
        _ = data_dict.pop(self.key_to_remove, None)
        return data_dict


class MaskTransform(AbstractTransform):
    def __init__(
        self,
        dct_for_where_it_was_used,
        mask_idx_in_seg=1,
        set_outside_to=0,
        data_key="data",
        seg_key="seg",
    ):
        """
        data[mask < 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        print(data_dict.keys())
        seg = data_dict.get(self.seg_key)
        print(self.seg_key, seg.shape)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning(
                "mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist"
            )
        data = data_dict.get(self.data_key)
        print(self.data_key, data.shape)
        print(self.dct_for_where_it_was_used)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    data[b, c][mask < 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict


def convert_3d_to_2d_generator(data_dict, data_key="data", label_key=None):
    shp = data_dict[data_key].shape
    data_dict[data_key] = data_dict[data_key].reshape(
        (shp[0], shp[1] * shp[2], shp[3], shp[4])
    )
    data_dict["orig_shape_data"] = shp
    if label_key is not None:
        shp = data_dict[label_key].shape
        data_dict[label_key] = data_dict[label_key].reshape(
            (shp[0], shp[1] * shp[2], shp[3], shp[4])
        )
        data_dict["orig_shape_seg"] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict, data_key="data", label_key=None):
    shp = data_dict["orig_shape_data"]
    current_shape = data_dict[data_key].shape
    data_dict[data_key] = data_dict[data_key].reshape(
        (shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1])
    )
    if label_key is not None:
        shp = data_dict["orig_shape_seg"]
        current_shape_seg = data_dict[label_key].shape
        data_dict[label_key] = data_dict[label_key].reshape(
            (shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1])
        )
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key=None):
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict, self.data_key, self.label_key)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key=None):
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict, self.data_key, self.label_key)


class ConvertSegmentationToRegionsTransform(AbstractTransform):
    def __init__(
        self,
        regions: dict,
        seg_key: str = "seg",
        output_key: str = "seg",
        seg_channel: int = 0,
    ):
        """
        regions are tuple of tuples where each inner tuple holds the class indices that are merged into one region, example:
        regions= ((1, 2), (2, )) will result in 2 regions: one covering the region of labels 1&2 and the other just 2
        :param regions:
        :param seg_key:
        :param output_key:
        """
        self.seg_channel = seg_channel
        self.output_key = output_key
        self.seg_key = seg_key
        self.regions = regions

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = np.zeros(output_shape, dtype=seg.dtype)
            for b in range(seg_shp[0]):
                for r, k in enumerate(self.regions.keys()):
                    for l in self.regions[k]:
                        region_output[b, r][seg[b, self.seg_channel] == l] = 1
            data_dict[self.output_key] = region_output
        return data_dict


class MoveSegAsOneHotToData(AbstractTransform):
    def __init__(
        self,
        channel_id,
        all_seg_labels,
        key_origin="seg",
        key_target="data",
        remove_from_origin=True,
    ):
        self.remove_from_origin = remove_from_origin
        self.all_seg_labels = all_seg_labels
        self.key_target = key_target
        self.key_origin = key_origin
        self.channel_id = channel_id

    def __call__(self, **data_dict):
        origin = data_dict.get(self.key_origin)
        target = data_dict.get(self.key_target)
        seg = origin[:, self.channel_id : self.channel_id + 1]
        seg_onehot = np.zeros(
            (seg.shape[0], len(self.all_seg_labels), *seg.shape[2:]), dtype=seg.dtype
        )
        for i, l in enumerate(self.all_seg_labels):
            seg_onehot[:, i][seg[:, 0] == l] = 1
        target = np.concatenate((target, seg_onehot), 1)
        data_dict[self.key_target] = target

        if self.remove_from_origin:
            remaining_channels = [
                i for i in range(origin.shape[1]) if i != self.channel_id
            ]
            origin = origin[:, remaining_channels]
            data_dict[self.key_origin] = origin
        return data_dict


class ScaledNoiseTransform(AbstractTransform):
    def __init__(
        self,
        noise_variance=0.8,
        gamma=0.95,
        p_per_sample=1,
        data_key="data",
        return_noise_vec=False,
    ):
        """
        Adds additive Gaussian Noise
        :param noise_variance: variance is uniformly sampled from that range
        :param p_per_sample:
        :param per_channel: if True, each channel will get its own variance sampled from noise_variance
        :param data_key:
        CAREFUL: This transform will modify the value range of your data!
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.noise_variance = noise_variance
        self.gamma = gamma
        self.return_noise_vec = return_noise_vec

    def __call__(self, **data_dict):
        noise_variance = self.ensure_tuple(self.noise_variance)
        gamma = self.ensure_tuple(self.gamma)
        data_dict[self.data_key + "_target"] = np.copy(data_dict[self.data_key])
        if self.return_noise_vec:
            data_dict[self.data_key + "_noisevec"] = np.zeros_like(
                data_dict[self.data_key][:, 0][:, None]
            )
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                if self.return_noise_vec:
                    (
                        data_dict[self.data_key][b],
                        data_dict[self.data_key + "_noisevec"][b],
                    ) = self.augment_scaled_gaussian_noise(
                        data_dict[self.data_key][b], noise_variance, gamma, True
                    )
                else:
                    data_dict[self.data_key][b] = self.augment_scaled_gaussian_noise(
                        data_dict[self.data_key][b], noise_variance, gamma, False
                    )
        return data_dict

    def ensure_tuple(self, arg):
        if isinstance(arg, (float, int)):
            return (arg, arg)
        else:
            return arg

    def augment_scaled_gaussian_noise(
        self, data_sample, noise_variance=0.8, gamma=0.95, return_noise_vec=False
    ):
        variance = self.sample_dist(noise_variance)
        gamma = self.sample_dist(gamma)

        noise_vec = np.random.normal(0.0, variance, size=data_sample[0].shape)
        for c in range(data_sample.shape[0]):
            data_sample[c] = (
                np.sqrt(gamma) * data_sample[c] + np.sqrt(1 - gamma) * noise_vec
            )
        return (data_sample, noise_vec) if return_noise_vec else data_sample

    def sample_dist(self, arg):
        return arg[0] if arg[0] == arg[1] else random.uniform(arg[0], arg[1])


class TestTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        print()
        for key in data_dict.keys():
            print(key + ": ", type(data_dict[key]))
            try:
                print(data_dict[key].shape)
            except:
                pass
        return data_dict
