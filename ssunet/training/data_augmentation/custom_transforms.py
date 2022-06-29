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
import random
from batchgenerators.transforms.abstract_transforms import AbstractTransform


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


class ScaledNoiseTransform(AbstractTransform):
    def __init__(self, noise_variance=0.8, gamma=0.95, p_per_sample=1, data_key="data", return_noise_vec=False):
        """
        Adds additive Gaussian Noise
        :param noise_variance: variance is uniformly sampled from that range
        :param p_per_sample:
        :param p_per_channel:
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
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                if self.return_noise_vec==True:
                    data_dict[self.data_key][b], data_dict[self.data_key+'_noisevec'][b] =\
                        self.augment_scaled_gaussian_noise(data_dict[self.data_key][b], 
                                                noise_variance, gamma,
                                                self.p_per_channel, self.per_channel, True)
                else:
                    data_dict[self.data_key][b] = self.augment_scaled_gaussian_noise(data_dict[self.data_key][b], 
                                                                        noise_variance, gamma,
                                                                        self.p_per_channel, self.per_channel)
        return data_dict

    def ensure_tuple(self, arg):
        if isinstance(arg, (float, int)):
            return (arg, arg)
        else:
            return arg

    def augment_scaled_gaussian_noise(self, data_sample, noise_variance=0.8,
                                gamma=0.95, return_noise_vec=False):
        variance = self.sample_dist(noise_variance)
        gamma = self.sample_dist(gamma)
        
        noise_vec = np.random.normal(0.0, variance, size=data_sample[c].shape)
        for c in range(data_sample.shape[0]):
            data_sample[c] = np.sqrt(gamma) * data_sample[c] +\
                np.sqrt(1-gamma) * noise_vec
        return (data_sample, noise_vec) if return_noise_vec else data_sample

    def sample_dist(self, arg):
        return arg[0] if arg[0] == arg[1] else random.uniform(arg[0], arg[1])
