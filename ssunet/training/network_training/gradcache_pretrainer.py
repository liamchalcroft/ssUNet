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


from _warnings import warn
from typing import Tuple
import sys

import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
from ssunet.network_architecture.neural_network import SegmentationNetwork
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler

matplotlib.use("agg")
from time import time, sleep
import torch
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from abc import abstractmethod
from datetime import datetime
from tqdm import trange
from ssunet.utilities.to_torch import maybe_to_torch, to_cuda

from grad_cache.functional import cached, cat_input_tensor
from solo.utils.knn import WeightedKNNClassifier

class GradCachePreTrainer(object):
    def __init__(self, deterministic=True, fp16=False):
        """
        A generic class that can train almost any neural network (RNNs excluded). It provides basic functionality such
        as the training loop, tracking of training and validation losses (and the target metric if you implement it)
        Training can be terminated early if the validation loss (or the target metric if implemented) do not improve
        anymore. This is based on a moving average (MA) of the loss/metric instead of the raw values to get more smooth
        results.

        What you need to override:
        - __init__
        - initialize
        - run_online_evaluation (optional)
        - finish_online_evaluation (optional)
        - validate
        - predict_test_case
        """
        self.fp16 = fp16
        self.amp_grad_scaler = None
        self.clip_grad = None
        self.freeze_encoder = False
        self.freeze_decoder = False
        self.extractor = False
        self.noisevec = False
        self.metabatch = 1


        if deterministic:
            np.random.seed(12345)
            torch.manual_seed(12345)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        ################# SET THESE IN self.initialize() ###################################
        self.network: Tuple[SegmentationNetwork, nn.DataParallel] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = None
        self.was_initialized = False

        ################# SET THESE IN INIT ################################################
        self.output_folder = None
        self.fold = None
        self.dataset_directory = None

        ################# SET THESE IN LOAD_DATASET OR DO_SPLIT ############################
        self.dataset = None  # these can be None for inference mode
        self.dataset_tr = None  # do not need to be used, they just appear if you are using the suggested load_dataset_and_do_split

        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.max_num_epochs = 1000
        self.num_batches_per_epoch = 250
        self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold

        ################# LEAVE THESE ALONE ################################################
        self.train_loss_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.knn_acc = []
        self.knn_acc_item = []
        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic

        self.use_progress_bar = True
        if 'nnunet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnunet_use_progress_bar']))

        ################# Settings for saving checkpoints ##################################
        self.save_every = 50
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

    @abstractmethod
    def initialize(self, training=True):
        """
        create self.output_folder

        modify self.output_folder if you are doing cross-validation (one folder per fold)

        set self.tr_gen and self.val_gen

        call self.initialize_network and self.initialize_optimizer_and_scheduler (important!)

        finally set self.was_initialized to True
        :param training:
        :return:
        """

    @abstractmethod
    def load_dataset(self):
        pass

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(1,2,(2 if self.extractor else 1))

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend()

            if self.extractor:
                ax2 = fig.add_subplot(122)
                ax2.plot(x_values, self.knn_acc, color='b', ls='-', label="knn_acc")

                ax2.set_xlabel("epoch")
                ax2.set_ylabel("kNN accuracy")
                ax2.legend()

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.knn_acc)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def load_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(join(self.output_folder, "model_best.model")):
            self.load_checkpoint(join(self.output_folder, "model_best.model"), train=train)
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            self.load_latest_checkpoint(train)

    def load_latest_checkpoint(self, train=True):
        if isfile(join(self.output_folder, "model_final_checkpoint.model")):
            return self.load_checkpoint(join(self.output_folder, "model_final_checkpoint.model"), train=train)
        if isfile(join(self.output_folder, "model_latest.model")):
            return self.load_checkpoint(join(self.output_folder, "model_latest.model"), train=train)
        if isfile(join(self.output_folder, "model_best.model")):
            return self.load_best_checkpoint(train)
        raise RuntimeError("No checkpoint found")

    def load_final_checkpoint(self, train=False):
        filename = join(self.output_folder, "model_final_checkpoint.model")
        if not isfile(filename):
            raise RuntimeError("Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
        return self.load_checkpoint(filename, train=train)

    def load_checkpoint(self, fname, train=True):
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train)

    @abstractmethod
    def initialize_network(self):
        """
        initialize self.network here
        :return:
        """
        pass

    @abstractmethod
    def initialize_optimizer_and_scheduler(self):
        """
        initialize self.optimizer and self.lr_scheduler (if applicable) here
        :return:
        """
        pass

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if train:
                if 'amp_grad_scaler' in checkpoint.keys():
                    self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.knn_acc = checkpoint[
            'plot_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.knn_acc = self.knn_acc[:self.epoch]

        self._maybe_init_amp()

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()

    def plot_network_architecture(self):
        """
        can be implemented (see nnUNetTrainer) but does not have to. Not implemented here because it imposes stronger
        assumptions on the presence of class variables
        :return:
        """
        pass

    def run_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)
        self.plot_network_architecture()

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for step, b in enumerate(tbar):
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        data_dict = next(self.tr_gen)

                        data1 = data_dict['data1']
                        data2 = data_dict['data2']
                        data1 = maybe_to_torch(data1)
                        data2 = maybe_to_torch(data2)
                        if torch.cuda.is_available():
                            data1 = to_cuda(data1)
                            data2 = to_cuda(data2)

                        self.optimizer.zero_grad()

                        outcache1 = []
                        outcache2 = []

                        if self.fp16:
                            with autocast():
                                outcache1.append(self.model(self.network,data1))
                                outcache2.append(self.model(self.network,data2))
                                del data1, data2
                                if step % self.metabatch == 0:
                                    l = self.loss(outcache1, outcache2)
                                    self.amp_grad_scaler.scale(l).backward()
                                    if self.clip_grad is not None:
                                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)
                                    self.amp_grad_scaler.step(self.optimizer)
                                    self.amp_grad_scaler.update()
                        else:
                            outcache1.append(self.model(self.network,data1))
                            outcache2.append(self.model(self.network,data2))
                            del data1, data2
                            if step % self.metabatch == 0:
                                l = self.loss(outcache1, outcache2)
                                l.backward()
                                if self.clip_grad is not None:
                                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)
                                self.optimizer.step()

                        if step % self.metabatch == 0:
                            self.run_online_knn(torch.cat(outcache1, dim=0), torch.cat(outcache2, dim=0))
                        
                        del outcache1, outcache2

                        if step % self.metabatch == 0:
                            tbar.set_postfix(loss=l.detach().cpu().numpy())
                            train_losses_epoch.append(l)
            else:
                for step, _ in enumerate(range(self.num_batches_per_epoch)):
                    data_dict = next(self.tr_gen)

                    data1 = data_dict['data1']
                    data2 = data_dict['data2']
                    data1 = maybe_to_torch(data1)
                    data2 = maybe_to_torch(data2)
                    if torch.cuda.is_available():
                        data1 = to_cuda(data1)
                        data2 = to_cuda(data2)

                    self.optimizer.zero_grad()

                    outcache1 = []
                    outcache2 = []

                    if self.fp16:
                        with autocast():
                            outcache1.append(self.model(self.network,data1))
                            outcache2.append(self.model(self.network,data2))
                            del data1, data2
                            if step % self.metabatch == 0:
                                l = self.loss(outcache1, outcache2)
                                self.amp_grad_scaler.scale(l).backward()
                                if self.clip_grad is not None:
                                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)
                                self.amp_grad_scaler.step(self.optimizer)
                                self.amp_grad_scaler.update()
                    else:
                        outcache1.append(self.model(self.network,data1))
                        outcache2.append(self.model(self.network,data2))
                        del data1, data2
                        if step % self.metabatch == 0:
                            l = self.loss(outcache1, outcache2)
                            l.backward()
                            if self.clip_grad is not None:
                                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)
                            self.optimizer.step()

                    if step % self.metabatch == 0:
                        self.run_online_knn(torch.cat(outcache1, dim=0), torch.cat(outcache2, dim=0))
                    
                    del outcache1, outcache2

                    if step % self.metabatch == 0:
                        train_losses_epoch.append(l)

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
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.print_to_log_file("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, "model_latest.model"))
            self.print_to_log_file("done")

    def run_online_knn(self, out1, out2):
        with torch.no_grad():
            # pool both views
            out1 = torch.mean(out1.view(out1.size(0), out1.size(1), -1), dim=2)
            out2 = torch.mean(out2.view(out2.size(0), out2.size(1), -1), dim=2)
            # gt targets are the patient ID... kNN trained on latent proj of view 1 should predict the same for view 2
            target = torch.Tensor(list(range(out1.shape[0]))).long()
            knn = WeightedKNNClassifier(k=2) # may want to play around with number of neighbours...
            knn(
                train_features = out1,
                train_targets = target,
                test_features = out2,
                test_targets = target
            )
            acc, _ = knn.compute()
            del knn, out1, out2
            self.knn_acc_item.append(acc)

    def finish_online_knn(self):
        self.knn_acc.append(np.mean(self.knn_acc_item))

        self.print_to_log_file("Average kNN accuracy:", self.knn_acc[-1])
        self.print_to_log_file("(interpret this as an estimate of separation in latent space.)")

        self.knn_acc_item = []

    def on_epoch_end(self):
        self.finish_online_knn()

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        continue_training = True
        return continue_training

    @cat_input_tensor
    def loss(self):
        # To be defined for each loss' custom trainer class
        raise NotImplementedError

    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    @cached
    def call_model(self, model, x):
        return model(x)

    def find_lr(self, num_iters=1000, init_value=1e-6, final_value=10., beta=0.98):
        """
        stolen and adapted from here: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        :param num_iters:
        :param init_value:
        :param final_value:
        :param beta:
        :return:
        """
        import math
        self._maybe_init_amp()
        mult = (final_value / init_value) ** (1 / num_iters)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        losses = []
        log_lrs = []

        for batch_num in range(1, num_iters + 1):
            # +1 because this one here is not designed to have negative loss...
            loss = self.run_iteration(self.tr_gen, do_backprop=True, run_online_evaluation=False).data.item() + 1

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

        import matplotlib.pyplot as plt
        lrs = [10 ** i for i in log_lrs]
        fig = plt.figure()
        plt.xscale('log')
        plt.plot(lrs[10:-5], losses[10:-5])
        plt.savefig(join(self.output_folder, "lr_finder.png"))
        plt.close()
        return log_lrs, losses
