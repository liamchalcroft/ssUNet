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


import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from ssunet.run.default_configuration import get_default_configuration
from ssunet.paths import default_plans_identifier
from ssunet.run.load_pretrained_weights import load_pretrained_weights
from ssunet.training.network_training.network_pretrainer import NetworkPreTrainer
from ssunet.training.network_training.gradcache_pretrainer import GradCachePreTrainer
from ssunet.utilities.task_name_id_conversion import convert_id_to_task_name


class ParseKwargs(argparse.Action):
    """
    Credit to https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument(
        "-c",
        "--continue_training",
        help="use this if you want to continue a training",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        help="plans identifier. Only change this if you created a custom experiment planner",
        default=default_plans_identifier,
        required=False,
    )
    parser.add_argument(
        "--use_compressed_data",
        default=False,
        action="store_true",
        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
        "is much more CPU and RAM intensive and should only be used if you know what you are "
        "doing",
        required=False,
    )
    parser.add_argument(
        "--deterministic",
        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
        "this is not necessary. Deterministic training will make you overfit to some random seed. "
        "Don't use that.",
        required=False,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--npz",
        required=False,
        default=False,
        action="store_true",
        help="if set then nnUNet will "
        "export npz files of "
        "predicted segmentations "
        "in the validation as well. "
        "This is needed to run the "
        "ensembling step so unless "
        "you are developing nnUNet "
        "you should enable this",
    )
    parser.add_argument(
        "--find_lr",
        required=False,
        default=False,
        action="store_true",
        help="not used here, just for fun",
    )
    parser.add_argument(
        "--fp32",
        required=False,
        default=False,
        action="store_true",
        help="disable mixed precision training and run old school fp32",
    )
    parser.add_argument(
        "--val_folder",
        required=False,
        default="validation_raw",
        help="name of the validation folder. No need to use this for most people",
    )
    parser.add_argument(
        "--disable_saving",
        required=False,
        action="store_true",
        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
        "will be removed at the end of the training). Useful for development when you are "
        "only interested in the results and want to save some disk space",
    )
    parser.add_argument(
        "-pretrained_weights",
        type=str,
        required=False,
        default=None,
        help="path to nnU-Net checkpoint file to be used as pretrained model (use .model "
        "file, for example model_final_checkpoint.model). Will only be used when actually training. "
        "Optional. Beta. Use with caution.",
    )
    parser.add_argument(
        "--detcon",
        required=False,
        default=False,
        help="Use of Detective Contrastive style learning. Options ['intra', 'inter']\
                            * Intra - Increase local negatives - different classes pulled apart in same image\
                            * Inter - Increase global positives - same classes pushed together together between images",
    )
    parser.add_argument(
        "-k",
        "--kwargs",
        nargs="*",
        action=ParseKwargs,
        help="Any further arguments to pass to Trainer, such as method-specific hyperparams, or metabatch for gradient cache.",
    )

    args = parser.parse_args()

    task = args.task
    network = args.network
    network_trainer = args.network_trainer
    plans_identifier = args.p
    find_lr = args.find_lr
    detcon = args.detcon
    kwargs = args.kwargs

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic

    fp32 = args.fp32
    run_mixed_precision = not fp32

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    (
        plans_file,
        output_folder_name,
        dataset_directory,
        trainer_class,
    ) = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError(
            "Could not find trainer class in ssunet.training.network_training"
        )

    assert issubclass(
        trainer_class, (NetworkPreTrainer, GradCachePreTrainer)
    ), "network_trainer was found but is not derived from NetworkPreTrainer"

    if kwargs is not None:
        print("Using additional kwargs: ")
        for key in kwargs.keys():
            print(key + ": ", kwargs[key])
        trainer = trainer_class(
            plans_file,
            output_folder=output_folder_name,
            dataset_directory=dataset_directory,
            unpack_data=decompress_data,
            deterministic=deterministic,
            fp16=run_mixed_precision,
            **kwargs
        )
    else:
        trainer = trainer_class(
            plans_file,
            output_folder=output_folder_name,
            dataset_directory=dataset_directory,
            unpack_data=decompress_data,
            deterministic=deterministic,
            fp16=run_mixed_precision,
        )

    if not trainer.was_initialized:
        trainer.initialize(True)

    if args.disable_saving:
        trainer.save_final_checkpoint = (
            False  # whether or not to save the final checkpoint
        )
        trainer.save_best_checkpoint = (
            False  # whether or not to save the best checkpoint according to
        )
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = (
            True  # whether or not to save checkpoint_latest. We need that in case
        )
        # the training chashes
        trainer.save_latest_only = (
            True  # if false it will not store/overwrite _latest but separate files each
        )

    if find_lr:
        trainer.find_lr()
    else:
        if args.continue_training:
            # -c was set, continue a previous training and ignore pretrained weights
            trainer.load_latest_checkpoint()
        elif (not args.continue_training) and (args.pretrained_weights is not None):
            # we start a new training. If pretrained_weights are set, use them
            load_pretrained_weights(trainer.network, args.pretrained_weights)
        else:
            # new training without pretraine weights, do nothing
            pass

        print()
        trainer.run_training()


if __name__ == "__main__":
    main()
