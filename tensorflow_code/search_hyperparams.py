"""Peform hyperparemeters search."""

import argparse
import os
from subprocess import check_call
import sys

from model.utils import Params


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/64x64_Amazon_Rainforest_Dataset',
                    help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir}".format(python=PYTHON,
                                                                                   model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform hypersearch over one parameter
    # learning_rates = [1e-4, 1e-3, 1e-2]
    # weight_decays = [1e-4, 1e-3, 1e-2]
    # loss_weights = [32, 64]
    # num_channels_list = [32, 64, 128]
    versions = [1, 2]
    # for learning_rate in learning_rates:
    # for weight_decay in weight_decays:
    # for loss_weight in loss_weights:
    # for num_channels in num_channels_list:
    for version in versions:
        # Modify the relevant parameter in params
        # params.learning_rate = learning_rate
        # params.weight_decay = weight_decay
        # params.loss_weight = loss_weight
        params.version = version

        # Launch job (name has to be unique)
        # job_name = "learning_rate_{}".format(learning_rate)
        # job_name = "weight_decay_{}".format(weight_decay)
        # job_name = "loss_weight_{}".format(loss_weight)
        # job_name = "num_channels_{}".format(num_channels)
        job_name = "version_{}".format(version)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)
