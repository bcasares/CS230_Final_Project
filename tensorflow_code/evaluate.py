"""Evaluate the model"""

import argparse
import logging
import os
import pandas as pd

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/64x64_Amazon_Rainforest_Dataset',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "test_Amazon_Rainforest")

    # Get the filenames and labels from the train and dev sets
    test_filenames_labels = pd.read_csv(os.path.join(data_dir, 'test_Amazon_Rainforest.csv'))
    test_filenames = [os.path.join(test_data_dir, f) + ".jpg" for f in test_filenames_labels.image_name.tolist()]

    # Build list with unique labels
    label_list = []
    for tag_str in test_filenames_labels.tags.values:
        labels = tag_str.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)
    assert(len(label_list) == 17) # Assert that test has the same labels as the train/dev

    # # Add onehot features for every label
    for label in label_list:
        test_filenames_labels[label] = test_filenames_labels['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

    # Convert the labels to a matrix
    # Labels will be  0 or 1 for each category
    test_labels = test_filenames_labels.drop(["image_name", 'tags'], axis=1).as_matrix()

    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, test_labels, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
