"""Split the Amazon Rainforedst dataset into train/dev/test and resize images to 64x64.

Original images have size (256, 256).
Resizing to (64, 64) reduces the dataset size from 634 MB to 57 MB, and loading smaller images
makes training faster.

"""

import argparse
import pandas as pd
import random
import os


from PIL import Image
from tqdm import tqdm


SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/Amazon_Rainforest_Dataset',
                    help="Directory with the Amazon Rainforest dataset")
parser.add_argument('--output_dir', default='data/64x64_Amazon_Rainforest_Dataset',
                    help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    data_dir = os.path.join(args.data_dir, 'train-jpg')

    # Get the filenames and label for each image
    filenames_labels = os.path.join(args.data_dir, 'train_v2.csv')
    filenames_labels = pd.read_csv(filenames_labels)

    # Split the images in 'Amazon_Rainforest_Dataset' into 60% train and 20% dev and 20% test.
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    filenames_labels = filenames_labels.sample(frac=1, random_state=230).reset_index(drop=True)
    split = int(0.9 * len(filenames_labels))
    split_2 = int(0.95 * len(filenames_labels))

    # Split the images and the labels
    train_filenames = filenames_labels[:split]
    dev_filenames = filenames_labels[split:split_2]
    test_filenames = filenames_labels[split_2:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_Amazon_Rainforest'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        filenames[split].to_csv(output_dir_split + ".csv", index=False)
        for filename, _ in tqdm(filenames[split].values):
            filename = os.path.join(data_dir, filename)
            resize_and_save(filename + ".jpg", output_dir_split, size=SIZE)

    print("Done building dataset")
