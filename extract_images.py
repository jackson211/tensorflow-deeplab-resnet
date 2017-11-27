
# coding: utf-8

# In[1]:

""" extract_images.py

    This script performs a sliding window on the input images and extracts the
    smaller patches.
    It can also perform basic data augmentation (rotations and flips).

    The dataset is expected to be one or several collections of images (at least
    the input and the ground truth). Images with the same id should have the same
    dimensions, e.g. ground_truth_4.tif and top_rgb_4.tif should represent the
    same tile (4).

"""

# imports
import numpy as np
from skimage import io
import skimage.transform
import os
from tqdm import tqdm

import warnings
# Filter the warnings for low contrast images
warnings.filterwarnings('ignore')


# In[2]:
# import config values
from config import patch_size, step_size, ROTATIONS, FLIPS, DATA_DIRECTORY,\
	  folders, train_ids, test_ids


# In[3]:

def sliding_window(image, stride=10, window_size=(20,20)):
    """Extract patches according to a sliding window.

    Args:
        image (numpy array): The image to be processed.
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).

    Returns:
        list: list of patches with window_size dimensions
    """
    patches = []
    # slide a window across the image
    for x in range(0, image.shape[0], stride):
        for y in range(0, image.shape[1], stride):
            new_patch = image[x:x + window_size[0], y:y + window_size[1]]
            if new_patch.shape[:2] == window_size:
                patches.append(new_patch)
    return patches


def transform(patch, flip=False, mirror=False, rotations=[]):
    """Perform data augmentation on a patch.

    Args:
        patch (numpy array): The patch to be processed.
        flip (bool, optional): Up/down symetry.
        mirror (bool, optional): left/right symetry.
        rotations (int list, optional) : rotations to perform (angles in deg).

    Returns:
        array list: list of augmented patches
    """
    transformed_patches = [patch]
    for angle in rotations:
        transformed_patches.append(skimage.img_as_ubyte(skimage.transform.rotate(patch, angle)))
    if flip:
        transformed_patches.append(np.flipud(patch))
    if mirror:
        transformed_patches.append(np.fliplr(patch))
    return transformed_patches


# We write the relevant parameters in a text file
details_file = open(DATA_DIRECTORY + 'details.txt', 'w')
details_file.write('Training tiles : {}\n'.format(train_ids))
details_file.write('Testing tiles : {}\n'.format(test_ids))
details_file.write('Sliding window patch size : ({},{})'.format(*patch_size))
details_file.write('Sliding window stride : {}'.format(step_size))
details_file.close()


for suffix, folder, files in tqdm(folders):
    tqdm.write(("=== PROCESSING {} ===").format(suffix.upper()))

    # We create the subfolders splitted in train and test
    os.mkdir(DATA_DIRECTORY + suffix + '_train')
    os.mkdir(DATA_DIRECTORY + suffix + '_test')

    # Generate generators to read the iamges
    train_dataset = (io.imread(folder + files.format(*id_)) for id_ in train_ids)
    test_dataset = (io.imread(folder + files.format(*id_)) for id_ in test_ids)

    train_samples = []
    test_samples = []
    for image in tqdm(train_dataset):
        # Use the sliding window to extract the patches
        for patches in sliding_window(image, window_size=patch_size, stride=step_size):
            # Append the augmented patches to the sequence
            train_samples.extend(transform(patches, flip=FLIPS[0], mirror=FLIPS[1], rotations=ROTATIONS))

    for image in tqdm(test_dataset):
        # Same as the previous loop, but without data augmentation (test dataset)
        # Sliding window with no overlap
        for patches in sliding_window(image, window_size=patch_size, stride=patch_size[0]):
            test_samples.extend(transform(patches))

    # We save the images on disk
    for i, sample in tqdm(enumerate(train_samples), total=len(train_samples), desc="Saving train samples"):
        io.imsave('{}/{}.png'.format(DATA_DIRECTORY + suffix + '_train', i), sample)

    tqdm.write("({} training set: done)".format(suffix))

    for i, sample in tqdm(enumerate(test_samples), total=len(test_samples), desc="Saving test samples"):
        io.imsave('{}/{}.png'.format(DATA_DIRECTORY + suffix + '_test', i), sample)
    tqdm.write("({} testing set: done)".format(suffix))


print("All done ! The dataset has been saved in {}.".format(DATA_DIRECTORY))

train_file = open(DATA_DIRECTORY + 'train_file.txt', 'w')
for i in range(57549):
    train_file.write('/irrg_train/{}.png /labels_train/{}.png\n'.format(i, i))
train_file.close()

test_file = open(DATA_DIRECTORY + 'test_file.txt', 'w')
for i in range(700):
    test_file.write('/irrg_test/{}.png /labels_test/{}.png\n'.format(i, i))
test_file.close()
