import glob
import os
import cv2
import imageio
from tqdm import tqdm

IMAGE_SIZE = (224, 224)

path_train_images = './salicon/images/train/'
path_train_fixations = './salicon/fixations/train/'
path_val_images = './salicon/images/val/'
path_val_fixations = './salicon/fixations/val/'

path_dataset = './dataset_salicon/'
if not os.path.exists(path_dataset):
    os.makedirs(path_dataset)

path_resized_images_train = 'images/train/'
if not os.path.exists(os.path.join(path_dataset, path_resized_images_train)):
    os.makedirs(os.path.join(path_dataset, path_resized_images_train))

path_resized_fixations_train = 'fixations/train/'
if not os.path.exists(os.path.join(path_dataset, path_resized_fixations_train)):
    os.makedirs(os.path.join(path_dataset, path_resized_fixations_train))

path_resized_images_val = 'images/val/'
if not os.path.exists(os.path.join(path_dataset, path_resized_images_val)):
    os.makedirs(os.path.join(path_dataset, path_resized_images_val))

path_resized_fixations_val = 'fixations/val/'
if not os.path.exists(os.path.join(path_dataset, path_resized_fixations_val)):
    os.makedirs(os.path.join(path_dataset, path_resized_fixations_val))

train_image_names = [image.split(os.path.sep)[-1].split('.')[0] for image in glob.glob(os.path.join(path_train_images, '*train*'))]
val_image_names = [image.split(os.path.sep)[-1].split('.')[0] for image in glob.glob(os.path.join(path_val_images, '*val*'))]


with open(os.path.join(path_dataset, 'train_images.txt'), 'a') as train_images_file, open(os.path.join(path_dataset, 'train_fixations.txt'), 'a') as train_fixations_file:
    first = True
    for img_name in tqdm(train_image_names):
        # resize
        image_resized = cv2.resize(imageio.imread(path_train_images + img_name + '.jpg'), IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        fixation_resized = cv2.resize(imageio.imread(path_train_fixations + img_name + '.png'), IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        if len(image_resized.shape) != 3 and len(fixation_resized) != 2:
            continue

        # write image
        imageio.imwrite(os.path.join(path_dataset, path_resized_images_train, img_name + '.png'), image_resized)
        imageio.imwrite(os.path.join(path_dataset, path_resized_fixations_train, img_name + '.png'), fixation_resized)

        # write path to file
        if first:
            train_images_file.write(path_resized_images_train + img_name + '.png')
            train_fixations_file.write(path_resized_fixations_train + img_name + '.png')
            first = False
        else:
            train_images_file.write('\n' + path_resized_images_train + img_name + '.png')
            train_fixations_file.write('\n' + path_resized_fixations_train + img_name + '.png')

with open(os.path.join(path_dataset, 'val_images.txt'), 'a') as val_images_file, open(os.path.join(path_dataset, 'val_fixations.txt'), 'a') as val_fixations_file:
    first = True
    for img_name in tqdm(val_image_names):
        # resize
        image_resized = cv2.resize(imageio.imread(path_val_images + img_name + '.jpg'), IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        fixation_resized = cv2.resize(imageio.imread(path_val_fixations + img_name + '.png'), IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        if len(image_resized.shape) != 3 and len(fixation_resized) != 2:
            continue

        # write image
        imageio.imwrite(os.path.join(path_dataset, path_resized_images_val, img_name + '.png'), image_resized)
        imageio.imwrite(os.path.join(path_dataset, path_resized_fixations_val, img_name + '.png'), fixation_resized)

        # write path to file
        if first:
            val_images_file.write(os.path.join(path_resized_images_val + img_name + '.png'))
            val_fixations_file.write(path_resized_fixations_val + img_name + '.png')
            first = False
        else:
            val_images_file.write('\n' + os.path.join(path_resized_images_val + img_name + '.png'))
            val_fixations_file.write('\n' + path_resized_fixations_val + img_name + '.png')




