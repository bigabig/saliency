import time
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from FixationDataset import FixationDataset
from Transformations import Rescale, ToTensor, Normalize
from torch.utils.data import DataLoader


def load_dataset(dataset, batch_size):
    composed = torchvision.transforms.Compose([Rescale(), ToTensor(), Normalize()])
    train_ds = FixationDataset(dataset, dataset + '/train_images.txt', dataset + '/train_fixations.txt', composed)
    valid_ds = FixationDataset(dataset, dataset + '/val_images.txt', dataset + '/val_fixations.txt', composed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)
    return train_dl, valid_dl


def load_test_dataset(dataset, batch_size):
    composed = torchvision.transforms.Compose([Rescale(), ToTensor(), Normalize()])
    test_ds = FixationDataset(dataset, dataset + '/test_images.txt', dataset + '/train_fixations.txt', composed, testing=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return test_dl


def freeze_generator_weights(generator):
    # Only the last two groups of convolutional layers in VGG-16 are modified during the training for saliency
    # prediction while the earlier layers remain fixed from the original VGG-16 model.
    layer_counter = 0
    for layer in generator.encoder.children():
        if layer_counter < 7:
            for param in layer.parameters():
                param.requires_grad = False  # freeze the parameters of the first 7 layers of the encoder
        layer_counter += 1


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# maybe helpful some time to plot images to tensorboard
# for batch in train_dl:
#     fixations = batch['fixation']
#     # create grid of images
#     img_grid = torchvision.utils.make_grid(fixations)
#
#     # show images
#     matplotlib_imshow(img_grid, one_channel=True)
#
#     # write to tensorboard
#     writer.add_image('four_fashion_mnist_images', img_grid)
#
#     # show model
#     writer.add_graph(generator, batch['image'].to(device))
#     writer.close()
#     break


def main_loop(start_epoch, epochs, save_checkpoint, training_loop, validation_loop, writer):

    # Iterate over epochs
    for epoch in range(start_epoch, epochs):

        epoch_time = time.time()

        # save checkpoint:
        if epoch > 0:
            save_checkpoint(epoch)

        # Training: Iterate over batches of data
        loss_sum_generator, loss_sum_discriminator, loss_sum_adversarial = training_loop()
        training_time = time.time()

        # Validation:
        validation_loss_sum = validation_loop()
        validation_time = time.time()

        print("----------------------------------------------")
        print(f"Epoch {epoch} summary:")
        print(f"Total time this epoch: \t\t {time.time() - epoch_time}")
        print(f"Time for training: \t\t\t {training_time - epoch_time}")
        print(f"Time for validation: \t\t {validation_time - training_time}")
        if len(loss_sum_discriminator) > 0:
            loss = sum(loss_sum_discriminator) / len(loss_sum_discriminator)
            print(f"Training Loss Discriminator: {loss}")
            writer.add_scalar('training loss discriminator', loss, epoch)
        if len(loss_sum_generator) > 0:
            loss = sum(loss_sum_generator) / len(loss_sum_generator)
            print(f"Training Loss Generator:\t {loss}")
            writer.add_scalar('training loss generator', loss, epoch)
        if len(loss_sum_adversarial) > 0:
            loss = sum(loss_sum_adversarial) / len(loss_sum_adversarial)
            print(f"Training Loss Adversarial:\t {loss}")
            writer.add_scalar('training loss adversarial', loss, epoch)
        if len(validation_loss_sum) > 0:
            loss = sum(validation_loss_sum) / len(validation_loss_sum)
            print(f"Validation Loss Generator:\t {loss}")
            writer.add_scalar('validation loss generator', loss, epoch)
