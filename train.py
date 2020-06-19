import torchvision
import torch
import time
import torch.optim as optim

from Discriminator import Discriminator
from FixationDataset import FixationDataset
from Generator import Generator
from LossFunctions import BCELossWithDownsampling
from Transformations import Rescale, ToTensor, Normalize
from torch.utils.data import DataLoader

# constants
MODEL_PATH = './model/'

# select correct device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# parameters
alpha = 0.05  # <- is used for loss weighting. We achieved the best performance for 0.005.
learning_rate = 0.0003  # We used AdaGrad for optimization, with an initial learning rate of 3 * 10**-4.
weight_decay = 0.0001  # Weight decay
batch_size = 32  # We train the networks [...] using a batch size of 32.
load_epoch = 25  # <- specify if you want to start from a checkpoint

# load data
dataset = './dataset_combined'  # or just './dataset'
composed = torchvision.transforms.Compose([Rescale(), ToTensor(), Normalize()])
train_ds = FixationDataset(dataset, dataset + '/train_images.txt', dataset + '/train_fixations.txt', composed)
valid_ds = FixationDataset(dataset, dataset + '/val_images.txt', dataset + '/val_fixations.txt', composed)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)

# create the models & use pretrained weights
generator = Generator(torch.load('./vgg16/vgg16-conv.pth')).to(device)
discriminator = Discriminator().to(device)

# freeze some weights of the generator
# " Only the last two groups of convolutional layers in VGG-16 are modified during the training for saliency prediction,
# while the earlier layers remain fixed from the original VGG-16 model."
layer_counter = 0
for layer in generator.encoder.children():
    if layer_counter < 7:
        for param in layer.parameters():
            param.requires_grad = False  # freeze the parameters of the first 7 layers of the encoder
    layer_counter += 1

# init the optimizers
# We used AdaGrad for optimization, with an initial learning rate of 3 * 10**-4.
# We used L2 weight regularization (weight decay) when training both the generator and discriminator (0.0001).
# pass only the not frozen parameters
opt_generator = optim.Adagrad(filter(lambda p: p.requires_grad, generator.parameters()),
                              lr=learning_rate, weight_decay=weight_decay)
opt_discriminator = optim.Adagrad(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                  lr=learning_rate, weight_decay=weight_decay)

# At train time, we first bootstrap the saliency prediction network function by training for 15 epochs
epochs = 250

# BCE, which is computed with respect to the down-sampled output and ground truth saliency
bce_loss_downsampling = BCELossWithDownsampling()
bce_loss = torch.nn.BCEWithLogitsLoss()

start_epoch = 0

# load checkpoint
if load_epoch > 0:
    print('Loading model checkpoint...')
    checkpoint = torch.load(MODEL_PATH + "Generator_" + str(load_epoch) + ".pt")
    generator.load_state_dict(checkpoint['model_state_dict'])
    opt_generator.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

    if start_epoch >= 15:
        checkpoint = torch.load(MODEL_PATH + "Discriminator_" + str(load_epoch) + ".pt")
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        opt_discriminator.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Starting training with epoch {start_epoch}!")

# Iterate over epochs
for epoch in range(start_epoch, epochs):

    epoch_time = time.time()
    updates = 1
    loss_sum_discriminator = []
    loss_sum_generator = []

    # save checkpoint:
    if epoch > 0 and epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': opt_generator.state_dict(),
        }, MODEL_PATH + "Generator_" + str(epoch) + ".pt")
        if epoch >= 15:
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': opt_discriminator.state_dict(),
            }, MODEL_PATH + "Discriminator_" + str(epoch) + ".pt")
        print('Saving model checkpoint...')

    # Training: Iterate over batches of data
    generator.train()
    discriminator.train()
    for batch in train_dl:
        batch_img, batch_fixation = batch['image'].to(device), batch['fixation'].to(device)

        # train only the generator in the first 15 epochs
        if epoch < 15:
            # train the generator
            generator_pred = generator.forward(batch_img)
            generator_loss = bce_loss_downsampling(generator_pred, batch_fixation)

            loss_sum_generator += generator_loss

            generator_loss.backward()
            opt_generator.step()
            opt_generator.zero_grad()

        # then perform adversial training
        else:
            # 1 is the target category of real samples and 0 for the category of fake (predicted) sample.
            labels_real = torch.ones(len(batch_img), dtype=torch.float).to(device)
            labels_fake = torch.zeros(len(batch_img), dtype=torch.float).to(device)

            # During the adversarial training, we alternate the training of the saliency prediction network and
            # discriminator network after each iteration (batch).
            if updates % 2 == 1:
                # train the discriminator

                # the input image that a saliency map corresponds to is essential, due the fact the goal is not only to
                # have the two saliency maps becoming indistinguishable but with the condition that they both correspond
                # the same input image. We therefore include both the image and saliency map as inputs to the
                # discriminator network.
                # first, calculate the loss on the real (ground truth) image
                discriminator_input = torch.cat((batch_img, batch_fixation), 1)
                discriminator_pred = discriminator.forward(discriminator_input).squeeze()
                discriminator_loss_real = bce_loss(discriminator_pred, labels_real)

                # second, calculate the loss on the fake (predicted) image
                generator_pred = generator.forward(batch_img)
                discriminator_input = torch.cat((batch_img, generator_pred), 1)
                discriminator_pred = discriminator.forward(discriminator_input).squeeze()
                discriminator_loss_fake = bce_loss(discriminator_pred, labels_fake)

                # third, combine those two losses as in the paper
                discriminator_loss = discriminator_loss_real + discriminator_loss_fake

                loss_sum_discriminator.append(discriminator_loss)

                discriminator_loss.backward()
                opt_discriminator.step()
                opt_discriminator.zero_grad()

            else:
                # train the generator

                # first, calculate the bce loss of the generator
                generator_pred = generator.forward(batch_img)
                generator_bce_loss = bce_loss_downsampling(generator_pred, batch_fixation)

                # second, calculate the bce loss of the discriminator on the fake (predicted) image
                discriminator_input = torch.cat((batch_img, generator_pred), 1)
                discriminator_pred = discriminator.forward(discriminator_input).squeeze()
                discriminator_bce_loss = bce_loss(discriminator_pred, labels_real)

                # third, combine those two loss as in the paper
                generator_loss = alpha * generator_bce_loss + discriminator_bce_loss

                loss_sum_generator.append(generator_loss)

                generator_loss.backward()
                opt_generator.step()
                opt_generator.zero_grad()

        updates += 1
    training_time = time.time()

    # Validation:
    generator.eval()
    with torch.no_grad():
        valid_loss = sum(bce_loss_downsampling(generator.forward(batch['image'].to(device)), batch['fixation'].to(device)) for batch in valid_dl)
    validation_time = time.time()

    print("----------------------------------------------")
    print(f"Epoch {epoch} summary:")
    print(f"Total time this epoch: \t\t {time.time() - epoch_time}")
    print(f"Time for training: \t\t\t {training_time - epoch_time}")
    print(f"Time for validation: \t\t {validation_time - training_time}")
    if epoch >= 15:
        print(f"Training Loss Discriminator: {sum(loss_sum_discriminator) / len(loss_sum_discriminator)}")
    print(f"Training Loss Generator:\t {sum(loss_sum_generator) / len(loss_sum_generator)}")
    print(f"Validation Loss Generator:\t {valid_loss / len(valid_dl)}")

# save the final model
torch.save(generator.state_dict(), MODEL_PATH + "Generator_final.pt")
torch.save(discriminator.state_dict(), MODEL_PATH + "Discriminator_final.pt")
