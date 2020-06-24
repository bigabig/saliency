import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Discriminator import Discriminator
from Generator import Generator
from LossFunctions import BCELossWithDownsampling
from train_helper import load_dataset, freeze_generator_weights, main_loop

# constants
MODEL_PATH = './model/'
PRETRAINED_DISCRIMINATOR = None  # './model/Discriminator_pretraining_final.pt'
PRETRAINED_GENERATOR = './model/Generator_pretraining_final.pt'
EXPERIMENT_NAME = 'adversarial_training_generator_alpha005'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# select correct device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# parameters
alpha = 0.05  # <- is used for loss weighting. We achieved the best performance for 0.005.
learning_rate = 0.0003  # We used AdaGrad for optimization, with an initial learning rate of 3 * 10**-4.
weight_decay = 0.0001  # Weight decay
batch_size = 2  # We train the networks [...] using a batch size of 32.
start_epoch = 0  # <- specify if you want to start from a checkpoint
epochs = 300
save_interval = 300  # save every save_interval epochs

# init tensorboard
writer = SummaryWriter('runs/'+EXPERIMENT_NAME)

# load data
train_dl, valid_dl = load_dataset('./dataset_debug', batch_size)

# create the models & use pretrained weights
generator = Generator(torch.load('./vgg16/vgg16-conv.pth')).to(device)
discriminator = Discriminator().to(device)

# freeze some weights of the generator
freeze_generator_weights(generator)

# init the optimizers
# We used AdaGrad for optimization, with an initial learning rate of 3 * 10**-4.
# We used L2 weight regularization (weight decay) when training both the generator and discriminator (0.0001).
# pass only the not frozen parameters
opt_generator = optim.Adagrad(filter(lambda p: p.requires_grad, generator.parameters()),
                              lr=learning_rate, weight_decay=weight_decay)
opt_discriminator = optim.Adagrad(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                  lr=learning_rate, weight_decay=weight_decay)

# BCE, which is computed with respect to the down-sampled output and ground truth saliency
bce_loss_downsampling = BCELossWithDownsampling()
bce_loss = torch.nn.BCEWithLogitsLoss()

# load checkpoint
if start_epoch > 0:
    print('Loading adversarial model checkpoint...')
    checkpoint = torch.load(MODEL_PATH + "Generator_" + str(start_epoch) + ".pt")
    generator.load_state_dict(checkpoint['model_state_dict'])
    opt_generator.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    checkpoint = torch.load(MODEL_PATH + "Discriminator_" + str(start_epoch) + ".pt")
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    opt_discriminator.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    if PRETRAINED_GENERATOR:
        print('Loading pretrained generator...')
        checkpoint = torch.load(PRETRAINED_GENERATOR)
        generator.load_state_dict(checkpoint['model_state_dict'])
        opt_generator.load_state_dict(checkpoint['optimizer_state_dict'])
    if PRETRAINED_DISCRIMINATOR:
        print('Loading pretained discriminator...')
        checkpoint = torch.load(PRETRAINED_DISCRIMINATOR)
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        opt_discriminator.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Starting training with epoch {start_epoch}!")


def save_checkpoint(current_epoch):
    if current_epoch % save_interval == 0:
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': opt_generator.state_dict(),
        }, MODEL_PATH + "Generator_" + str(current_epoch) + ".pt")
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': opt_discriminator.state_dict(),
        }, MODEL_PATH + "Discriminator_" + str(current_epoch) + ".pt")
        print('Saving model checkpoint...')


def training_loop():
    loss_sum_discriminator = []
    loss_sum_generator = []
    loss_sum_adversarial = []

    # During the adversarial training, we alternate the training of the saliency prediction network and
    # discriminator network after each iteration (batch).
    #updates = 1

    # Training: Iterate over batches of data
    generator.train()
    discriminator.train()
    for batch in train_dl:
        batch_img, batch_fixation = batch['image'].to(device), batch['fixation'].to(device)

        # 1 is the target category of real samples and 0 for the category of fake (predicted) sample.
        labels_real = torch.ones(len(batch_img), dtype=torch.float, requires_grad=False).to(device)
        labels_fake = torch.zeros(len(batch_img), dtype=torch.float, requires_grad=False).to(device)

        # train the DISCRIMINATOR
        #if updates % 2 == 1:
        opt_discriminator.zero_grad()

        # the input image that a saliency map corresponds to is essential, due the fact the goal is not only to
        # have the two saliency maps becoming indistinguishable but with the condition that they both correspond
        # the same input image. We therefore include both the image and saliency map as inputs to the
        # discriminator network.
        # first, calculate the loss on the real (ground truth) image
        discriminator_input = torch.cat((batch_img, batch_fixation), 1)
        discriminator_pred = discriminator.forward(discriminator_input).squeeze()
        discriminator_loss_real = bce_loss(discriminator_pred, labels_real)
        discriminator_loss_real.backward()

        # second, calculate the loss on the fake (predicted) image
        generator_pred = generator.forward(batch_img)
        discriminator_input = torch.cat((batch_img, generator_pred), 1)
        discriminator_pred = discriminator.forward(discriminator_input).squeeze()
        discriminator_loss_fake = bce_loss(discriminator_pred, labels_fake)
        discriminator_loss_fake.backward()

        # third, combine those two losses as in the paper
        discriminator_loss = discriminator_loss_real + discriminator_loss_fake
        loss_sum_discriminator.append(discriminator_loss)
        # discriminator_loss.backward()

        opt_discriminator.step()

        # train the GENERATOR
        #else:
        opt_generator.zero_grad()

        # first, calculate the bce loss of the generator
        generator_pred = generator.forward(batch_img)
        generator_bce_loss = bce_loss_downsampling(generator_pred, batch_fixation)
        loss_sum_generator.append(generator_bce_loss)

        # second, calculate the bce loss of the discriminator on the fake (predicted) image
        discriminator_input = torch.cat((batch_img, generator_pred), 1)
        discriminator_pred = discriminator.forward(discriminator_input).squeeze()
        discriminator_bce_loss = bce_loss(discriminator_pred, labels_real)

        # third, combine those two loss as in the paper
        generator_loss = alpha * generator_bce_loss + discriminator_bce_loss  # TODO: wird mit dem discriminator loss überhaupt der generator geupdated?
        loss_sum_adversarial.append(generator_loss)
        generator_loss.backward()

        opt_generator.step()

        # updates += 1

    return loss_sum_generator, loss_sum_discriminator, loss_sum_adversarial


def validation_loop():
    generator.eval()
    with torch.no_grad():
        return [bce_loss_downsampling(generator.forward(batch['image'].to(device)), batch['fixation'].to(device)) for batch in valid_dl]


main_loop(start_epoch, epochs, save_checkpoint, training_loop, validation_loop, writer)

# save the final model
torch.save({'model_state_dict': generator.state_dict()}, MODEL_PATH + "Generator_final.pt")
torch.save({'model_state_dict': discriminator.state_dict()}, MODEL_PATH + "Discriminator_final.pt")

# finish tensorboard
writer.close()
