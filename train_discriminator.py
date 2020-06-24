import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Discriminator import Discriminator
from Generator import Generator
from train_helper import load_dataset, freeze_generator_weights, main_loop

# constants
MODEL_PATH = './model/'
EXPERIMENT_NAME = 'discriminator_pretraining'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# select correct device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# parameters
learning_rate = 0.0003  # We used AdaGrad for optimization, with an initial learning rate of 3 * 10**-4.
weight_decay = 0.0001  # Weight decay
batch_size = 2  # We train the networks [...] using a batch size of 32.
start_epoch = 0  # <- specify if you want to start from a checkpoint
epochs = 100
save_interval = 15  # save every save_interval epochs

# init tensorboard
writer = SummaryWriter('runs/'+EXPERIMENT_NAME)

# load data
train_dl, valid_dl = load_dataset('./dataset_debug', batch_size)

# create the models & use pretrained weights
generator = Generator(torch.load('./vgg16/vgg16-conv.pth')).to(device)
discriminator = Discriminator().to(device)

# freeze some weights of the generator
freeze_generator_weights(generator)

# init the optimizer
opt_discriminator = optim.Adagrad(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                  lr=learning_rate, weight_decay=weight_decay)

# BCE, which is computed with respect to the down-sampled output and ground truth saliency
bce_loss = torch.nn.BCEWithLogitsLoss()

# load generator
print('Loading pretrained generator...')
checkpoint = torch.load(MODEL_PATH + "Generator_pretraining_final.pt")
generator.load_state_dict(checkpoint['model_state_dict'])

# load checkpoint
if start_epoch > 0:
    print('Loading discriminator checkpoint...')
    checkpoint = torch.load(MODEL_PATH + "Discriminator_pretraining" + str(start_epoch) + ".pt")
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    opt_discriminator.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Starting training with epoch {start_epoch}!")


def save_checkpoint(current_epoch):
    if current_epoch % save_interval == 0:
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': opt_discriminator.state_dict(),
        }, MODEL_PATH + "Discriminator_pretraining" + str(current_epoch) + ".pt")
        print('Saving discriminator checkpoint...')


def training_loop():
    loss_sum_discriminator = []

    # Training: Iterate over batches of data
    generator.eval()
    discriminator.train()
    for batch in train_dl:
        batch_img, batch_fixation = batch['image'].to(device), batch['fixation'].to(device)

        # 1 is the target category of real samples and 0 for the category of fake (predicted) sample.
        labels_real = torch.ones(len(batch_img), dtype=torch.float, requires_grad=False).to(device)
        labels_fake = torch.zeros(len(batch_img), dtype=torch.float, requires_grad=False).to(device)

        # train the DISCRIMINATOR

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
        opt_discriminator.zero_grad()

    return [], loss_sum_discriminator, []


def validation_loop():
    return []


main_loop(start_epoch, epochs, save_checkpoint, training_loop, validation_loop, writer)

# save the final model
torch.save({
    'model_state_dict': discriminator.state_dict(),
    'optimizer_state_dict': opt_discriminator.state_dict(),
}, MODEL_PATH + "Discriminator_pretraining_final.pt")

# finish tensorboard
writer.close()
