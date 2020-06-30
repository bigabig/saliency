import os
import torch
from Generator import Generator
from LossFunctions import BCELossWithDownsampling, WeightedMSELossWithDownsampling, WeightedMSELoss
from train_helper import load_dataset, freeze_generator_weights, main_loop
from torch.utils.tensorboard import SummaryWriter

# constants
MODEL_PATH = './model/'
EXPERIMENT_NAME = 'generator_pretraining_renamed'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# select correct device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# parameters
learning_rate = 0.0003  # 0.001
batch_size = 8  # We train the networks [...] using a batch size of 32.
start_epoch = 0  # <- specify if you want to start from a checkpoint
epochs = 100
weight_decay = 0.0001  # Weight decay
save_interval = 1  # save every save_interval epochs

# init tensorboard
writer = SummaryWriter('runs/'+EXPERIMENT_NAME)

# load data
train_dl, valid_dl = load_dataset('./dataset', batch_size)

# create the models & use pretrained weights
generator = Generator(torch.load('./vgg16/vgg16-conv.pth')).to(device)

# freeze some weights of the generator
freeze_generator_weights(generator)

# init the optimizers
opt_generator = torch.optim.Adagrad(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)

# BCE, which is computed with respect to the down-sampled output and ground truth saliency
loss_function = BCELossWithDownsampling()
# loss_function = WeightedMSELoss(alpha=1.01)

# load checkpoint
if start_epoch > 0:
    print('Loading model checkpoint...')
    checkpoint = torch.load(MODEL_PATH + "Generator_pretraining" + str(start_epoch) + ".pt")
    generator.load_state_dict(checkpoint['model_state_dict'])
    opt_generator.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']

print(f"Starting training with epoch {start_epoch}!")


def save_checkpoint(current_epoch):
    if current_epoch % save_interval == 0:
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': opt_generator.state_dict(),
        }, MODEL_PATH + "Generator_pretraining" + str(current_epoch) + ".pt")
        print('Saving generator checkpoint...')


def training_loop():
    loss_sum_generator = []
    generator.train()
    for batch in train_dl:
        batch_img, batch_fixation = batch['image'].to(device), batch['fixation'].to(device)

        # train the generator
        generator_pred = generator(batch_img)
        generator_loss = loss_function(generator_pred, batch_fixation)

        loss_sum_generator.append(generator_loss)

        generator_loss.backward()
        opt_generator.step()
        opt_generator.zero_grad()
    return loss_sum_generator, [], []


def validation_loop():
    generator.eval()
    with torch.no_grad():
        return [loss_function(generator.forward(batch['image'].to(device)), batch['fixation'].to(device)) for batch in valid_dl]


main_loop(start_epoch, epochs, save_checkpoint, training_loop, validation_loop, writer)

# save the final model
torch.save({
    'model_state_dict': generator.state_dict(),
    'optimizer_state_dict': opt_generator.state_dict(),
}, MODEL_PATH + "Generator_pretraining_final.pt")

# finish tensorboard
writer.close()
