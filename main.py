import torchvision
import torch
import torch.optim as optim

from Discriminator import Discriminator
from FixationDataset import FixationDataset
from Generator import Generator
from LossFunctions import WeightedMSELoss, BCELossWithDownsampling
from Transformations import Rescale, ToTensor, Normalize
from torch.utils.data import DataLoader

# select correct device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# parameters
alpha = 1.5  # <- is used for loss weighting
learning_rate = 0.01
bs = 32  # We train the networks [...] using a batch size of 32.
MODEL_PATH = './model/final_model.pt'
CHECKPOINT_BASE_PATH = './model/'
load_epoch = 0  # <- specify if you want to start from a checkpoint

# load data
composed = torchvision.transforms.Compose([Rescale(), ToTensor(), Normalize()])
train_ds = FixationDataset('./dataset', './dataset/train_images.txt', './dataset/train_fixations.txt', composed)
valid_ds = FixationDataset('./dataset', './dataset/val_images.txt', './dataset/val_fixations.txt', composed)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=bs*2)

# create the models
generator = Generator(torch.load('./vgg16/vgg16-conv.pth')).to(device)
discriminator = Discriminator().to(device)

# init the models
# TODO: random weights?!

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
opt_generator = optim.Adagrad(filter(lambda p: p.requires_grad, generator.parameters()), lr=0.0003)  # pass only the not frozen parameters
opt_discriminator = optim.Adagrad(filter(lambda p: p.requires_grad, generator.parameters()), lr=0.0003)  # pass only the not frozen parameters

content_loss_fcn = BCELossWithDownsampling()

loss_fcn = WeightedMSELoss(alpha)
# opt = optim.Adam(net.parameters(), lr=learning_rate)

start_epoch = 0
epochs = 3
output_steps = 1
current_step = 1

# load checkpoint
if load_epoch > 0:
    print('Loading model checkpoint...')
    checkpoint = torch.load(CHECKPOINT_BASE_PATH + "My_Model_" + str(load_epoch) + ".pt")
    generator.load_state_dict(checkpoint['model_state_dict'])
    opt_generator.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    valid_loss = checkpoint['loss']

# Iterate over epochs
for epoch in range(start_epoch, epochs):

    # save checkpoint:
    if epoch > 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': opt_generator.state_dict(),
            'loss': valid_loss
        }, CHECKPOINT_BASE_PATH + "My_Model_" + str(epoch) + ".pt")
        print('Saving model checkpoint...')

    # Training: Iterate over batches of data
    generator.train()
    for batch in train_dl:
        xb, yb = batch['image'].to(device), batch['fixation'].to(device)
        print(yb)
        pred = generator.forward(xb)  # net(xb) also works
        loss = loss_fcn(pred, yb)

        if current_step % output_steps == 0:
            print(f'Loss: {loss}')

        loss.backward()
        opt_generator.step()
        opt_generator.zero_grad()

        current_step += 1

    # Validation:
    generator.eval()
    with torch.no_grad():
        valid_loss = sum(loss_fcn(generator.forward(batch['image'].to(device)), batch['fixation'].to(device)) for batch in valid_dl)
    print(epoch, valid_loss / len(valid_dl))

# save the final model
torch.save(generator.state_dict(), MODEL_PATH)

