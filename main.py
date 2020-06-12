import torchvision
import torch
import torch.optim as optim

from FixationDataset import FixationDataset
from Generator import Generator
from LossFunctions import WeightedMSELoss
from Transformations import Rescale, ToTensor, Normalize
from torch.utils.data import DataLoader

# select correct device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# parameters
alpha = 1.5  # <- is used for loss weighting
learning_rate = 0.01
bs = 16  # <- batch size
MODEL_PATH = './model/final_model.pt'
CHECKPOINT_BASE_PATH = './model/'
load_epoch = 0  # <- specify if you want to start from a checkpoint

# load data
composed = torchvision.transforms.Compose([Rescale(), ToTensor(), Normalize()])
train_ds = FixationDataset('./dataset', './dataset/train_images.txt', './dataset/train_fixations.txt', composed)
valid_ds = FixationDataset('./dataset', './dataset/val_images.txt', './dataset/val_fixations.txt', composed)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=bs*2)

# init cnn training
net = Generator(torch.load('./vgg16/vgg16-conv.pth')).to(device)
loss_fcn = WeightedMSELoss(alpha)
opt = optim.Adam(net.parameters(), lr=learning_rate)

start_epoch = 0
epochs = 3
output_steps = 1
current_step = 1

# load checkpoint
if load_epoch > 0:
    print('Loading model checkpoint...')
    checkpoint = torch.load(CHECKPOINT_BASE_PATH + "My_Model_" + str(load_epoch) + ".pt")
    net.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    valid_loss = checkpoint['loss']

# Iterate over epochs
for epoch in range(start_epoch, epochs):

    # save checkpoint:
    if epoch > 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': valid_loss
        }, CHECKPOINT_BASE_PATH + "My_Model_" + str(epoch) + ".pt")
        print('Saving model checkpoint...')

    # Training: Iterate over batches of data
    net.train()
    for batch in train_dl:
        xb, yb = batch['image'].to(device), batch['fixation'].to(device)
        pred = net.forward(xb)  # net(xb) also works
        loss = loss_fcn(pred, yb)

        if current_step % output_steps == 0:
            print(f'Loss: {loss}')

        loss.backward()
        opt.step()
        opt.zero_grad()

        current_step += 1

    # Validation:
    net.eval()
    with torch.no_grad():
        valid_loss = sum(loss_fcn(net.forward(batch['image'].to(device)), batch['fixation'].to(device)) for batch in valid_dl)
    print(epoch, valid_loss / len(valid_dl))

# save the final model
torch.save(net.state_dict(), MODEL_PATH)

