import imageio
import torchvision
import torch
from Discriminator import Discriminator
from FixationDataset import FixationDataset
from Generator import Generator
from Transformations import Rescale, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.nn.functional as F

# constants
MODEL_PATH = './model/'
OUT_PATH = './out/'

# select correct device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# parameters
load_epoch = 14  # <- specify if you want to start from a checkpoint
batch_size = 32

# load data
composed = torchvision.transforms.Compose([Rescale(), ToTensor(), Normalize()])
test_ds = FixationDataset('./dataset', './dataset/test_images.txt', './dataset/train_fixations.txt', composed, testing=True)
test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=True)

# create the models & use pretrained weights
generator = Generator(torch.load('./vgg16/vgg16-conv.pth')).to(device)
discriminator = Discriminator().to(device)

# load checkpoint
if load_epoch > 0:
    print('Loading model checkpoint...')
    checkpoint = torch.load(MODEL_PATH + "Generator_" + str(load_epoch) + ".pt")
    generator.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']

print(f"Starting testing with checkpoint from epoch {start_epoch}!")


generator.eval()
with torch.no_grad():

    for batch in test_dl:
        batch_img, batch_name = batch['image'].to(device), batch['name']

        generator_pred = generator.forward(batch_img)
        generator_img = 255 * F.sigmoid(generator_pred).cpu()
        generator_img = generator_img.type(torch.uint8)

        images, _, width, height = generator_img.shape
        for i in range(images):
            fixation_map = generator_img[i].view((width, height))
            imageio.imwrite(OUT_PATH + batch['name'][i], fixation_map)