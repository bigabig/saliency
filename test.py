import os
import imageio
import torch
from Discriminator import Discriminator
from Generator import Generator
import torch.nn.functional as F
from train_helper import load_test_dataset


# constants
MODEL_PATH = './model/'
OUT_PATH = './out/'

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model directory {MODEL_PATH} does not exist!")
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# select correct device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# parameters
load_epoch = 295  # <- specify if you want to start from a checkpoint
load_final = True  # <- if this is set 'load_epoch' is ignored and instead the final model is loaded!
batch_size = 32

# load data
test_dl = load_test_dataset('./dataset_debug', batch_size)

# create the models & use pretrained weights
generator = Generator(torch.load('./vgg16/vgg16-conv.pth')).to(device)
discriminator = Discriminator().to(device)

# load checkpoint
if not load_final and load_epoch > 0:
    print('Loading model checkpoint...')
    checkpoint = torch.load(MODEL_PATH + "Generator_" + str(load_epoch) + ".pt")
    generator.load_state_dict(checkpoint['model_state_dict'])
elif load_final:
    print('Loading final model...')
    checkpoint = torch.load(MODEL_PATH + "Generator_final.pt")
    generator.load_state_dict(checkpoint['model_state_dict'])

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
