#Valery Lozko
#CPSC8430 HW4

from torch_fidelity import calculate_metrics
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.utils import save_image
import DCGAN
import WGAN
import ACGAN
import os

os.makedirs('./images', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)

subset_indices = list(range(1000))
subset_dataset = Subset(dataset, subset_indices)
dataloader = DataLoader(subset_dataset, batch_size=128, shuffle=False)

os.makedirs('./images/real_images', exist_ok=True)
for i, (images, _) in enumerate(dataloader):
    for j, image in enumerate(images):
        save_image((image + 1) / 2, f'./images/real_images/{i * 128 + j}.png')

def generate_images(images, save_path):
    os.makedirs(save_path, exist_ok=True)
    for key in images.keys():
        for idx, image in enumerate(images[key]):
            save_image(image, os.path.join(save_path, f'{key}_{idx}.png'))


def evaluate_performance(fake_images_path, real_images_path):
    metrics = calculate_metrics(
        input1=fake_images_path,
        input2=real_images_path,
        fid = True,
        isc=False,
        kid=False,
        prc=False,
        batch_size=128
    )
    return metrics['frechet_inception_distance']

def evaluate_image(image):
    return image.mean().item()

def find_best(images, save_path):
    best_images = {}
    for label in images.keys():
        for image in images[label]:
            score = evaluate_image(image)
            if label not in best_images or score > evaluate_image(best_images[label]):
                best_images[label] = image

    images = [image for _, image in best_images.items()]
    grid = torch.stack(images, dim=0)
    save_image(grid, save_path, nrow = 5)

DCGAN_images = DCGAN.test()
WGAN_images = WGAN.test()
ACGAN_images = ACGAN.test()

generate_images(DCGAN_images, './images/DCGAN/')
generate_images(WGAN_images, './images/WGAN/')
generate_images(ACGAN_images, './images/ACGAN/')

DCGAN_perf = evaluate_performance('./images/DCGAN/', './images/real_images/')
WGAN_perf = evaluate_performance('./images/WGAN/', './images/real_images/')
ACGAN_perf = evaluate_performance('./images/ACGAN/', './images/real_images/')

find_best(DCGAN_images, './images/DCGAN.png')
find_best(WGAN_images, './images/WGAN.png')
find_best(ACGAN_images, './images/ACGAN.png')

print('These are the following FID scores for the GANs:')
print(f'DCGAN: {DCGAN_perf}')
print(f'WGAN: {WGAN_perf}')
print(f'ACGAN: {ACGAN_perf}')