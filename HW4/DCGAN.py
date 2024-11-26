# Valery Lozko
# CPSC8430 HW4

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

os.makedirs('./models', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, noise_dim=100, text_dim=10, feature_map_size=512):
        super(Generator, self).__init__()
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 256, 4 * 4 * feature_map_size),
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(feature_map_size, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
    def forward(self, noise, text_input):
        text_emb = self.text_embedding(text_input)
        combined = torch.cat((noise, text_emb), dim=1)
        x = self.fc(combined)
        x = x.view(-1, 512, 4, 4)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, text_dim=10):
        super(Discriminator, self).__init__()
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU()
        )
        self.image_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.LeakyReLU(0.2)
        )
        self.final_layers = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=1),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text_input):
        text_emb = self.text_embedding(text_input).view(-1, 256, 1, 1)
        text_emb = text_emb.repeat(1, 1, 4, 4)
        image_features = self.image_features(image)
        combined = torch.cat((image_features, text_emb), dim=1)
        return self.final_layers(combined)

def train(train_loader):
    noise_dim = 100
    text_dim = 10
    lr = 0.0002
    beta1 = 0.5

    G = Generator(noise_dim=noise_dim, text_dim=text_dim)
    D = Discriminator(text_dim=text_dim)

    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    criterion = nn.BCELoss()

    num_epochs = 50

    real_label = 0.9
    fake_label = 0.0

    discriminator_losses = []
    generator_losses = []

    for epoch in range(num_epochs):
        running_loss_D = 0.0
        running_loss_G = 0.0

        for i, (real_images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)

            text_input = torch.nn.functional.one_hot(labels, num_classes=text_dim).float().to(device)

            if i % 2 == 0:
                D.zero_grad()

                real_labels = torch.full((batch_size, 1), real_label, device=device)
                output_real = D(real_images, text_input)
                loss_real = criterion(output_real, real_labels)

                noise = torch.randn(batch_size, noise_dim, device=device)
                fake_images = G(noise, text_input)
                fake_labels = torch.full((batch_size, 1), fake_label, device=device)
                output_fake = D(fake_images.detach(), text_input)
                loss_fake = criterion(output_fake, fake_labels)

                loss_D = loss_real + loss_fake
                loss_D.backward()
                optimizer_D.step()

                running_loss_D += loss_D.item()

            G.zero_grad()
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_images = G(noise, text_input)
            output = D(fake_images, text_input)

            loss_G = criterion(output, real_labels)
            loss_G.backward()
            optimizer_G.step()

            running_loss_G += loss_G.item()

        discriminator_losses.append(running_loss_D / len(train_loader))
        generator_losses.append(running_loss_G / len(train_loader))

        tqdm.write(
            f"Epoch [{epoch + 1}/{num_epochs}] Avg D Loss: {running_loss_D / len(train_loader):.4f}, Avg G Loss: {running_loss_G / len(train_loader):.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Discriminator and Generator Loss')
    plt.legend()
    plt.savefig('./DCGAN_loss')

    torch.save(G.state_dict(), "./models/dcgan_generator.pth")
    torch.save(D.state_dict(), "./models/dcgan_discriminator.pth")

def test():
    G = Generator(noise_dim=100, text_dim=10).to(device)
    G.load_state_dict(torch.load("./models/dcgan_generator.pth", map_location=device, weights_only=True))
    G.eval()

    num_per_class = 100
    noise_dim = 100
    num_classes = 10

    generated_images = {}

    with torch.no_grad():
        for label in range(num_classes):
            noise = torch.randn(num_per_class, noise_dim, device=device)
            labels = torch.full((num_per_class,), label, dtype=torch.long, device=device)
            text_input = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)

            fake_images = G(noise, text_input)
            fake_images = (fake_images + 1) / 2

            generated_images[label] = fake_images

    return generated_images
