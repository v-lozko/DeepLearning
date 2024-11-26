# Valery Lozko
# CPSC8430 HW4

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('./models', exist_ok=True)

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, feature_map_size=512):
        super(Generator, self).__init__()
        self.class_embedding = nn.Embedding(num_classes, 50)

        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 50, 4 * 4 * feature_map_size),
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(feature_map_size, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.class_embedding(labels)
        combined_input = torch.cat((noise, label_emb), dim=1)
        x = self.fc(combined_input)
        x = x.view(-1, 512, 4, 4)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.adv_layer = nn.Linear(4 * 4 * 512, 1)
        self.aux_layer = nn.Linear(4 * 4 * 512, num_classes)

    def forward(self, image):
        features = self.feature_extractor(image).view(-1, 4 * 4 * 512)
        validity = self.adv_layer(features)
        label_logits = self.aux_layer(features)
        return validity, label_logits

def train(train_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs= 50
    noise_dim= 100
    num_classes= 10

    G = Generator()
    D = Discriminator()

    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    adversarial_loss = nn.BCEWithLogitsLoss()
    auxiliary_loss = nn.CrossEntropyLoss()

    discriminator_losses = []
    generator_losses = []

    for epoch in range(num_epochs):
        running_loss_D = 0.0
        running_loss_G = 0.0

        for i, (real_images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            real_images = real_images.to(device)
            real_labels = labels.to(device)
            batch_size = real_images.size(0)

            optimizer_D.zero_grad()

            real_validity, real_label_logits = D(real_images)
            real_validity_loss = adversarial_loss(real_validity, torch.ones((batch_size, 1), device=device))
            real_label_loss = auxiliary_loss(real_label_logits, real_labels)

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_images = G(noise, fake_labels)

            fake_validity, fake_label_logits = D(fake_images.detach())
            fake_validity_loss = adversarial_loss(fake_validity, torch.zeros((batch_size, 1), device=device))

            loss_D = real_validity_loss + fake_validity_loss + real_label_loss
            loss_D.backward()
            optimizer_D.step()

            running_loss_D += loss_D.item()

            optimizer_G.zero_grad()

            fake_validity, fake_label_logits = D(fake_images)
            g_validity_loss = adversarial_loss(fake_validity, torch.ones((batch_size, 1), device=device))
            g_label_loss = auxiliary_loss(fake_label_logits, fake_labels)

            loss_G = g_validity_loss + g_label_loss
            loss_G.backward()
            optimizer_G.step()

            running_loss_G += loss_G.item()

        discriminator_losses.append(loss_D.item())
        generator_losses.append(loss_G.item())

        avg_loss_D = running_loss_D / len(train_loader)
        avg_loss_G = running_loss_G / len(train_loader)

        tqdm.write(
            f"Epoch [{epoch + 1}/{num_epochs}] Avg D Loss: {avg_loss_D:.4f}, Avg G Loss: {avg_loss_G:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Discriminator and Generator Loss')
    plt.legend()
    plt.savefig('./ACGAN_loss')

    torch.save(G.state_dict(), "./models/acgan_generator.pth")
    torch.save(D.state_dict(), "./models/acgan_discriminator.pth")

def test():
    G = Generator(noise_dim=100, num_classes=10).to(device)
    G.load_state_dict(torch.load("./models/acgan_generator.pth", map_location=device, weights_only=True))
    G.eval()

    num_per_class = 100
    noise_dim = 100
    num_classes = 10

    generated_images = {}

    with torch.no_grad():
        for label in range(num_classes):
            noise = torch.randn(num_per_class, noise_dim, device=device)
            labels = torch.full((num_per_class,), label, dtype=torch.long, device=device)

            fake_images = G(noise, labels)
            fake_images = (fake_images + 1) / 2

            generated_images[label] = fake_images

    return generated_images

