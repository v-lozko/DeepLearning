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

    def forward(self, noise, text_input):
        text_emb = self.text_embedding(text_input)
        combined = torch.cat((noise, text_emb), dim=1)
        x = self.fc(combined)
        x = x.view(-1, 512, 4, 4)
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, text_dim=10):
        super(Critic, self).__init__()
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU()
        )
        self.image_features = nn.Sequential(
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
        self.final_layers = nn.Sequential(
            nn.Conv2d(512 + 256, 1, kernel_size=1),
            nn.Flatten()
        )

    def forward(self, image, text_input):
        text_emb = self.text_embedding(text_input).view(-1, 256, 1, 1)
        text_emb = text_emb.repeat(1, 1, 4, 4)
        image_features = self.image_features(image)
        combined = torch.cat((image_features, text_emb), dim=1)
        return self.final_layers(combined)

def train(train_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs= 50
    noise_dim = 100
    n_critic = 5

    G = Generator(noise_dim=noise_dim).to(device)
    C = Critic().to(device)

    optimizer_G = optim.RMSprop(G.parameters(), lr=0.00005)
    optimizer_C = optim.RMSprop(C.parameters(), lr=0.00005)

    critic_losses = []
    generator_losses = []

    for epoch in range(num_epochs):
        running_loss_C = 0.0
        running_loss_G = 0.0

        for i, (real_images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)

            text_input = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)

            for _ in range(n_critic):
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake_images = G(noise, text_input)

                output_real = C(real_images, text_input)
                output_fake = C(fake_images.detach(), text_input)

                loss_C = -torch.mean(output_real) + torch.mean(output_fake)

                optimizer_C.zero_grad()
                loss_C.backward()
                optimizer_C.step()

                running_loss_C += loss_C.item()

                critic_losses.append(loss_C.item())

                for p in C.parameters():
                    p.data.clamp_(-0.01, 0.01)

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_images = G(noise, text_input)
            output_fake = C(fake_images, text_input)

            loss_G = -torch.mean(output_fake)

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            running_loss_G += loss_G.item()

        critic_losses.append(running_loss_C / len(train_loader))
        generator_losses.append(running_loss_G / len(train_loader))

        tqdm.write(
            f"Epoch [{epoch + 1}/{num_epochs}] Avg C Loss: {running_loss_C / len(train_loader):.4f}, Avg G Loss: {running_loss_G / len(train_loader):.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(critic_losses, label='Critic Loss')
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Critic and Generator Loss')
    plt.legend()
    plt.savefig('./WGAN_loss')

    torch.save(G.state_dict(), "./models/wgan_generator.pth")
    torch.save(C.state_dict(), "./models/wgan_critic.pth")

def test():
    G = Generator(noise_dim=100, text_dim=10).to(device)
    G.load_state_dict(torch.load("./models/wgan_generator.pth", map_location=device, weights_only=True))
    G.eval()

    num_per_class = 100
    noise_dim = 100
    num_classes = 10

    generate_images = {}

    with torch.no_grad():
        for label in range(num_classes):
            noise = torch.randn(num_per_class, noise_dim, device=device)
            labels = torch.full((num_per_class,), label, dtype=torch.long, device=device)
            text_input = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)

            fake_images = G(noise, text_input)
            fake_images = (fake_images + 1) / 2

            generate_images[label] = fake_images

    return generate_images

