import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
dataset = torchvision.datasets.MNIST(root='dataset/', transform=transform, download=True, train=True)
dataset.class_to_idx
batch_size = 32
loader = DataLoader(dataset, batch_size, shuffle=True)
plt.figure(figsize=(1,1))
plt.imshow(next(iter(loader))[0][0].permute((1,2,0)), cmap='gray')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
from torch import nn
class Discriminator(nn.Module):
  def __init__(self, in_features):
    super().__init__()

    self.disc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.LeakyReLU(0.01),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.disc(x)
class Generator(nn.Module):
  def __init__(self, z_dim, img_dim):
    super().__init__()

    self.gen = nn.Sequential(
        nn.Linear(z_dim, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, img_dim),
        nn.Tanh()
    )

  def forward(self, x):
    return self.gen(x)
z_dim = 64
image_dim = 28*28*1

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
criterion = nn.BCELoss()

opt_disc = torch.optim.Adam(disc.parameters(), lr=3e-4)
opt_gen = torch.optim.Adam(gen.parameters(), lr=3e-4)

epochs = 10
writer_fake = SummaryWriter('logs/fake')
writer_real = SummaryWriter('logs/real')
step = 0
for epoch in range(epochs):
  for b, (real, _) in enumerate(loader):
    real = real.view(-1, 784).to(device)

    noise = torch.randn(batch_size, z_dim).to(device)

    fake = gen(noise)

    #training the Discriminator

    disc_real = disc(real).view(-1)

    lossD_real = criterion(disc_real, torch.ones_like(disc_real))

    disc_fake = disc(fake).view(-1)

    lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

    lossD = (lossD_real + lossD_fake)/2

    disc.zero_grad()

    lossD.backward(retain_graph=True)

    opt_disc.step()

    # training the Generator

    output = disc(fake).view(-1)

    lossG = criterion(output, torch.ones_like(output))

    gen.zero_grad()

    lossG.backward()

    opt_gen.step()

    if b ==0:
      print(f"Epoch [{epoch}/{epochs}] Batch {b}/{len(loader)} LossD : {lossD:.4f} LossG : {lossG:.4f}")

      with torch.no_grad():
        fake = gen(fixed_noise).reshape(-1,1,28,28)
        data = real.reshape(-1,1,28,28)

        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(real, normalize=True)

        writer_fake.add_image('MNIST FAKE IMAGES', img_grid_fake, global_step=step)
        writer_real.add_image('MNIST REAL IMAGES', img_grid_real, global_step=step)
        step+=1
% load_ext tensorboard
% tensorboard --logdir logs
