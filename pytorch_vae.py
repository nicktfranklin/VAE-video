# Based on https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.ipynb
import torch
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
from matplotlib import pyplot as plt
import math

print("Pytorch Version {}".format(torch.__version__))

def rescale(X, t_min, t_max):
    r_min = float(np.min(X))
    r_max = float(np.max(X))
    return ((X - r_min) / (r_max - r_min)) * (t_max - t_min) + t_min


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

# Encoder and decoder use the DC-GAN architecture
class Encoder(torch.nn.Module):
    def __init__(self, z_dim, n_channel=3, size_=8):
        super(Encoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 64 * n_channel, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64 * n_channel, 128 * n_channel, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128 * n_channel, 256 * n_channel, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            Flatten(),
            torch.nn.Linear(256 * n_channel * (size_**2), 1024 * n_channel),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024 * n_channel, z_dim)
        ])

    def forward(self, x):
        # print('Encoder')
        # print(x.size())
        for layer in self.model:
            x = layer(x)
            # print(x.size())
        return x


class Decoder(torch.nn.Module):
    def __init__(self, z_dim, n_channel=3, size_=8):
        super(Decoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 1024 * n_channel),
            torch.nn.ReLU(),
            torch.nn.Linear(1024 * n_channel, 256 * (size_**2) * n_channel),
            torch.nn.ReLU(),
            Reshape((256 * n_channel, size_, size_)),
            torch.nn.ConvTranspose2d(256 * n_channel, 128 * n_channel, 4, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128 * n_channel, 64 * n_channel, 4, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64 * n_channel, n_channel, 4, 2, padding=1),
            torch.nn.Sigmoid()
        ])

    def forward(self, x):
        # print('Decoder')
        # print(x.size())
        for layer in self.model:
            x = layer(x)
            # print(x.size())
        return x


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


class Model(torch.nn.Module):
    def __init__(self, z_dim, n_channel=3, size_=8):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim, n_channel, size_)
        self.decoder = Decoder(z_dim, n_channel, size_)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed


# Convert a numpy array of shape [batch_size, height, width, 3] into a displayable array
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples, t_min = 0, t_max = 255):
    # rescale samples to new
    samples = rescale(samples, t_min, t_max)

    _samples = []
    n, height, width, n_channel = np.shape(samples)
    cnt = int(math.floor(math.sqrt(n)))
    _samples = np.zeros((height*cnt, width*cnt, 0))
    for c in range(n_channel):
        _samples0 = np.reshape(samples[:, :, :, c], (n, height, width, 1))
        _samples0 = np.transpose(_samples0, axes=[1, 0, 2, 3])
        _samples0 = np.reshape(_samples0, [height, cnt, cnt, width])
        _samples0 = np.transpose(_samples0, axes=[1, 0, 2, 3])
        _samples0 = np.reshape(_samples0, [height*cnt, width*cnt, 1])
        _samples = np.concatenate([_samples, _samples0], axis=2)
    return np.array(_samples, dtype=int)



def train(
    dataloader,
    z_dim=2,
    n_epochs=10,
    use_cuda=True,
    print_every=100,
    plot_every=500,
    n_channel=3,
    size_=8):

    model = Model(z_dim, n_channel, size_)
    if use_cuda:
        model = model.cuda()
    #print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    i = -1
    for epoch in range(n_epochs):
        for images in dataloader:
            i += 1
            optimizer.zero_grad()
            x = Variable(images, requires_grad=False)
            true_samples = Variable(
                torch.randn(200, z_dim),
                requires_grad=False
            )
            if use_cuda:
                x = x.cuda()
                true_samples = true_samples.cuda()

            z, x_reconstructed = model(x)

            mmd = compute_mmd(true_samples, z)
            nll = (x_reconstructed - x).pow(2).mean()
            loss = nll + mmd
            loss.backward()
            optimizer.step()
            if i % print_every == 0:
                print("(Batch {}) Negative log likelihood is {:.5f}, mmd loss is {:.5f}".format(
                    i, nll.data[0], mmd.data[0]))
            if i % plot_every == 0:
                gen_z = Variable(
                    torch.randn(25, z_dim),
                    requires_grad=False
                )
                if use_cuda:
                    gen_z = gen_z.cuda()
                samples = model.decoder(gen_z)
                samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
                print('Generating Figure')
                plt.subplots(figsize=(10,10))
                plt.axis('off')
                plt.imshow(convert_to_display(samples))
                plt.savefig('samples_batch_{}.png'.format(i), dpi=300)
    return model


def embed(dataloader, model, z_dim, use_cuda=True):
    n, n_c, x_size, y_size = np.shape(dataloader.dataset)
    # print(np.shape(dataloader.dataset))

    Z = np.zeros((n, z_dim))
    X_reconstructed = np.zeros((n, n_c, x_size, y_size))

    for ii, images in enumerate(dataloader):
        x = Variable(images, requires_grad=False)
        if use_cuda:
            x = x.cuda()
        z, x_reconstructed = model(x)
        # print(z)
        # print(x_reconstructed)
        if use_cuda:
            Z[ii, :] = np.reshape(z.data.cpu().numpy(), -1)
            X_reconstructed[ii, :] = np.reshape(x_reconstructed.data.cpu().numpy(), (1, n_c, x_size, y_size))
        else:
            Z[ii, :] = np.reshape(z.data.numpy(), -1)
            X_reconstructed[ii, :] = np.reshape(x_reconstructed.data.numpy(), (1, n_c, x_size, y_size))
    return Z, X_reconstructed


def main(z_dim=100, batch_size=200, n_epochs=10,  use_cuda=False, debug=False):
    # load the datasets
    # load the pre-processed data
    print('Loading Data...')
    X = np.array(np.load('./video_color_proc_64.npy'), dtype=float)
    if debug:
        n = np.shape(X)[0]
        X = X[range(0, n, 10), :, :, :]
    X = torch.Tensor(rescale(X, t_min=0, t_max=1))
    print('Data Loaded!')

    video_train = torch.utils.data.DataLoader(
        np.transpose(X, axes=[0, 3, 1, 2]),
        batch_size=batch_size, shuffle=True, num_workers=3,
        pin_memory=True,
        )
    print('Begin Training')
    model = train(video_train, z_dim=z_dim, n_epochs=n_epochs,
                  use_cuda=use_cuda, size_=8)
    torch.save(model, 'trained_vae_64.pt')
    print('Trainig complete')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~

    print('Encoding Video')

    video_to_embed =  torch.utils.data.DataLoader(
        np.transpose(X, axes=[0, 3, 1, 2]),
        batch_size=1, shuffle=False, num_workers=3,
        pin_memory=True
    )
    Z, X_reconstructed = embed(video_to_embed, model, z_dim)

    np.save('video_color_Z_embedded_64.npy', Z)
    np.save('video_color_X_reconstructed_64.npy', X_reconstructed)

    print('Done!')

if __name__ == "__main__":
    main(z_dim=100, batch_size=256, n_epochs=4000, use_cuda=True, debug=False)
