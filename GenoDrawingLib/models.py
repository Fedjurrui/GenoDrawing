
from torch import nn
class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.latent_space = dims
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, dims),
            nn.Sigmoid()
        )

    def forward(self, x1):
        x1 = self.encoder_cnn(x1)
        x1 = self.flatten(x1)
        x1 = nn.Dropout(0.3)(x1)
        x = self.encoder_lin(x1)
        return x
class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.latent_space = dims
        self.decoder_lin = nn.Sequential(
            nn.Linear(dims, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 8192),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.decoder_unflatten = nn.Unflatten(1, unflattened_size=(128, 8, 8))
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1),
            nn.Sigmoid(),

        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.decoder_unflatten(x)
        x = self.decoder_cnn(x)

        return x

class Embedding_predictor(nn.Module):
    def __init__(self, n_snps, dims):
        super(Embedding_predictor, self).__init__()
        self.dims = dims
        self.snp_predictor = nn.Sequential(
            nn.Linear(n_snps, 300),
            nn.Sigmoid(),
            nn.Linear(300, dims),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.snp_predictor(x)
        return x
