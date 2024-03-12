import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
import numpy as np
import os
from tqdm import tqdm

import dataset

#UNet model - conditioned on noise_level

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()

        self.model = nn.Sequential(
            DoubleConvBlock(in_channels, out_channels), nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            DoubleConvBlock(out_channels, out_channels),
            DoubleConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_features=256, embedding=False):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.num_features = num_features
        self.embedding = embedding

        self.init_conv = DoubleConvBlock(self.in_channels, self.num_features)

        self.encoder_block_1 = UNetEncoder(self.num_features, self.num_features)
        self.encoder_block_2 = UNetEncoder(self.num_features, self.num_features * 2)

        self.bottleneck = nn.Sequential(nn.AvgPool2d(5), nn.GELU())

        if self.embedding:
            self.timeembed1 = EmbedFC(1, 2 * self.num_features)
            self.timeembed2 = EmbedFC(1, 1 * self.num_features)

        self.decoder_block_1 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.num_features, 2 * self.num_features, 5, 5),
            nn.GroupNorm(8, 2 * self.num_features),
            nn.ReLU(),
        )

        self.decoder_block_2 = UNetDecoder(4 * self.num_features, self.num_features)
        self.decoder_block_3 = UNetDecoder(2 * self.num_features, self.num_features)


        if self.embedding:
            self.out = nn.Sequential(
                nn.Conv2d(2 * self.num_features, self.num_features, 3, 1, 1),
                nn.GroupNorm(8, self.num_features),
                nn.ReLU(),
                nn.Conv2d(self.num_features, int(self.in_channels / 2), 3, 1, 1),
            )
        else:
            self.out = nn.Sequential(
                nn.Conv2d(2 * self.num_features, self.num_features, 3, 1, 1),
                nn.GroupNorm(8, self.num_features),
                nn.ReLU(),
                nn.Conv2d(self.num_features, self.in_channels, 3, 1, 1),
            )

    def forward(self, x, noise_level=None):
        x = self.init_conv(x)

        e_1 = self.encoder_block_1(x)
        e_2 = self.encoder_block_2(e_1)

        bottleneck = self.bottleneck(e_2)

        d_1 = self.decoder_block_1(bottleneck)

        if self.embedding:
            if noise_level == None:
                raise Exception("No value of noise level assigned")

            temb1 = self.timeembed1(noise_level).view(-1, self.num_features * 2, 1, 1)
            temb2 = self.timeembed2(noise_level).view(-1, self.num_features, 1, 1)

            d_2 = self.decoder_block_2(d_1 + temb1, e_2)
            d_3 = self.decoder_block_3(d_2 + temb2, e_1)
        else:
            d_2 = self.decoder_block_2(d_1, e_2)
            d_3 = self.decoder_block_3(d_2, e_1)

        out = self.out(torch.cat((d_3, x), 1))

        return out


# Util function
# Returns a specific index t of a passed list of values vals while considering the batch dimension.
def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Diffusion model

class Diffusion(nn.Module):
    def __init__(self, device, num_features=256, channels=4, num_timesteps=400, model_path = None):
        super().__init__()
        self.device = device
        self.num_features = num_features
        self.channels = channels
        self.num_timesteps = num_timesteps

        # Initialize the UNet network
        self.model = UNet(in_channels=channels, num_features=num_features, embedding=True).to(device=self.device) 

        if not (model_path is None):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        

    def set_noise_schedule(self, start=1e-4, end=2e-2):
        """ Defines the noise schedule (gamma_t) and associated coefficients (1/sqrt etc.)
        from a linear distribution of the betas (beta_t)

        Args:
            start, end: boundaries for the choice of the beta_t
        Returns:
            Nothing, but saves the noise schedule in the diffusion model.
        """

        betas = torch.linspace(start, end, self.num_timesteps)
        alphas = 1. - betas
        gammas = torch.cumprod(alphas, dim=0)
        previous_gammas = F.pad(gammas[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.gammas = gammas
        self.inv_sqrt_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1. - alphas)
        self.sqrt_gammas = torch.sqrt(gammas)
        self.sqrt_one_minus_gammas = torch.sqrt(1. - gammas)
        self.previous_gammas = previous_gammas
        self.sqrt_previous_gammas = torch.sqrt(previous_gammas)

    def forward(self, x_low_res, y_start):
        """ Performs one forward step of the diffusion model: 
        - Adds noise to the low-resolution upsampled image. 
        - The noise is chosen uniformly between two consecutive gammas (t-1, t) corresponding to a random timestep t.
        - Predict the noise added based on the UNet model and compute the MSE.

        Args:
            x_low_res: low-resolution upsampled image
            y_start: noisy image obtained at the previous iteration
        Returns:
            The MSE between the noise added and the predicted noise
        """

        b  = y_start.shape[0]
        t = np.random.randint(0, self.num_timesteps)

        sqrt_gamma = torch.FloatTensor(
            np.random.uniform(self.sqrt_previous_gammas[t-1], self.sqrt_previous_gammas[t], size=b)
        ).to(self.device)
        sqrt_gamma = sqrt_gamma.view(-1, 1, 1, 1)

        noise = torch.randn_like(y_start).to(self.device)
        
        # Perturbed image obtained by forward diffusion process at random time step t
        y_noisy = sqrt_gamma * y_start + (1 - sqrt_gamma**2).sqrt() * noise

        # Concatenate original image and noisy image
        concat_x_y_noisy = torch.cat([x_low_res, y_noisy], dim=1).to(self.device)

        # The model predict actual noise added at time step t
        pred_noise = self.model(concat_x_y_noisy, noise_level=sqrt_gamma)
        
        mse_loss = nn.MSELoss()

        return mse_loss(noise, pred_noise)


    @torch.no_grad()
    def inference(self, x_low_res):
        """ Denoises an upsampled low resolution image. 
        Loops over the num_timesteps iterations and iteratively removes the predicted noise.

        Args:
            x_low_res: low-resolution upsampled image
        Returns:
            blurred_image_to_enhance: denoised image
        """
        
        blurred_image_to_enhance = torch.rand_like(x_low_res, device=self.device)
        # Loop over number of timesteps
        for i in reversed(range(0, self.num_timesteps)):
            # Enhance the image at each time step:

            t = torch.full((1,), i, device=self.device, dtype=torch.long)

            # Load parameters corresponding to timestep t
            beta = get_index_from_list(self.betas, t, x_low_res.shape)
            inv_sqrt_alpha = get_index_from_list(self.inv_sqrt_alphas, t, x_low_res.shape)
            sqrt_one_minus_alpha = get_index_from_list(self.sqrt_one_minus_alphas, t, x_low_res.shape)
            sqrt_one_minus_gamma = get_index_from_list(self.sqrt_one_minus_gammas, t, x_low_res.shape)
            sqrt_gamma = get_index_from_list(self.sqrt_gammas, t, x_low_res.shape)
    
            if i > 0:
                noise = torch.randn_like(x_low_res) 
            else:
                noise = torch.zeros_like(x_low_res)
            
            concat_x_y = torch.cat([x_low_res, blurred_image_to_enhance], dim=1)
            blurred_image_to_enhance = inv_sqrt_alpha * (blurred_image_to_enhance - beta * self.model(concat_x_y, noise_level=sqrt_gamma) / sqrt_one_minus_gamma) + sqrt_one_minus_alpha * noise
                    
        return blurred_image_to_enhance


    def train(self, dataset, batch_size=64, num_epochs=30, lr=1e-4, save_path=None):
        """ Trains the diffusion model.

        Args:
            x_low_res: low-resolution upsampled image
        Returns:
            blurred_image_to_enhance: denoised image
        """
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        
        for epoch in range(num_epochs):

            step = 0 
            for x, y in tqdm(dataloader): # Check how to get the data
                
                optimizer.zero_grad() # Not sure if this step is useful

                loss = self.forward(x,y)
                loss.backward()
                optimizer.step()

                step += 1

        if not (save_path is None):
            torch.save(self.model.state_dict(), save_path + "diffusion_{}_epochs.pth".format(num_epochs))

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_features = 256
    channels = 4 # 4 is for 2 position coordinates and 2 components of speed
    num_epochs = 30
    batch_size = 64
    num_timesteps = 400

    diffusion_model = Diffusion(device=device, num_features = num_features, channels = channels, num_timesteps=num_timesteps)
    diffusion_model.set_noise_schedule(start=1e-4, end=2e-2)

    # wind_dataset = dataset.WindDataset("dataset/data_matrix.npy", "dataset/label_matrix.npy")
    
    # To implement only on sub sample
    path = "dataset/test"
    file_names = os.listdir(path)
   
    current_data_matrix, current_label_matrix = dataset.generate_random_dataset(dataset_path= path, save_path="dataset/test/", size=10)

    wind_dataset = dataset.WindDataset(data_matrix_path="dataset/test/data_matrix.npy", label_matrix_path="dataset/test/label_matrix.npy")

    diffusion_model.train(
        dataset=wind_dataset,
        save_path="models/",
        num_epochs=num_epochs,
        batch_size=batch_size,
    )