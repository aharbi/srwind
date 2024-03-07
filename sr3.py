import numpy as np
import unet
import dataset
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

"""
This code is adpated from the implementation in:
https://github.com/TeaPearce/Conditional_Diffusion_MNIST
"""


class RegressionSR3(nn.Module):
    def __init__(self, device="cuda", in_channels=2, num_features=256, model_path=None):
        super(RegressionSR3, self).__init__()

        self.device = device

        self.model = unet.UNet(in_channels=in_channels, num_features=num_features)
        self.model.to(device=device)

        if not (model_path is None):
            self.model.load_state_dict(
                torch.load(model_path, map_location=torch.device(self.device))
            )

    def train(self, dataset, batch_size=64, num_epochs=30, lr=1e-4, save_path=None):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss = nn.MSELoss()

        self.model.train()

        for epoch in range(num_epochs):

            optim.param_groups[0]["lr"] = lr * (1 - epoch / num_epochs)

            pbar = tqdm(dataloader)

            loss_ema = None
            for x, y in pbar:
                optim.zero_grad()

                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x)

                output = loss(y_hat, y)

                output.backward()

                if loss_ema is None:
                    loss_ema = output.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * output.item()

                pbar.set_description("Loss: {:.4f}".format(loss_ema))
                optim.step()

            if not (save_path is None):
                torch.save(
                    self.model.state_dict(),
                    save_path + "regression_sr3_{}.pth".format(epoch),
                )

    def inference(self, x):

        self.model.eval()

        return self.model(x)


class DiffusionSR3(nn.Module):
    def __init__(
        self,
        device="cuda",
        in_channels=4,
        T=400,
        betas=(1e-4, 0.02),
        num_features=256,
        model_path=None,
    ):
        super(DiffusionSR3, self).__init__()

        self.device = device
        self.T = T

        self.model = unet.UNet(
            in_channels=in_channels, num_features=num_features, embedding=True
        )
        self.model.to(device=device)

        for k, v in self.noise_schedule(betas[0], betas[1], self.T).items():
            self.register_buffer(k, v)

        if not (model_path is None):
            self.model.load_state_dict(
                torch.load(model_path, map_location=torch.device(self.device))
            )

    def noise_schedule(self, beta1, beta2, T):

        beta_t = (beta2 - beta1) * torch.arange(
            0, T + 1, dtype=torch.float32
        ) / T + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def train(self, dataset, batch_size=64, num_epochs=30, lr=1e-4, save_path=None):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss = nn.MSELoss()

        self.model.train()

        for epoch in range(num_epochs):

            optim.param_groups[0]["lr"] = lr * (1 - epoch / num_epochs)

            pbar = tqdm(dataloader)

            loss_ema = None
            for x, y in pbar:
                optim.zero_grad()

                noise_hat, noise = self.forward(x, y)

                output = loss(noise_hat, noise)

                output.backward()

                if loss_ema is None:
                    loss_ema = output.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * output.item()

                pbar.set_description("Loss: {:.4f}".format(loss_ema))
                optim.step()

            if not (save_path is None):
                torch.save(
                    self.model.state_dict(),
                    save_path + "diffusion_sr3_{}.pth".format(epoch),
                )

    def forward(self, x, y):
        # Only used for training!
        t = torch.randint(1, self.T + 1, (y.shape[0],))
        noise = torch.randn_like(y)

        y_t = (
            self.sqrtab[t, None, None, None] * y
            + self.sqrtmab[t, None, None, None] * noise
        )

        x_y = torch.cat([x, y_t], dim=1)
        x_y = x_y.to(self.device)

        return self.model(x_y, (t / self.T).to(self.device)), noise

    def inference(self, x):

        y_t = torch.randn(x.shape)

        for i in range(self.T, 0, -1):
            print("Sampling timestep {}".format(i), end="\r")

            t = torch.tensor([i / self.T]).to(self.device)
            t = t.repeat(x.shape[0], 1, 1, 1)

            z = torch.randn(x.shape).to(self.device) if i > 1 else 0

            x_y = torch.cat([x, y_t], dim=1)

            eps = self.model(x_y, t)

            y_t = (
                self.oneover_sqrta[i] * (y_t - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return y_t
