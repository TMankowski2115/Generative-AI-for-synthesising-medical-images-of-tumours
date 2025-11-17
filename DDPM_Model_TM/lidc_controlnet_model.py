import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ======================================================================
#  TIMESTEP EMBEDDING
# ======================================================================

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ======================================================================
#  RESBLOCK
# ======================================================================

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.act = nn.SiLU()

        self.emb_proj = nn.Linear(emb_dim, out_channels)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.conv1(x)
        emb_add = self.emb_proj(emb)[:, :, None, None]
        h = h + emb_add
        h = self.act(self.norm1(h))

        h = self.conv2(h)
        h = self.norm2(h)

        return self.act(h + self.skip(x))


# ======================================================================
#  ZERO CONV
# ======================================================================

class ZeroConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


# ======================================================================
#  CONTROLNET – 3 kanały (lung_left, lung_right, nodule)
# ======================================================================

class ControlNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4

        # scaling
        self.scale_lung = 0.25
        self.scale_nodule = 0.7

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1, 3, padding=1),
            nn.GroupNorm(8, ch1),
            nn.SiLU()
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(ch1, ch2, 3, padding=1),
            nn.GroupNorm(8, ch2),
            nn.SiLU()
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(ch2, ch3, 3, padding=1),
            nn.GroupNorm(8, ch3),
            nn.SiLU()
        )

        self.out1 = ZeroConv(ch1)
        self.out2 = ZeroConv(ch2)
        self.out3 = ZeroConv(ch3)

        self.pool = nn.AvgPool2d(2)

    def forward(self, cond_img):
        # cond_img: [B,3,H,W]
        cond_img = F.avg_pool2d(cond_img, 5, 1, 2)

        lung_left = cond_img[:, 0:1] * self.scale_lung
        lung_right = cond_img[:, 1:2] * self.scale_lung
        nod = cond_img[:, 2:3] * self.scale_nodule

        cond_scaled = torch.cat([lung_left, lung_right, nod], dim=1)

        h1 = self.down1(cond_scaled)
        c1 = self.out1(h1)
        x = self.pool(h1)

        h2 = self.down2(x)
        c2 = self.out2(h2)
        x = self.pool(h2)

        h3 = self.down3(x)
        c3 = self.out3(h3)

        return c1, c2, c3


# ======================================================================
#  UNET + CONTROLNET
# ======================================================================

class LIDCControlNetUNet(nn.Module):
    def __init__(self, base_channels=64, emb_dim=256, cond_dim=5):
        super().__init__()

        self.emb_dim = emb_dim

        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4

        # embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # down
        self.down1 = ResBlock(1, ch1, emb_dim)
        self.down2 = ResBlock(ch1, ch2, emb_dim)
        self.down3 = ResBlock(ch2, ch3, emb_dim)
        self.mid = ResBlock(ch3, ch3, emb_dim)

        self.pool = nn.AvgPool2d(2)

        # up
        self.up_conv3 = nn.ConvTranspose2d(ch3, ch3, 4, 2, 1)
        self.up3 = ResBlock(ch3 + ch3, ch2, emb_dim)

        self.up_conv2 = nn.ConvTranspose2d(ch2, ch2, 4, 2, 1)
        self.up2 = ResBlock(ch2 + ch2, ch1, emb_dim)

        self.up_conv1 = nn.ConvTranspose2d(ch1, ch1, 4, 2, 1)
        self.up1 = ResBlock(ch1 + ch1, ch1, emb_dim)

        self.out_conv = nn.Conv2d(ch1, 1, 3, padding=1)

        # ControlNet 3-kanałowy
        self.controlnet = ControlNet(in_channels=3, base_channels=base_channels)

    def forward(self, x, t, lung_mask, nodule_mask, cond_vec):
        assert x.ndim == 4, f"x must be [B,1,H,W], got {x.shape}"

        # embeddings
        t_emb = timestep_embedding(t, self.emb_dim)
        t_emb = self.time_embed(t_emb)

        c_emb = self.cond_embed(cond_vec)
        emb = t_emb + c_emb

        # ControlNet
        cond_img = torch.cat([lung_mask, nodule_mask], dim=1)  # lung:2 , nodule:1 → 3 kanały
        c1, c2, c3 = self.controlnet(cond_img)

        # down
        d1 = self.down1(x, emb) + c1
        x1 = self.pool(d1)

        d2 = self.down2(x1, emb) + c2
        x2 = self.pool(d2)

        d3 = self.down3(x2, emb) + c3
        x3 = self.pool(d3)

        mid = self.mid(x3, emb)

        # up
        u3 = self.up_conv3(mid)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up3(u3, emb)

        u2 = self.up_conv2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2, emb)

        u1 = self.up_conv1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1(u1, emb)

        return self.out_conv(u1)
