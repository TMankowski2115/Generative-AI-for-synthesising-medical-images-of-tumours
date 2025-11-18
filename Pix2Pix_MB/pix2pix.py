import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Parametry
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 200
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 100
LAMBDA_SSIM = 5
SAVE_DIR = "training_results_SSIM"
os.makedirs(SAVE_DIR, exist_ok=True)

# SSIM Loss
def gaussian_kernel(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def create_window(window_size=11, sigma=1.5, channel=1):
    _1D_window = gaussian_kernel(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, sigma=1.5, size_average=True, val_range=None):

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
    
    padd = 0
    (_, channel, height, width) = img1.size()
    
    window = create_window(window_size, sigma, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_loss(img1, img2, window_size=11, sigma=1.5, size_average=True):
    return 1 - ssim(img1, img2, window_size, sigma, size_average)

class LungCTDataset(Dataset):
    def __init__(self, dataset_type='train', category='with_nodules', transform=None):

        self.transform = transform
        self.dataset_type = dataset_type
        self.category = category
        
        base_path = f"dataset_lidc_masks/{dataset_type}/{category}"
        self.img_dir = base_path
        self.mask_dir = base_path
        
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        
        img_path = os.path.join(self.img_dir, img_file)
        
        if self.category == "clean":
            # _lung_mask_ dla maski płuc
            mask_file = img_file.replace('_slice_', '_lung_mask_').replace('.png', '.npy')
            mask_path = os.path.join(self.mask_dir, mask_file)
        else:
            # _lung_mask_ dla maski płuc z guzkiem
            mask_file = img_file.replace('_slice_', '_combined_mask_').replace('.png', '.npy')
            mask_path = os.path.join(self.mask_dir, mask_file)
        
        img = Image.open(img_path).convert('L')
        
        mask_array = np.load(mask_path)
        mask = Image.fromarray(mask_array.astype(np.uint8))
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return mask, img, self.category

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # 128 ->64
        e2 = self.enc2(e1)   # 64 ->32
        e3 = self.enc3(e2)   # 32 ->16
        e4 = self.enc4(e3)   # 16 ->8
        e5 = self.enc5(e4)   # 8 ->4
        e6 = self.enc6(e5)   # 4 ->2 (bottleneck 2x2)
        
        # Decoder
        d1 = self.dec1(e6)
        d1 = torch.cat([d1, e5], 1)
        
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e4], 1)
        
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e3], 1)
        
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e2], 1)
        
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e1], 1)
        
        out = self.final(d5)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname in ['Conv2d', 'ConvTranspose2d']:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname == 'InstanceNorm2d' and m.affine:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def denorm(tensor):
    return tensor * 0.5 + 0.5

def update_learning_rate(optimizer, epoch, initial_lr, decay_start_epoch=100, total_epochs=200):
    if epoch < decay_start_epoch:
        lr = initial_lr
    else:
        decay_progress = (epoch - decay_start_epoch) / (total_epochs - decay_start_epoch)
        lr = initial_lr * (1.0 - decay_progress)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if __name__ == "__main__":
    # Utworzenie datasetu
    train_with_nodules = LungCTDataset('train', 'with_nodules', transform=transform)
    train_clean = LungCTDataset('train', 'clean', transform=transform)
    train_dataset = ConcatDataset([train_with_nodules, train_clean])
    
    # Datasety testowe
    test_with_nodules = LungCTDataset('test', 'with_nodules', transform=transform)
    test_clean = LungCTDataset('test', 'clean', transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Rozmiar zbioru treningowego: {len(train_dataset)}")
    print(f"- Z guzkami: {len(train_with_nodules)}")
    print(f"- Bez guzków: {len(train_clean)}")
    print(f"Rozmiar zbioru testowego z guzkami: {len(test_with_nodules)}")
    print(f"Rozmiar zbioru testowego bez guzków: {len(test_clean)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = UNetGenerator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    lr_history = []

    for epoch in range(EPOCHS):
        current_lr_G = update_learning_rate(optimizer_G, epoch, LR, decay_start_epoch=100, total_epochs=EPOCHS)
        current_lr_D = update_learning_rate(optimizer_D, epoch, LR, decay_start_epoch=100, total_epochs=EPOCHS)
        lr_history.append(current_lr_G)
        
        loss_G_mean = 0
        loss_D_mean = 0
        ssim_mean = 0
        
        for i, (mask, img, category) in enumerate(train_dataloader):
            mask = mask.to(device)
            img = img.to(device)
            
            optimizer_D.zero_grad()
            
            real_output = D(mask, img)
            real_target = torch.ones_like(real_output)
            loss_real = criterion_GAN(real_output, real_target)
            
            fake_img = G(mask)
            fake_output = D(mask, fake_img.detach())
            fake_target = torch.zeros_like(fake_output)
            loss_fake = criterion_GAN(fake_output, fake_target)
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            optimizer_G.zero_grad()
            fake_output = D(mask, fake_img)
            gen_target = torch.ones_like(fake_output)
            loss_GAN = criterion_GAN(fake_output, gen_target)
            loss_L1 = criterion_L1(fake_img, img) * LAMBDA_L1
            
            with torch.no_grad():
                current_ssim = ssim(fake_img, img)
                ssim_mean += current_ssim.item()
            
            loss_SSIM = ssim_loss(fake_img, img) * LAMBDA_SSIM
            
            loss_G = loss_GAN + loss_L1 + loss_SSIM
            loss_G.backward()
            optimizer_G.step()
            
            loss_G_mean += loss_G.item()
            loss_D_mean += loss_D.item()
        
        loss_G_mean /= len(train_dataloader)
        loss_D_mean /= len(train_dataloader)
        ssim_mean /= len(train_dataloader)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}] LR: {current_lr_G:.6f} Loss D: {loss_D_mean:.4f} Loss G: {loss_G_mean:.4f} SSIM: {ssim_mean:.4f}')
        
        # zapisywanie logów i przykładowych zdjęc
        if (epoch+1) % 5 == 0:
            with open(os.path.join(SAVE_DIR, 'training_log.txt'), 'a') as f:
                f.write(f'Epoch {epoch+1}: LR = {current_lr_G:.6f}, Loss D = {loss_D_mean:.4f}, Loss G = {loss_G_mean:.4f}, SSIM = {ssim_mean:.4f}\n')
            
        if (epoch+1) % 10 == 0:
            G.eval()
            fig, axes = plt.subplots(10, 3, figsize=(15, 40))
            
            for i in range(5):
                if len(test_with_nodules) > 0:
                    sample_idx = np.random.randint(0, len(test_with_nodules))
                    sample_mask, sample_img, category = test_with_nodules[sample_idx]
                    sample_mask = sample_mask.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        sample_fake_img = G(sample_mask)
                    
                    sample_mask_np = denorm(sample_mask.squeeze().cpu()).numpy()
                    sample_img_np = denorm(sample_img.squeeze().cpu()).numpy()
                    sample_fake_img_np = denorm(sample_fake_img.squeeze().cpu()).numpy()
                    
                    axes[i, 0].imshow(sample_mask_np, cmap='gray')
                    axes[i, 0].set_title(f'Mask with nodules {i+1}')
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(sample_fake_img_np, cmap='gray')
                    axes[i, 1].set_title(f'Generated {i+1}')
                    axes[i, 1].axis('off')
                    
                    axes[i, 2].imshow(sample_img_np, cmap='gray')
                    axes[i, 2].set_title(f'Real with nodules {i+1}')
                    axes[i, 2].axis('off')
            
            for i in range(5, 10):
                if len(test_clean) > 0:
                    sample_idx = np.random.randint(0, len(test_clean))
                    sample_mask, sample_img, category = test_clean[sample_idx]
                    sample_mask = sample_mask.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        sample_fake_img = G(sample_mask)
                    
                    sample_mask_np = denorm(sample_mask.squeeze().cpu()).numpy()
                    sample_img_np = denorm(sample_img.squeeze().cpu()).numpy()
                    sample_fake_img_np = denorm(sample_fake_img.squeeze().cpu()).numpy()
                    
                    axes[i, 0].imshow(sample_mask_np, cmap='gray')
                    axes[i, 0].set_title(f'Clean mask {i-4}')
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(sample_fake_img_np, cmap='gray')
                    axes[i, 1].set_title(f'Generated clean {i-4}')
                    axes[i, 1].axis('off')
                    
                    axes[i, 2].imshow(sample_img_np, cmap='gray')
                    axes[i, 2].set_title(f'Real clean {i-4}')
                    axes[i, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f'epoch_{epoch+1}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            G.train()
            
            torch.save({
                'epoch': epoch+1,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'loss_D': loss_D_mean,
                'loss_G': loss_G_mean,
                'learning_rate': current_lr_G,
            }, os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
