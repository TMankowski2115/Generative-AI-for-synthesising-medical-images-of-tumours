import os, re, math, random, argparse, pathlib, copy
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils as tvu
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator
import torchvision.transforms.functional as tvf  


IMG, BATCH, EPOCHS, LR = 128, 16, 100, 1e-5
T_STEPS                = 1000
DEVICE                 = "cuda" if torch.cuda.is_available() else "cpu"

_re_lung  = re.compile(r"^(.*)_(?:lung|combined)_mask_(\d+)\.npy$")
_re_nod   = re.compile(r"^(.*)_nodule_mask_(\d+)\.npy$")
_re_slice = re.compile(r"^(.*)_slice_(\d+)\.png$")

def load_png(p):
    return Image.open(p).convert("L")
def load_npy_mask(p):
    a = np.load(p);  a = a[len(a)//2] if a.ndim==3 else a
    return Image.fromarray(((a>0)*255).astype(np.uint8))

# ─── dataset ──────────────────────────────────────────────────────────────
class LungSlice(Dataset):
    def __init__(self, root):
        root = pathlib.Path(root)
        lung, nod, png = {}, {}, {}
        for p in root.rglob("*.npy"):
            if (m:=_re_lung.match(p.name)): lung[(m[1],m[2])] = p
            elif (m:=_re_nod.match(p.name)): nod[(m[1],m[2])]  = p
        for p in root.rglob("*.png"):
            if (m:=_re_slice.match(p.name)): png[(m[1],m[2])] = p

        self.trip = [(lung[k], nod.get(k), png[k]) for k in lung if k in png]
        random.shuffle(self.trip)

        self.ct  = transforms.Compose([
            transforms.Resize((IMG,IMG), antialias=True),
            transforms.ToTensor(), transforms.Lambda(lambda t:t*2-1)
        ])
        self.bin = transforms.Compose([
            transforms.Resize((IMG,IMG), antialias=True,
                              interpolation=Image.NEAREST),
            transforms.ToTensor(), transforms.Lambda(lambda t:(t>.5).float())
        ])

    def __len__(self): return len(self.trip)
    def __getitem__(self,i):
        l,n,c = self.trip[i]
        m_lung = self.bin(load_npy_mask(l))
        ct     = self.ct(load_png(c))
        m_nod  = torch.zeros_like(m_lung)
        if n: m_nod = self.bin(load_npy_mask(n))
        return m_lung, ct, m_nod

# ─── UNet + scheduler ────────────────────────────────────────────────
def build_model():
    unet = UNet2DModel(
        sample_size        = IMG,
        in_channels        = 3,
        out_channels       = 1,
        block_out_channels = (32, 64, 128),          # 3 poziomy
        down_block_types   = ("DownBlock2D",)*3,
        up_block_types     = ("UpBlock2D",)*3,
        layers_per_block   = 1,
    )
    sched = DDPMScheduler(num_train_timesteps=T_STEPS)
    return unet, sched

# ─── wizualizacja ────────────────────────────────────────────────
def outline(mask_bin: torch.Tensor) -> torch.Tensor:           # mask_bin: (1,H,W)
    k = torch.ones(1,1,3,3, device=mask_bin.device)
    edge = (F.conv2d(mask_bin.unsqueeze(0),k,padding=1)>0)[0] & ~mask_bin.bool()
    return edge.squeeze(0)                                     # (H,W)

def pick_cases(dataset, want_pos=2, want_neg=2, total=4):
    pos, neg = [], []
    for i in range(len(dataset)):
        ml, ct, nd = dataset[i]
        (pos if nd.sum() > 0 else neg).append((ml, ct, nd))

    random.shuffle(pos); random.shuffle(neg)
    out = pos[:want_pos] + neg[:want_neg]

    rest = pos[want_pos:] + neg[want_neg:]
    while len(out) < total and rest:
        out.append(rest.pop())
    while len(out) < total:
        out.append(random.choice(out))

    return out[:total]

@torch.no_grad()
def sample_grid(unet, base_sched, dataset, fn, steps=50):
    unet.eval()
    cases = pick_cases(dataset)
    mask  = torch.stack([c[0] for c in cases]).to(DEVICE)
    y_gt  = torch.stack([c[1] for c in cases])
    nod   = torch.stack([c[2] for c in cases]).to(DEVICE)

    scheduler = copy.deepcopy(base_sched)         
    scheduler.set_timesteps(steps, device=DEVICE)  

    x = torch.randn((4,1,IMG,IMG), device=DEVICE)
    for t in scheduler.timesteps:
        eps = unet(torch.cat([mask,nod,x],1), t).sample
        x   = scheduler.step(eps, t, x).prev_sample
    gen = x.clamp(-1,1).cpu()

    rows=[]
    for n,y,g in zip(nod.cpu(), y_gt, gen):
        n3 = (n.repeat(3,1,1)*2-1); y3=y.repeat(3,1,1); g3=g.repeat(3,1,1)
        ov=g3.clone()
        col=torch.tensor([1.,0.,0.],device=ov.device,dtype=ov.dtype).view(3,1)
        ov[:, outline(n)] = col
        rows.extend([n3,y3,g3,ov])

    grid=tvu.make_grid(rows,nrow=4,normalize=True,value_range=(-1,1))
    img=tvf.to_pil_image(grid); hdr=28
    full=Image.new("RGB",(img.width,img.height+hdr),(255,255,255)); full.paste(img,(0,hdr))
    d=ImageDraw.Draw(full); f=ImageFont.load_default()
    for i,lbl in enumerate(["MASKA","ORYG","GEN","GEN+MASK"]):
        w,h=d.textbbox((0,0),lbl,font=f)[2:]
        d.text((i*IMG+(IMG-w)//2,(hdr-h)//2),lbl,font=f,fill=0)
    full.save(fn)

# ─── trening  ────────────────────────────────────────────────
def train(dataset_dir, save_dir):
    ds=LungSlice(dataset_dir)
    loader=DataLoader(ds,batch_size=BATCH,shuffle=True,
                      pin_memory=True,num_workers=0,drop_last=True)
    unet,scheduler=build_model()
    optim=torch.optim.AdamW(unet.parameters(),lr=LR)
    acc=Accelerator(); unet,optim,loader=acc.prepare(unet,optim,loader)

    for ep in range(1,EPOCHS+1):
        unet.train(); tot=0
        for mask,x0,nod in tqdm(loader,desc=f"ep{ep:02d}",ncols=80):
            b=x0.size(0)
            t=torch.randint(0,scheduler.config.num_train_timesteps,
                            (b,),device=x0.device)
            noise=torch.randn_like(x0)
            x_t=scheduler.add_noise(x0,noise,t)
            loss=F.mse_loss(unet(torch.cat([mask,nod,x_t],1),t).sample,noise)
            acc.backward(loss); optim.step(); optim.zero_grad()
            tot+=loss.item()*b

        if acc.is_main_process:
            print(f"[ep {ep:02d}] loss={tot/len(ds):.4f}")
            ep_dir=os.path.join(save_dir,f"ep{ep:03d}")
            os.makedirs(ep_dir,exist_ok=True)
            unet.save_pretrained(ep_dir)
            sample_grid(unet, scheduler, ds,
                        os.path.join(save_dir,f"samples_ep{ep:03d}.png"), steps=50)

# ─── main ────────────────────────────────────────────────────────────────
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",required=True)
    parser.add_argument("--save_dir", default="ct_ddpm_small_results")
    args=parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    train(args.dataset_dir,args.save_dir)
