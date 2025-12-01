# dcgan.py  —  one-file, directly runnable DCGAN (train + save + sample)
import os
import argparse
import time
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

# class Generator(nn.Module):
#     def __init__(self, channels: int):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),  # 1x1 -> 4x4
#             nn.BatchNorm2d(1024),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # 4x4 -> 8x8
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),   # 8x8 -> 16x16
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         return self.main(z)
class Generator(nn.Module):
    def __init__(self, channels: int, latent_dim: int = 100):
        super().__init__()
        self.main = nn.Sequential(
            # 初始全连接（可选）或直接使用 ConvTranspose 初始扩张
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),  # 1x1 -> 4x4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 第1次 PixelShuffle 上采样 (4x4 -> 8x8)
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
            nn.PixelShuffle(2),  # 输出通道：1024 -> 256 (因为 1024 = 256 * 2^2)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 第2次 PixelShuffle (8x8 -> 16x16)
            nn.Conv2d(256, 256 * 4, 3, padding=1, bias=False),  # 256*4=1024
            nn.PixelShuffle(2),  # 1024 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 第3次 PixelShuffle (16x16 -> 32x32)
            nn.Conv2d(256, 256 * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 第4次 PixelShuffle (32x32 -> 64x64)
            nn.Conv2d(256, 256 * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 第5次 PixelShuffle (64x64 -> 128x128)
            nn.Conv2d(256, 256 * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 最后一层普通卷积（不进行上采样）
            nn.Conv2d(256, channels, 3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # 输入 z: [B, latent_dim, 1, 1]
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.main = nn.Sequential(
            # Input: (channels, 128, 128) -> Output: (64, 64, 64)
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 64, 64) -> (128, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 32, 32) -> (256, 16, 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (256, 16, 16) -> (512, 8, 8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # (512, 8, 8) -> (1024, 4, 4)
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # (1024, 4, 4) -> (1, 1, 1)
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)  # Flatten to (B,)


# -------------------------------
# Training utilities
# -------------------------------

def get_dataloader(name: str, batch_size: int, channels: int) -> DataLoader:
    # 统一到 32x32，输入归一化到 [-1, 1]，匹配 Tanh 输出
    tfm = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5]*channels,
            std=[0.5]*channels
        )
    ])

    if name.lower() in ["mnist", "fashion-mnist", "fashion_mnist", "fmnist"]:
        is_fashion = name.lower() != "mnist"
        dataset = (datasets.FashionMNIST if is_fashion else datasets.MNIST)(
            root="./data", train=True, download=True, transform=tfm
        )
    elif name.lower() in ["cifar10", "cifar-10"]:
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    else:
        raise ValueError("Unsupported dataset. Use: mnist | fashion-mnist | cifar10")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


def save_samples(G: nn.Module, device, fixed_z, out_dir: str, step: int, nrow: int = 8):
    G.eval()
    with torch.no_grad():
        fake = G(fixed_z.to(device)).detach().cpu()
        fake = (fake * 0.5) + 0.5  # [-1,1] -> [0,1]
        grid = vutils.make_grid(fake[: nrow * nrow], nrow=nrow, padding=2)
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        vutils.save_image(grid, os.path.join(out_dir, "samples", f"step_{step:07d}.png"))
    G.train()


def save_ckpt(G, D, out_dir: str, epoch: int, step: int):
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    torch.save(
        {"epoch": epoch, "step": step, "state_dict":G.state_dict()},
        os.path.join(out_dir, "checkpoints", f"generator_e{epoch:03d}_s{step:07d}.pt")
    )
    torch.save(
        {"epoch": epoch, "step": step, "state_dict": D.state_dict()},
        os.path.join(out_dir, "checkpoints", f"discriminator_e{epoch:03d}_s{step:07d}.pt")
    )


def train(args):
    device = torch.device(f"cuda:{args.cuda_index}")
    print(f"Using device: {device}")

    # channels: MNIST/Fashion=1，CIFAR10=3（可通过参数覆盖）
    loader = get_dataloader(args.dataset, args.batch_size, args.channels)

    G = Generator(args.channels).to(device)
    D = Discriminator(args.channels).to(device)

    criterion = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 固定噪声用于对比可视化
    fixed_z = torch.randn(args.sample_grid**2, 100, 1, 1)

    # 训练
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for i, (imgs, _) in enumerate(loader, start=1):
            imgs = imgs.to(device, non_blocking=True)

            # -----------------
            #  Train D
            # -----------------
            D.train(); G.train()
            z = torch.randn(imgs.size(0), 100, 1, 1, device=device)
            fake = G(z).detach()

            real_logits = D(imgs)
            fake_logits = D(fake)

            real_labels = torch.ones_like(real_logits, device=device)
            fake_labels = torch.zeros_like(fake_logits, device=device)

            d_loss_real = criterion(real_logits, real_labels)
            d_loss_fake = criterion(fake_logits, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()

            # -----------------
            #  Train G
            # -----------------
            z = torch.randn(imgs.size(0), 100, 1, 1, device=device)
            fake = G(z)
            g_logits = D(fake)
            g_loss = criterion(g_logits, real_labels)  # 让D判真

            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

            step += 1

            if step % args.log_every == 0:
                print(f"[e {epoch:03d}/{args.epochs:03d}] [i {i:04d}/{len(loader):04d}] "
                      f"D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}  step: {step}")

            if step % args.sample_every == 0:
                save_samples(G, device, fixed_z, args.out_dir, step, nrow=args.sample_grid)
            #
            # if step % args.save_every == 0:
            #     save_ckpt(G, D, args.out_dir, epoch, step)

        # 每个 epoch 末也存一次
        save_ckpt(G, D, args.out_dir, epoch, step)
        save_samples(G, device, fixed_z, args.out_dir, step, nrow=args.sample_grid)

    # 最终权重（便于直接载入）
    torch.save(G.state_dict(), os.path.join(args.out_dir, "generator_final128.pt"))
    torch.save(D.state_dict(), os.path.join(args.out_dir, "discriminator_final128.pt"))
    print(f"Training done in {time.time()-t0:.1f}s. Checkpoints & samples saved to: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Directly runnable DCGAN (train & save)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion-mnist", "cifar10"],
                        help="dataset name")
    parser.add_argument("--channels", type=int, default=1, help="image channels (1 for MNIST/Fashion, 3 for CIFAR10)")
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cuda", action="store_true", help="use CUDA if available")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="runs_dcgan")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--sample-grid", type=int, default=8, help="nrow for sample grid (nrow*nrow images)")

    args = parser.parse_args()
    # 自动调整 channels：若用户未改 channels 且用 cifar10，则设为3
    if args.dataset.lower() in ["cifar10", "cifar-10"] and args.channels == 1:
        args.channels = 3

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    import time
    main()
