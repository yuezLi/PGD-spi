import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ---------------------------
# 1) 你的 DCGAN 生成器（和 models/dcgan.py 保持一致）
# ---------------------------
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

# ---------------------------
# 2) 一些工具
# ---------------------------
def make_A_gaussian(n, m, seed=0, device='cuda'):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0/np.sqrt(m), size=(m, n)).astype(np.float32)
    return torch.from_numpy(A).to(device)  # (m,n)

def build_A_inpainting(mask_hw, device='cuda'):
    """
    mask_hw: torch.float32 (H,W) 或 (1,H,W)，值为1表示被“观测”的像素
    生成 A 为“选取这些像素”的行选择矩阵：y=mask[x==1] 的堆叠
    """
    if mask_hw.dim() == 3:
        mask_hw = mask_hw[0]
    H, W = mask_hw.shape
    idx = torch.nonzero(mask_hw.reshape(-1) > 0.5, as_tuple=False).squeeze(1)
    m = idx.numel()
    n = H*W
    # 稀疏更合理，但为了和 PGD 代码兼容，这里返回 (m,n) 的稠密选择矩阵
    A = torch.zeros((m, n), dtype=torch.float32, device=device)
    A[torch.arange(m, device=device), idx] = 1.0
    return A  # (m,n)

def load_mask(path, H=32, W=32, device='cuda'):
    if path.lower().endswith('.npy'):
        arr = np.load(path)
        if arr.ndim == 2:
            pass
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        else:
            raise ValueError('mask npy should be (H,W) or (1,H,W)')
        img = torch.from_numpy(arr.astype(np.float32))
        if img.shape != (H,W):
            img = torch.nn.functional.interpolate(img.unsqueeze(0).unsqueeze(0), size=(H,W), mode='nearest')[0,0]
    else:
        # 读图并转灰度
        from PIL import Image
        img = Image.open(path).convert('L').resize((W,H))
        img = torch.from_numpy(np.array(img)/255.0).float()
    # 二值化
    img = (img > 0.5).float()
    return img.to(device)  # (H,W)

def get_A(args, n, H=32, W=32, device='cuda'):
    if args.A_type == 'gaussian':
        A = make_A_gaussian(n, args.m, seed=args.seed, device=device)
    elif args.A_type == 'inpainting':
        assert args.mask_path != '', 'inpainting 需要 --mask_path'
        mask = load_mask(args.mask_path, H=H, W=W, device=device)
        A_full = build_A_inpainting(mask, device=device)  # (m_full, n)
        # 若指定 m，小于可观测像素数时，随机再抽 m 行
        if args.m > 0 and args.m < A_full.shape[0]:
            perm = torch.randperm(A_full.shape[0], device=device)[:args.m]
            A = A_full[perm]
        else:
            A = A_full
    elif args.A_type == 'from_npy':
        assert args.A_path != '', 'from_npy 需要 --A_path 指向 (m,n) 的 .npy'
        A_np = np.load(args.A_path).astype(np.float32)
        assert A_np.shape[1] == n, f"A shape mismatch: got {A_np.shape}, expected (m,{n})"
        A = torch.from_numpy(A_np).to(device)
    else:
        raise ValueError('unknown A_type')

    if args.save_A and args.A_type != 'from_npy':
        os.makedirs(os.path.dirname(args.A_out), exist_ok=True)
        np.save(args.A_out, A.detach().cpu().numpy())
        print('Saved A to', args.A_out)
    return A


def to_vec(img_t):  # (1,C,32,32) -> (n,)
    return img_t.view(-1)

def from_vec(x_vec, C=1, H=32, W=32):
    return x_vec.view(1, C, H, W)

def clamp_tanh_range(x):
    # 生成器输出是 [-1,1]，PGD里也保持该范围更稳
    return x.clamp(-1.0, 1.0)

def save_vis(x_true, x_rec, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with torch.no_grad():
        # 显示到 [0,1]
        def denorm(v): return (v + 1.0) * 0.5
        fig, axs = plt.subplots(1, 2, figsize=(6,3))
        axs[0].imshow(denorm(x_true.squeeze(0).squeeze(0)).cpu().numpy(), cmap='gray')
        axs[0].set_title('Target (x*)'); axs[0].axis('off')
        axs[1].imshow(denorm(x_rec.squeeze(0).squeeze(0)).cpu().numpy(), cmap='gray')
        axs[1].set_title('Reconstruction'); axs[1].axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

# ---------------------------
# 3) 投影步骤： argmin_z ||G(z) - w||^2
#    支持 warm-start & 多重随机重启
# ---------------------------
def power_sigma_max(A, iters=30):
    v = torch.randn(A.size(1), device=A.device)
    v = v / v.norm()
    for _ in range(iters):
        v = (A.t() @ (A @ v))
        v = v / (v.norm() + 1e-12)
    Av = A @ v
    return (Av.norm() / (v.norm() + 1e-12)).item()
def project_onto_G(G, w_img, z_prev=None, z_dim=100, iters=200, lr=0.1, restarts=1, device='cuda'):
    G.eval()
    best_loss, best_x, best_z = None, None, None

    for r in range(restarts):
        if z_prev is not None and r == 0:
            z = z_prev.clone().detach()
        else:
            z = torch.randn(1, z_dim, 1, 1, device=device)

        z.requires_grad_(True)
        opt = optim.Adam([z], lr=lr, betas=(0.9, 0.999))

        for _ in range(iters):
            x_hat = G(z)  # (1,C,32,32), [-1,1]
            loss = ((x_hat - w_img)**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 可选：对 z 做个轻微 clamp 避免发散（一般不需要）
            # with torch.no_grad():
            #     z.copy_(z.clamp(-3, 3))

        with torch.no_grad():
            x_hat = G(z)
            final_loss = ((x_hat - w_img)**2).mean().item()
            if (best_loss is None) or (final_loss < best_loss):
                best_loss = final_loss
                best_x = x_hat.detach()
                best_z = z.detach()

    return best_x, best_z

# ---------------------------
# 4) 主算法：PGD-GAN
# ---------------------------
def pgd_gan(G, A, y, T=10, eta=0.5,
            inner_iters=200, inner_lr=0.1, restarts=1,
            z_init=None,save_dir=None ,device='cuda', C=1, H=32, W=32):
    """
    A: (m, n), y: (m,), x in R^n with n=C*H*W
    """
    n = C*H*W
    A_t = A.t()  # (n, m)
    # 初始化 x0 = 0
    x = torch.zeros(n, device=device)
    if eta is None:
        sigma_max = power_sigma_max(A)
        eta = 1.0 / (sigma_max**2 + 1e-12)
    z_prev = z_init
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for t in range(T):
        # w_t = x_t + eta * A^T (y - A x_t)
        Ax = A @ x  # (m,)
        w = x + eta * (A_t @ (y - Ax))  # (n,)
        w_img = from_vec(w, C, H, W)
        with torch.no_grad():
            w_img_fixed = w_img.detach()

        # 投影: x_{t+1} = PG(w_t)
        x_hat_img, z_prev = project_onto_G(
            G, w_img_fixed, z_prev=z_prev,
            iters=inner_iters, lr=inner_lr, restarts=restarts,
            device=device
        )
        x = to_vec(x_hat_img)  # 回到向量
        if save_dir is not None:
            with torch.no_grad():
                img = (x_hat_img.squeeze().cpu().numpy() + 1) / 2.0  # [-1,1] → [0,1]
                plt.imsave(os.path.join(save_dir, f"iter_{t+1}.png"), img, cmap="gray")

        # 可选：打印监控
        with torch.no_grad():
            meas_loss = ((A @ x - y)**2).mean().item()
        print(f"[PGD] iter {t+1}/{T} | meas MSE: {meas_loss:.6f}")

    return from_vec(x, C, H, W), z_prev

# ---------------------------
# 5) 入口：加载 G、构造/读取测量、运行重建
# ---------------------------
def load_target_from_dataset(dataset='mnist', idx=0, device='cuda'):
    tfms = T.Compose([T.Resize(128), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    root = './data'
    if dataset == 'mnist':
        ds = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=tfms)
    elif dataset == 'fashion-mnist':
        ds = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=tfms)
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion-mnist'")
    img, _ = ds[idx]
    return img.unsqueeze(0).to(device)  # (1,1,32,32), [-1,1]

def load_target_from_image(path, device='cuda'):
    img = Image.open(path).convert('L').resize((128,128))
    x = T.ToTensor()(img)  # [0,1]
    x = x * 2.0 - 1.0      # [-1,1]
    return x.unsqueeze(0).to(device)  # (1,1,32,32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_pkl', type=str, default=r'runs_dcgan128/ckpts/G_last.pt', help='path to trained generator.pkl')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','fashion-mnist','image'])
    parser.add_argument('--image_path', type=str, default='./image', help='if dataset=image, provide a grayscale image path')
    parser.add_argument('--index', type=int, default=3, help='sample index for test set')
    parser.add_argument('--m', type=int, default=1024, help='number of measurements')
    parser.add_argument('--eta', type=float, default=0.5, help='outer step size')
    parser.add_argument('--T', type=int, default=15, help='outer iterations')
    parser.add_argument('--inner_iters', type=int, default=200, help='projection inner GD steps')
    parser.add_argument('--inner_lr', type=float, default=0.0001, help='projection inner GD lr')
    parser.add_argument('--restarts', type=int, default=1, help='number of random restarts in projection')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str, default='./pgd_results')
    parser.add_argument('--A_type', type=str, default='from_npy',choices=['gaussian', 'inpainting', 'from_npy'],help='measurement operator type')
    parser.add_argument('--A_path', type=str, default='A_gaussian_m16384_n16384_seed0.npy',help='path to .npy file with (m,n) dense A if A_type=from_npy')
    parser.add_argument('--y_path', type=str, default='y_m16384_n16384.npy',help='path to a  image or .npy for  (H,) ')
    parser.add_argument('--save_A', action='store_true',help='save generated A to disk for reuse')
    parser.add_argument('--A_out', type=str, default='./pgd_results/A.npy',help='where to save A if --save_A')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu'
    C,H,W = 1,128,128
    n = C*H*W

    # 1) 加载训练好的生成器
    G = Generator(channels=C).to(device)
    state = torch.load(args.generator_pkl, map_location=device)
    G.load_state_dict(state)
    # G.load_state_dict(state['state_dict'])
    G.eval()
    print("Loaded generator from:", args.generator_pkl)

    # 2) 取一个目标图像 x* （在生成器范围内/附近），并生成测量 y = A x*
    if args.dataset == 'image':
        x_true = load_target_from_image(args.image_path, device=device)  # (1,1,32,32)
    else:
        x_true = load_target_from_dataset(args.dataset, args.index, device=device)
    x_true_vec = to_vec(x_true)

    A = get_A(args, n, H=H, W=W, device=device)
    print(A.shape)
    # # 保存 A 为图像
    # if args.A_type in ['inpainting', 'gaussian', 'from_npy']:
    #     os.makedirs(args.outdir, exist_ok=True)
    #
    #     # A 的形状是 (m, n)，n = C*H*W
    #     A_cpu = A.detach().cpu().numpy()
    #
    #     for i in range(min(10, A_cpu.shape[0])):  # 只保存前10行，避免太多
    #         mask = A_cpu[i].reshape(H, W)  # 把这一行展开为 H×W
    #         mask_img = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)  # 归一化到[0,1]
    #         plt.imsave(os.path.join(args.outdir, f"A_row{i + 1}.png"), mask_img, cmap="gray")
    #
    #     print(f"Saved first {min(10, A_cpu.shape[0])} rows of A as images in {args.outdir}")

    y = A @ x_true_vec  # (m,)
    # y_numpy = y.detach().cpu().numpy()
    # os.makedirs(args.outdir, exist_ok=True)
    # save_path = os.path.join(args.outdir, "y_result.npy")
    # np.save(save_path, y_numpy)
    # 3) 运行 PGD-GAN
    # y = torch.from_numpy(np.load(args.y_path)).to(device)
    x_rec, _ = pgd_gan(
        G, A, y,
        T=args.T, eta=args.eta,
        inner_iters=args.inner_iters, inner_lr=args.inner_lr,
        restarts=args.restarts, device=device, C=C, H=H, W=W,
        save_dir=os.path.join(args.outdir, "iters")
    )

    # 4) 保存可视化
    os.makedirs(args.outdir, exist_ok=True)
    out_img = os.path.join(args.outdir, f"recon_{args.dataset}_{args.index}_m{args.m}.png")
    save_vis(x_true, x_rec, out_img)
    print("Saved result to:", out_img)

    # 5) 计算指标
    with torch.no_grad():
        l2 = ((x_rec - x_true)**2).mean().item()
        meas_mse = ((A @ to_vec(x_rec) - y)**2).mean().item()
    print(f"L2 (per-pixel) = {l2:.6f} | Measurement MSE = {meas_mse:.6f}")

if __name__ == '__main__':
    main()
