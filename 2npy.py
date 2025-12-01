# spi_mnist_dgi.py
import os
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image


def save_gray(img2d, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, img2d, cmap='gray')


def minmax01(x, eps=1e-8):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def load_mnist_image(a=128, index=0, invert=False):
    """
    取一张MNIST测试集图片，缩放到 a×a，返回归一化到[0,1]的numpy二维数组 (a, a)
    invert=True 时把黑底白字 → 白底黑字（可视化友好）
    """
    tfm = T.Compose([
        T.Resize(a),
        T.ToTensor(),        # [0,1], shape: (1,H,W)
    ])
    ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    img, _ = ds[index]             # img: torch.Tensor (1, a, a), [0,1]
    img = img[0].numpy()           # (a, a)
    img = img * 2.0 - 1.0
    return img


def make_A_gaussian(m, n, seed=0):
    """
    生成随机高斯测量矩阵 A ∈ R^{m×n}
    常用做法是 N(0, 1/sqrt(m))，能让每个测量的幅度更稳定
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(loc=0.0, scale=1.0/np.sqrt(m), size=(m, n)).astype(np.float32)
    return A


def measure(A, x_vec):
    """
    单像素测量：y = A @ x ，A:(m,n), x:(n,)
    """
    return (A @ x_vec).astype(np.float32)


def dgi_reconstruct(y, A, a):
    """
    DGI（差分鬼成像）简化版重建：
        recon = sum_i ( (y_i - mean(y)) * pattern_i )
    其中 pattern_i 是 A 的第 i 行 reshape 为 (a,a)
    注意：A 若是零均值高斯，这个式子本质上 ~ A^T y（差个均值和比例因子），
         但这里按DGI直观公式写出来，更贴合鬼成像描述。
    """
    m, n = A.shape
    assert n == a*a
    y_mean = float(y.mean())
    yy = (y - y_mean).reshape(m, 1)          # (m,1)
    # recon_vec = sum_i ( (y_i - mean(y)) * A[i,:] )
    recon_vec = (yy * A).sum(axis=0)         # (n,)
    recon = recon_vec.reshape(a, a)
    # 可选归一化到[0,1]方便显示
    recon = minmax01(recon)
    return recon
import numpy as np

def add_noise(y, noise_type="gaussian", noise_level=0.05, snr_db=0):
    """
    给测量向量 y 添加噪声
    noise_type: 'gaussian' | 'uniform' | 'saltpepper' | 'poisson' | 'mixed'
    noise_level: 噪声强度 (高斯/均匀的标准差或幅度, 椒盐的概率)
    snr_db: 高斯噪声信噪比 (dB)
    """

    if noise_type == "gaussian":
        # 信号功率
        P_signal = np.mean(y**2)
        P_noise = P_signal / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(P_noise), size=y.shape).astype(np.float32)
        return y + noise

    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_level, noise_level, size=y.shape).astype(np.float32)
        return y + noise

    elif noise_type == "saltpepper":
        noisy = y.copy()
        rand = np.random.rand(*y.shape)
        noisy[rand < noise_level/2] = -1.0
        noisy[rand > 1 - noise_level/2] = 1.0
        return noisy

    elif noise_type == "poisson":
        y_min = y.min()
        if y_min < 0:
            y_shift = y - y_min
        else:
            y_shift = y
        noisy = np.random.poisson(y_shift * 255) / 255.0
        return noisy.astype(np.float32)

    elif noise_type == "mixed":
        # 高斯 + 椒盐
        y_gauss = add_noise(y, noise_type="gaussian", snr_db=snr_db)
        y_mixed = add_noise(y_gauss, noise_type="saltpepper", noise_level=noise_level)
        return y_mixed

    else:
        raise ValueError(f"未知噪声类型: {noise_type}")


def load_custom_image(image_path, target_size=128):
    """加载自定义图片并预处理"""
    # 读取图片并转为灰度
    img = Image.open(image_path).convert('L')

    # 调整大小和归一化
    transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])  # [-1,1]范围
    ])

    img_tensor = transform(img)
    return img_tensor[0].numpy()  # 转为numpy (a,a)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_image', type=str, default='reconstructed_image1.png', help='使用自定义图片路径（优先级高于MNIST）')
    parser.add_argument('--a', type=int, default=128, help='图像边长，最终尺寸 a×a')
    parser.add_argument('--index', type=int, default=2, help='MNIST 测试集第几张（0开始）')
    parser.add_argument('--sampling_rate', type=float, default=0.1, help='采样率 m/n，比如 0.1')
    parser.add_argument('--m', type=int, default=-1, help='观测数 m（优先级高于 sampling_rate；-1表示按采样率计算）')
    parser.add_argument('--seed', type=int, default=0, help='随机种子（控制 A 的复现性）')
    parser.add_argument('--outdir', type=str, default='./spi_results', help='输出目录')
    parser.add_argument('--save_patterns', type=int, default=0, help='保存前多少个采样pattern为图片；0表示不保存')
    parser.add_argument('--invert', action='store_true', help='把MNIST黑底白字转成白底黑字以便显示')
    args = parser.parse_args()

    a = args.a
    n = a * a

    # 1) 取一张 MNIST 图，保存
    if args.custom_image:
        if not os.path.exists(args.custom_image):
            raise FileNotFoundError(f"自定义图片不存在: {args.custom_image}")
        img = load_custom_image(args.custom_image, a)
        img_name = os.path.splitext(os.path.basename(args.custom_image))[0]
    else:
        img = load_mnist_image(a=a, index=args.index, invert=args.invert)
        img_name = f'mnist_idx{args.index}'# (a,a), [0,1]
    os.makedirs(args.outdir, exist_ok=True)
    save_gray(img, os.path.join(args.outdir, f'mnist_a{a}_idx{args.index}.png'))

    # 2) 生成随机高斯矩阵 A，保存为 .npy
    if args.m > 0:
        m = int(args.m)
    else:
        m = max(1, int(round(args.sampling_rate * n)))
    A = make_A_gaussian(m, n, seed=args.seed)
    # mask_path = r'utils\mask.mat'
    # with h5py.File(mask_path, 'r') as file:
    #     A = np.array(file['mask'])/128
    np.save(os.path.join(args.outdir, f'A1_gaussian_m{m}_n{n}_seed{args.seed}.npy'), A)
    # 3) 单像素测量：y = A @ x（x是向量）
    x_vec = img.flatten().astype(np.float32)  # 使用 flatten()
    # (n,), 使用[0,1]强度
    y = measure(A, x_vec)
    # y = add_noise(y)
    # y = y / (np.max(np.abs(y)) + 1e-8)
    np.save(os.path.join(args.outdir, f'y1_m{m}_n{n}.npy'), y)

    # 4) DGI 简单重建，保存结果
    recon = dgi_reconstruct(y, A, a)                # (a,a), [0,1]
    save_gray(recon, os.path.join(args.outdir, f'recon_DGI_m{m}_n{n}.png'))

    # 5) 可视化若干个采样 pattern（A 的行）为灰度图，便于检查
    k = min(args.save_patterns, m)
    for i in range(k):
        patt = A[i].reshape(a, a)
        patt_viz = minmax01(patt)                   # 归一化显示
        save_gray(patt_viz, os.path.join(args.outdir, f'{i+1:02d}.png'))

    # 打印信息
    print(f'Image saved to: {os.path.join(args.outdir, f"mnist_a{a}_idx{args.index}.png")}')
    print(f'A saved to:     {os.path.join(args.outdir, f"A_gaussian_m{m}_n{n}_seed{args.seed}.npy")}    shape=({m},{n})')
    print(f'y saved to:     {os.path.join(args.outdir, f"y_m{m}_n{n}.npy")}                         shape=({m},)')
    print(f'DGI recon:      {os.path.join(args.outdir, f"recon_DGI_m{m}_n{n}.png")}')
    if k > 0:
        print(f'Patterns (first {k}) saved as PNG in {args.outdir}')


if __name__ == '__main__':
    main()
