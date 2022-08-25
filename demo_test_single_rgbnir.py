import os
import math
import torch
import numpy as np
from PIL import Image
from libtiff import TIFF
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from FGDNet import FGDNet


os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.backends.cudnn.benchmark = False

def padding_size(x, d):
    x = x + 2
    return math.ceil(x / d) * d - x


def add_noise(src, alpha, sigma):
    if not alpha == 0:
        src = alpha * np.random.poisson(src / alpha).astype(float)
    noise = np.random.normal(0, sigma, src.shape)
    src = src + noise
    src = np.clip(src, 0, 1.0)
    return src


def main():
    # Load Model
    model = FGDNet().cuda()
    model.load_state_dict(torch.load('models/FGDNet_rgbnir.pkl'))
    model.eval()
    lpfunc = lpips.LPIPS(net='vgg').cuda()

    # Load Image - RGB-NIR
    src = TIFF.open('examples/RGB_NIR/urban_0000_noisy.tiff', mode='r')
    noisy_img = src.read_image()
    src.close()
    noisy_img = np.array(noisy_img/255.0, dtype=float)

    src = TIFF.open('examples/RGB_NIR/urban_0000_nir.tiff', mode='r')
    guidance_img = src.read_image()
    src.close()
    guidance_img = np.array(guidance_img/255.0, dtype=float)

    src = TIFF.open('examples/RGB_NIR/urban_0000_gt.tiff', mode='r')
    gt_img = src.read_image()
    src.close()
    gt_img = np.array(gt_img/255.0, dtype=float)

    # Convert Images Into Tensors
    noisy_img = torch.from_numpy(noisy_img).permute(2, 0, 1).float().unsqueeze(0).cuda().contiguous()
    guidance_img = torch.from_numpy(guidance_img).float().unsqueeze(0).unsqueeze(0).cuda().contiguous()
    gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).float().unsqueeze(0).cuda().contiguous()

    # Conduct Image Padding
    h, w = noisy_img.shape[2], noisy_img.shape[3]
    h_psz = padding_size(h, 4)
    w_psz = padding_size(w, 4)
    padding = torch.nn.ReflectionPad2d((0, w_psz, 0, h_psz))
    noisy_img = padding(noisy_img)
    guidance_img = padding(guidance_img)

    # Start Denoising
    with torch.no_grad():
        denoised_r, _, _ = model(noisy_img[:,0,:,:,].unsqueeze(0), guidance_img)
        denoised_g, _, _ = model(noisy_img[:,1,:,:,].unsqueeze(0), guidance_img)
        denoised_b, _, _ = model(noisy_img[:,2,:,:,].unsqueeze(0), guidance_img)
    denoised_img = torch.cat([denoised_r, denoised_g, denoised_b], dim=1)
    denoised_img = torch.clamp(denoised_img, 0, 1.0)
    denoised_img = denoised_img[:,:,:h,:w]
    
    # Compute PSNR, SSIM, & LPIPS
    lpips_value = lpfunc(denoised_img, gt_img).item()

    denoised_img = denoised_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    gt_img = gt_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    psnr_value = psnr(denoised_img, gt_img)
    ssim_value = ssim(denoised_img, gt_img, multichannel=True)

    print('==========================================================================')
    print('PSNR={}, SSIM={}, LPIPS={}'.format(psnr_value, ssim_value, lpips_value))
    print('==========================================================================')

    # Save Images
    im = Image.fromarray(np.uint8(denoised_img*255))
    im.save('results/RGB_NIR/urban_0000_res.png')


if __name__ == '__main__':
    main()
