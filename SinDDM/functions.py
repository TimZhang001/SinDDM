
import torch
from skimage import morphology, filters
from inspect import isfunction
import numpy as np
from PIL import Image
from pathlib import Path
import glob
import os
import shutil
import cv2

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False




def dilate_mask(mask, mode):
    if mode == "harmonization":
        element = morphology.disk(radius=7)
    if mode == "editing":
        element = morphology.disk(radius=20)
    mask = mask.permute((1, 2, 0))
    mask = mask[:, :, 0]
    mask = morphology.binary_dilation(mask, selem=element)
    mask = filters.gaussian(mask, sigma=5)
    mask = mask[:, :, None, None]
    mask = mask.transpose(3, 2, 0, 1)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


# for roi_sampling

def stat_from_bbs(image, bb):
    y_bb, x_bb, h_bb, w_bb = bb
    bb_mean = torch.mean(image[:, :,y_bb:y_bb+h_bb, x_bb:x_bb+w_bb], dim=(2,3), keepdim=True)
    bb_std = torch.std(image[:, :, y_bb:y_bb+h_bb, x_bb:x_bb+w_bb], dim=(2,3), keepdim=True)
    return [bb_mean, bb_std]


def extract_patch(image, bb):
    y_bb, x_bb, h_bb, w_bb = bb
    image_patch = image[:, :,y_bb:y_bb+h_bb, x_bb:x_bb+w_bb]
    return image_patch


# for clip sampling
def thresholded_grad(grad, quantile=0.8):
    """
    Receives the calculated CLIP gradients and outputs the soft-tresholded gradients based on the given quantization.
    Also outputs the mask that corresponds to remaining gradients positions.
    """
    grad_energy = torch.norm(grad, dim=1)
    grad_energy_reshape = torch.reshape(grad_energy, (grad_energy.shape[0],-1))
    enery_quant = torch.quantile(grad_energy_reshape, q=quantile, dim=1, interpolation='nearest')[:,None,None] #[batch ,1 ,1]
    gead_energy_minus_energy_quant = grad_energy - enery_quant
    grad_mask = (gead_energy_minus_energy_quant > 0)[:,None,:,:]

    gead_energy_minus_energy_quant_clamp = torch.clamp(gead_energy_minus_energy_quant, min=0)[:,None,:,:]#[b,1,h,w]
    unit_grad_energy = grad / grad_energy[:,None,:,:] #[b,c,h,w]
    unit_grad_energy[torch.isnan(unit_grad_energy)] = 0
    sparse_grad = gead_energy_minus_energy_quant_clamp * unit_grad_energy #[b,c,h,w]
    return sparse_grad, grad_mask

# helper functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def create_img_scales(foldername, filename, scale_factor=1.411, image_size=None, create=False, auto_scale=None,
                      area_scale_0=3110, scale_0_dim_min=42, scale_0_dim_max=55):
    """
    Receives path to the desired training image and scale_factor that defines the downsampling rate.
    optional argument image_size can be given to reshape the original training image.
    optional argument auto_scale - limits the training image to have a given #pixels.
    The function creates the downsampled and upsampled blurry versions of the training image.
    Calculates n_scales such that RF area is ~40% of the smallest scale area with the given scale_factor.
    Also calculates the MSE loss between upsampled/downsampled images for starting T calculation (see paper).


    returns:
            sizes: list of image sizes for each scale
            rescale_losses: list of MSE losses between US/DS images for each scale
            scale_factor: modified scale_factor to allow 40% area ratio
    """
    
    if filename is None:
        image_files = [os.path.basename(x) for x in glob.glob(foldername + '*.png')]
    else:
        image_files = ['/' +filename]    
    
    # 创建临时文件夹，删除文件夹里面的所有文件和文件夹
    save_foldername     = os.path.join(foldername, 'tmp/')
    Path(save_foldername).mkdir(parents=True, exist_ok=True)
    for file in os.listdir(save_foldername):
        file_path = os.path.join(save_foldername, file)
        if os.path.isfile(file_path) and create:
            os.remove(file_path)
        elif os.path.isdir(file_path) and create:
            shutil.rmtree(file_path)

    sizes               = []
    rescale_losses_list = []
    for filename in image_files:
    
        # image_name and mask name
        image_name = foldername + filename
        mask_name  = image_name.replace('/test/', '/ground_truth/').replace('.png', '_mask.png')
        
        # load image
        orig_image = Image.open(image_name).convert('RGB')
        mask_image = Image.open(mask_name).convert('L')

        # 默认分辨率为image_size
        orig_image = orig_image.resize(image_size, Image.LANCZOS)
        mask_image = mask_image.resize(image_size, Image.LANCZOS)

        # convert to PNG extension for lossless conversion
        filename = filename.rsplit( ".", 1 )[ 0 ] + '.png'
        if image_size is None:
            image_size = (orig_image.size)
        if auto_scale is not None:
            scaler = np.sqrt((image_size[0] * image_size[1])/auto_scale)
            if scaler > 1:
                image_size = (int(image_size[0]/scaler), int(image_size[1]/scaler))
        sizes             = []
        downscaled_images = []
        recon_images      = []
        rescale_losses    = []

        # auto resize
        # rf_net = 35
        #area_scale_0 = 4096  # defined such that rf_net^2/area_scale0 ~= 70% 感受野面积占原图的70%
        s_dim        = min(image_size[0], image_size[1])
        l_dim        = max(image_size[0], image_size[1])
        scale_0_dim  = int(round(np.sqrt(area_scale_0*s_dim/l_dim)))
        
        # clamp between 42 and 55
        scale_0_dim   = scale_0_dim_min if scale_0_dim < scale_0_dim_min else (scale_0_dim_max if scale_0_dim > scale_0_dim_max else scale_0_dim )
        small_val     = scale_0_dim
        min_val_image = min(image_size[0], image_size[1])
        n_scales      = int(round( (np.log(min_val_image/small_val)) / (np.log(scale_factor)) ) + 1)
        scale_factor  = np.exp((np.log(min_val_image / small_val)) / (n_scales - 1))

        for i in range(n_scales):
            cur_size = (int(round(image_size[0] / np.power(scale_factor, n_scales - i - 1))), int(round(image_size[1] / np.power(scale_factor, n_scales - i - 1))))
            cur_img  = orig_image.resize(cur_size, Image.LANCZOS)
            cur_mask = mask_image.resize(cur_size, Image.NEAREST)
                        
            # dilate mask
            image_mask = np.array(cur_mask)
            image_disk = cv2.getStructuringElement(cv2.MORPH_RECT, (int(3*n_scales), int(3*n_scales)))
            image_mask = cv2.dilate(image_mask, image_disk)
            cur_mask   = Image.fromarray(image_mask)

            # convert to binary mask
            cur_mask = cur_mask.point(lambda x: 0 if x < 255 else 255, '1')

            path_to_save = save_foldername + 'scale_' + str(i) + '/'
            if create:
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                cur_img.save(path_to_save + filename)
                cur_mask.save(path_to_save + filename.replace('.png', '_mask.png'))
            downscaled_images.append(cur_img)
            sizes.append(cur_size)
        for i in range(n_scales - 1):
            recon_image = downscaled_images[i].resize(sizes[i + 1], Image.BILINEAR)
            recon_images.append(recon_image)
            rescale_losses.append(np.linalg.norm(np.subtract(downscaled_images[i + 1], recon_image)) / np.asarray(recon_image).size)
            if create:
                path_to_save = save_foldername + 'scale_' + str(i + 1)
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                recon_image.save(path_to_save + filename.replace('.png', '_recon.png'))

        rescale_losses_list.append(rescale_losses)
    
    rescale_losses = np.mean(np.asarray(rescale_losses_list), axis=0)

    return sizes, rescale_losses, scale_factor, n_scales



class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        if val > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count