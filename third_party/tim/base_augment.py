import numpy as np
import scipy.signal
import torch
import albumentations as Augment

# ------- 基于albumentations的数据增强 ---------------
class BaseAugmentPipe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.get_augmentations()
    
    def get_augmentations(self, target_size):

        # 进行resize
        aug_resize      = Augment.Resize(height=target_size[0], width=target_size[1], p=1)
        
        # 进行水平垂直翻转
        aug_flip        = Augment.Compose([Augment.VerticalFlip(p=0.5), Augment.HorizontalFlip(p=0.5),])
        
        # 90度旋转
        aug_rotate90    = Augment.RandomRotate90(p=0.5)

        # 随机旋转
        aug_scaleRotate = Augment.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.10, rotate_limit=45, p=0.5)

        # 亮度/对比度拉升
        aug_brtContrast = Augment.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5)
        
        # 增加GaussNoise噪声
        #aug_noise       = Augment.GaussNoise(p=0.5)

        # RandomGravel
        #aug_gravel      = Augment.RandomGravel(p=0.5)

        # HueSaturationValue
        aug_hueSat      = Augment.HueSaturationValue(p=0.5)

        # RGBShift
        aug_rgbShift    = Augment.RGBShift(p=0.5)

        # MultiplicativeNoise
        aug_multiNoise  = Augment.MultiplicativeNoise(p=0.5)

        # FancyPCA
        aug_fancyPCA    = Augment.FancyPCA(p=0.5)

        # ColorJitter
        aug_colorJitter = Augment.ColorJitter(p=0.5)

        # Sharpen
        aug_sharpen     = Augment.Sharpen(p=0.5)

        # PixelDropout
        aug_pixelDrop   = Augment.PixelDropout(p=0.5)

        # 组合
        self.augment_fun = Augment.Compose([aug_resize, aug_flip, aug_rotate90, aug_scaleRotate, aug_brtContrast,
                                            aug_hueSat, aug_rgbShift, aug_multiNoise, aug_fancyPCA,
                                            aug_colorJitter, aug_sharpen, aug_pixelDrop])
        
    def forward(self, image, mask=None):
        if mask is None:
            processed = self.augment_fun(image = image)
            return processed['image']
        else:
            processed = self.augment_fun(image = image, mask=mask)
            return processed['image'], processed['mask']
