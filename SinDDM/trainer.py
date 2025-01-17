"""
the DDPM trainer was originally based on
https://github.com/lucidrains/denoising-diffusion-pytorch
"""

import copy
import os
import datetime
import albumentations

from functools import partial

from SinDDM.functions import *
from SinDDM.models import EMA
from third_party.tim.cutpaste import CutPasteTensor

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from matplotlib import pyplot as plt
from skimage.exposure import match_histograms
from text2live_util.util import get_augmentations_template
from tqdm import tqdm
import random
import cv2

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


# dataset classes

class sinDDMDataset(Dataset):
    def __init__(self, folder, image_size, blurry_img=False, train_num_steps = 10000, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder      = folder
        self.image_size  = image_size
        self.blurry_img  = blurry_img
        self.train_steps = train_num_steps
        
        self.get_file_paths(folder, exts)

        # 没有使用任何复杂的扩充方法
        self.transform = transforms.Compose([

            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def get_file_paths(self, folder, exts=['jpg', 'jpeg', 'png']):
        self.paths       = []
        self.paths_recon = []
        self.paths_mask  = []    
        
        image_files = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        for i in range(len(image_files)):
            cur_path = str(image_files[i])
            if cur_path.endswith('_mask.png'):
                self.paths_mask.append(image_files[i])
            elif cur_path.endswith('_recon.png'):
                self.paths_recon.append(image_files[i]) 
            else:
                self.paths.append(image_files[i])
    
    def __len__(self):
        return len(self.paths) * self.train_steps

    def __getitem__(self, index):
        
        # src image
        index     = index % len(self.paths)
        img       = Image.open(self.paths[index]).convert('RGB')
        trans_img = self.transform(img)

        # mask image 
        image_mask = Image.open(self.paths_mask[index]).convert('L')
        image_mask = np.array(image_mask)

        # 进行膨胀操作
        image_disk = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        image_mask = cv2.dilate(image_mask, image_disk)

        # convert 0-1 
        image_mask = image_mask.astype(np.float32) / 255.0

        # convert [H W]->[1, H, W]
        image_mask = torch.from_numpy(image_mask).unsqueeze(0)

        # blurry image
        if self.blurry_img:
            img_recon   = Image.open(self.paths_recon[index]).convert('RGB')
            trans_recon = self.transform(img_recon)
        else:
            trans_recon = trans_img.clone()
        
        return trans_img, trans_recon, image_mask


class MultiscaleTrainer(object):

    def __init__(
            self,
            ms_diffusion_model,
            folder,
            *,
            ema_decay=0.995,
            n_scales=None,
            scale_factor=1,
            image_sizes=None,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_sample_every=25000,
            avg_window=100,
            sched_milestones=None,
            results_folder='./results',
            device=None,
    ):
        super().__init__()
        self.device = device
        if sched_milestones is None:
            self.sched_milestones = [10000, 30000, 60000, 80000, 90000]
        else:
            self.sched_milestones = sched_milestones
        if image_sizes is None:
            image_sizes = []
        self.model     = ms_diffusion_model
        self.ema       = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema    = step_start_ema
        self.save_sample_every = save_sample_every
        self.avg_window        = avg_window

        self.batch_size   = train_batch_size
        self.n_scales     = n_scales
        self.scale_factor = scale_factor
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every
        
        self.input_paths = []
        self.ds_list     = []
        self.dl_list     = []
        self.data_list   = []
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        folder = os.path.join(folder, 'tmp/')

        for i in range(n_scales):
            self.input_paths.append(folder + 'scale_' + str(i))
            blurry_img = True if i > 0 else False
            self.ds_list.append(sinDDMDataset(self.input_paths[i], image_sizes[i], blurry_img, train_num_steps))
            self.dl_list.append(cycle(DataLoader(self.ds_list[i], batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=8)))
         
        self.opt       = Adam(ms_diffusion_model.parameters(), lr=train_lr)
        self.scheduler = MultiStepLR(self.opt, milestones=self.sched_milestones, gamma=0.5)

        self.step          = 0
        self.running_loss  = []
        self.running_scale = []
        self.avg_t         = []

        self.loss_fg       = AverageValueMeter()
        self.loss_bg       = AverageValueMeter()
        self.loss_image    = AverageValueMeter()

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.reset_parameters()
        self.get_image_augmentations(image_sizes[-1])
        self.get_tensor_augmentations()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'sched': self.scheduler.state_dict(),
            'running_loss': self.running_loss,
            'running_scale': self.running_scale
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        plt.rcParams['figure.figsize'] = [16, 8]

        plt.plot(self.running_loss)
        plt.grid(True)
        plt.ylim((0, 0.25))
        plt.savefig(str(self.results_folder / 'running_loss'))
        plt.clf()

    def load(self, milestone, mode='train'):
        model_path = str(self.results_folder / f'model-{milestone}.pt')
        if not os.path.exists(model_path):
            print(f'no model found at {model_path}')
            if mode in ['sample', 'sample_eval']:
                assert False, 'no model found, cannot sample'
            else:
                return
        else:
            print(f'loading model from {model_path}')      
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device)
            self.step = data['step']
            self.model.load_state_dict(data['model'])
            self.ema_model.load_state_dict(data['ema'])
            self.scheduler.load_state_dict(data['sched'])
            self.running_loss = data['running_loss']
        #    self.running_scale = data['running_scale']

    def get_image_augmentations(self, image_size):

        # 进行水平翻转
        #aug_flip = albumentations.Compose([albumentations.VerticalFlip(p=0.5), albumentations.HorizontalFlip(p=0.5),])
        
        # 90度旋转
        #aug_rotate90 = albumentations.RandomRotate90(p=0.5)

        # 随机旋转
        #aug_scaleRotate = albumentations.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.10, rotate_limit=45, p=0.5)

        # 亮度拉升
        #aug_brightness_contrast = albumentations.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5)

        # ColorJitter
        #aug_colorJitter = albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5)

        # HueSaturationValue
        #aug_hueSat      = albumentations.HueSaturationValue(p=0.5)

        # RGBShift
        #aug_rgbShift    = albumentations.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5)

        # crop
        dst_h, dst_w    = int(image_size[0] * 0.90), int(image_size[1] * 0.90)
        aug_crop        = albumentations.RandomCrop(height=dst_h, width=dst_w, p=1)
        
        # 组合
        self.augment_fun = albumentations.Compose([#aug_flip, aug_rotate90, 
                                                   #aug_brightness_contrast, aug_colorJitter,
                                                   aug_crop])
    
    def get_tensor_augmentations(self):
        self.cutpaste     = CutPasteTensor()
        self.base_augment = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), 
                                                transforms.RandomVerticalFlip(p=0.0),])
    
    def image_augment(self, batch_data):
        
        out_data    = []
        image_list1 = []
        image_list2 = []
        image_list3 = []

        # batch size
        for i in range(batch_data[0].shape[0]):
            
            # 转化为numpy  [-1, 1] -- > [0, 1]处理
            image1 = np.transpose(batch_data[0][i].numpy(), (1, 2, 0))
            image2 = np.transpose(batch_data[1][i].numpy(), (1, 2, 0))
            image3 = np.transpose(batch_data[2][i].numpy(), (1, 2, 0))
            image1 = (image1 + 1) * 0.5
            image2 = (image2 + 1) * 0.5
            
            # 进行数据增强
            self.get_image_augmentations(image1.shape[:2])
            image_concat = np.concatenate((image1, image2), axis=2)
            processed    = self.augment_fun(image = image_concat, mask = image3)
            
            # [0, 1] --> [-1, 1]
            image_concat_ed = (processed['image'] * 2) - 1
            image1 = image_concat_ed[:, :, :3]
            image2 = image_concat_ed[:, :, 3:]
            image1 = torch.from_numpy(np.transpose(image1, (2, 0, 1)))
            image2 = torch.from_numpy(np.transpose(image2, (2, 0, 1)))
            image3 = torch.from_numpy(np.transpose(processed['mask'], (2, 0, 1)))
            
            image_list1.append(image1)
            image_list2.append(image2)
            image_list3.append(image3)
        
        out_data.append(torch.stack(image_list1))
        out_data.append(torch.stack(image_list2))
        out_data.append(torch.stack(image_list3))

        return out_data
    
    def tensor_augment(self, batch_data):
        
        out_data    = []
        image_list1 = []
        image_list2 = []

        # batch size
        for i in range(batch_data[0].shape[0]):
            image1 = batch_data[0][i].detach().clone()
            image2 = batch_data[1][i].detach().clone()
            
            # 水平翻转
            if random.random() < 0.5 and 0:
                image1 = torch.flip(image1, dims=[1])
                image2 = torch.flip(image2, dims=[1])
            
            # 垂直翻转
            if random.random() < 0.5 and 0:
                image1 = torch.flipud(image1, dims=[2])
                image2 = torch.flipud(image2, dims=[2])

            # 90度旋转
            if random.random() < 0.5 and 0:
                image1 = torch.rot90(image1, 1, [1, 2])
                image2 = torch.rot90(image2, 1, [1, 2])
            
            # 270度旋转
            if random.random() < 0.5 and 0:
                image1 = torch.rot90(image1, 3, [1, 2])
                image2 = torch.rot90(image2, 3, [1, 2])
            
            # 亮度拉升
            if random.random() < 0.5 and 0:
                image1 = image1 + torch.randn(image1.shape).to(self.device) * 0.1
                image2 = image2 + torch.randn(image2.shape).to(self.device) * 0.1

            image_list1.append(image1)
            image_list2.append(image2)
        
        out_data.append(torch.stack(image_list1))
        out_data.append(torch.stack(image_list2))

        return out_data

    def train(self, augment=False):

        backwards   = partial(loss_backwards, self.fp16)
        loss_avg    = 0
        s_weights   = torch.tensor(self.model.num_timesteps_trained, device=self.device, dtype=torch.float)

        while self.step < self.train_num_steps:

            # t weighted multinomial sampling
            s = torch.multinomial(input=s_weights, num_samples=1)  # uniform when train_full_t = True
            for i in range(self.gradient_accumulate_every):

                input_data = next(self.dl_list[s])
                
                # data augment
                if s > 0:
                    input_data = self.image_augment(input_data) if augment else input_data

                # send to device
                for j in range(len(input_data)):
                    input_data[j] = input_data[j].to(self.device)

                # forward pass    
                loss, reuslts  = self.model(input_data, s)
                loss_avg += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)
                # 判定'loss_fg'是否在字典中
                if 'fg' in reuslts.keys():
                    self.loss_fg.update(reuslts['fg'], self.batch_size)
                if 'bg' in reuslts.keys():
                    self.loss_bg.update(reuslts['bg'], self.batch_size)
                if 'image' in reuslts.keys():
                    self.loss_image.update(reuslts['image'], self.batch_size)

            if self.step % self.avg_window == 0:
                print(f'step:{self.step} loss:{loss_avg/self.avg_window} lr:{self.scheduler.get_last_lr()[0]} fg:{self.loss_fg.avg} image:{self.loss_image.avg}')
                self.running_loss.append(loss_avg/self.avg_window)
                loss_avg = 0
                self.loss_fg.reset()
                self.loss_bg.reset()
                self.loss_image.reset()

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            self.scheduler.step()
            self.step += 1
            if self.step % self.save_sample_every == 0:
                milestone     = self.step // self.save_sample_every
                batches       = num_to_groups(16, self.batch_size)
                images_list   = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                milestone_str = str(milestone).zfill(6)     
                
                # ========================================================================
                all_images  = torch.cat(images_list, dim=0)
                all_images  = (all_images + 1) * 0.5
                utils.save_image(all_images, str(self.results_folder / f'sample-{milestone_str}.png'), nrow=4)
                
                # ========================================================================
                input_images= (input_data[0] + 1) * 0.5
                utils.save_image(input_images, str(self.results_folder / f'input-{milestone_str}.png'), nrow=8)
                
                # ========================================================================
                self.save(milestone)

        print('training completed')

    def sample_scales(self, scale_mul=None, batch_size=16, custom_sample=False, custom_image_size_idxs=None,
                      custom_scales=None, image_name='', start_noise= True, custom_t_list=None, desc=None, save_unbatched=True):
        if desc is None:
            desc = f'sample_{str(datetime.datetime.now()).replace(":", "_")}'
        if self.ema_model.reblurring:
            desc = desc + '_rblr'
        if self.ema_model.sample_limited_t:
            desc = desc + '_t_lmtd'
        # sample with custom t list
        if custom_t_list is None:
            custom_t_list = self.ema_model.num_timesteps_ideal[1:]
        # sample with custom scale list
        if custom_scales is None:
            custom_scales = [*range(self.n_scales)]
            n_scales = self.n_scales
        else:
            n_scales = len(custom_scales)
        if custom_image_size_idxs is None:
            custom_image_size_idxs = [*range(self.n_scales)]

        samples_from_scales      = []
        samples_from_scales_show = []
        final_results_folder     = Path(str(self.results_folder / 'final_samples'))
        final_results_folder.mkdir(parents=True, exist_ok=True)
        
        if scale_mul is not None:
            scale_0_size = (
                    int(self.model.image_sizes[custom_image_size_idxs[0]][0] * scale_mul[0]),
                    int(self.model.image_sizes[custom_image_size_idxs[0]][1] * scale_mul[1]))
        else:
            scale_0_size = None
        
        t_list         = [self.ema_model.num_timesteps_trained[0]] + custom_t_list
        res_sub_folder = '_'.join(str(e) for e in t_list)
        final_img      = None
        
        for i in range(n_scales):
            if start_noise and i == 0:
                cur_sample = self.ema_model.sample(batch_size=batch_size, scale_0_size=scale_0_size, s=custom_scales[i])
                samples_from_scales.append(cur_sample)
                samples_from_scales_show.append(cur_sample)
                
            elif i == 0: # start_noise == False, means injecting the original training image
                orig_sample_0 = Image.open((self.input_paths[custom_scales[i]] + '/' + image_name)).convert("RGB")
                samples_from_scales.append((transforms.ToTensor()(orig_sample_0) * 2 - 1).repeat(batch_size, 1, 1, 1).to(self.device))

            else:
                '''
                if i == 1 and 0:
                    aug_samples, _, _        = self.cutpaste(samples_from_scales[i-1])
                    #aug_samples              = self.base_augment(samples_from_scales[i-1])
                    samples_from_scales[i-1] = aug_samples.clone()
                    final_img                = (aug_samples + 1) * 0.5
                    utils.save_image(final_img, str(final_results_folder / res_sub_folder) + f'_out_s{i-1}_{desc}_sm_{scale_mul[0]}_{scale_mul[1]}_cutpaste.png', nrow=4)
                '''
                cur_sample = self.ema_model.sample_via_scale(batch_size,
                                                             samples_from_scales[i - 1],
                                                             s=custom_scales[i],
                                                             scale_mul=scale_mul,
                                                             custom_sample=custom_sample,
                                                             custom_img_size_idx=custom_image_size_idxs[i],
                                                             custom_t=custom_t_list[int(custom_scales[i])-1],
                                                             )
                samples_from_scales.append(cur_sample)
                samples_from_scales_show.append(cur_sample)
            final_img = (samples_from_scales_show[i] + 1) * 0.5
            utils.save_image(final_img, str(final_results_folder / res_sub_folder) + f'_out_s{i}_{desc}_sm_{scale_mul[0]}_{scale_mul[1]}.png', nrow=4)

        if save_unbatched:
            final_results_folder = Path(str(self.results_folder / f'final_samples_unbatched_{desc}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            for b in range(batch_size):
                utils.save_image(final_img[b], str(final_results_folder / res_sub_folder) + f'_out_b{b}.png')


    def bat_sample_scales(self, scale_mul=None, batch_size=16, custom_sample=False, 
                          custom_t_list=None, total_num=1000):

        # sample with custom t list
        if custom_t_list is None:
            custom_t_list = self.ema_model.num_timesteps_ideal[1:]
        
        # sample with custom scale list
        custom_scales = [*range(self.n_scales)]
        n_scales = self.n_scales
        custom_image_size_idxs = [*range(self.n_scales)]

        final_results_folder = Path(str(self.results_folder / 'sample_eval'))
        final_results_folder.mkdir(parents=True, exist_ok=True)
        
        if scale_mul is not None:
            scale_0_size = (
                    int(self.model.image_sizes[custom_image_size_idxs[0]][0] * scale_mul[0]),
                    int(self.model.image_sizes[custom_image_size_idxs[0]][1] * scale_mul[1]))
        else:
            scale_0_size = None
        
        for kk in range(int(total_num / batch_size) + 1):
            final_img                = None
            samples_from_scales      = []
            samples_from_scales_show = []   
            for i in range(n_scales):
                if i == 0:
                    cur_sample = self.ema_model.sample(batch_size=batch_size, scale_0_size=scale_0_size, s=custom_scales[i])
                    samples_from_scales.append(cur_sample)
                    samples_from_scales_show.append(cur_sample)
                else:
                    cur_sample = self.ema_model.sample_via_scale(batch_size,
                                                                samples_from_scales[i - 1],
                                                                s=custom_scales[i],
                                                                scale_mul=scale_mul,
                                                                custom_sample=custom_sample,
                                                                custom_img_size_idx=custom_image_size_idxs[i],
                                                                custom_t=custom_t_list[int(custom_scales[i])-1],
                                                                )
                    samples_from_scales.append(cur_sample)
                    samples_from_scales_show.append(cur_sample)
                final_img = (samples_from_scales_show[i] + 1) * 0.5

            for b in range(batch_size):
                index_num = kk * batch_size + b
                save_name = os.path.join(final_results_folder, f'fake_image_{str(index_num).zfill(5)}.png')
                utils.save_image(final_img[b], save_name)

    def image2image(self, input_folder='', input_file='', mask='', hist_ref_path='', image_name='', start_s=1, custom_t=None, batch_size=16, scale_mul=(1, 1), device=None, use_hist=False, save_unbatched=True, auto_scale=None, mode=None):
        if custom_t is None:
            custom_t = self.ema_model.num_timesteps_ideal # 0 - use default sampling t
        input_path = os.path.join(input_folder, input_file)
        input_img = Image.open(input_path).convert("RGB")

        image_size = input_img.size
        if auto_scale is not None:
            scaler = np.sqrt((image_size[0] * image_size[1]) / auto_scale)
            if scaler > 1:
                image_size = (int(image_size[0] / scaler), int(image_size[1] / scaler))
                input_img = input_img.resize(image_size, Image.LANCZOS)

        if mode == 'harmonization':
            mask_path = os.path.join(input_folder, mask)
            mask_img = Image.open(mask_path).convert("RGB")
            mask_img = mask_img.resize(image_size, Image.LANCZOS)
            mask_img = transforms.ToTensor()(mask_img)
            mask_img = dilate_mask(mask_img, mode=mode)
            mask_img = torch.from_numpy(mask_img).to(self.device)
        else:
            mask_img = 1

        if use_hist:
            image_name = image_name.rsplit(".", 1)[0] + '.png'
            orig_sample_0 = Image.open((hist_ref_path + image_name)).convert("RGB")  # next(self.dl_list[0])
            input_img_ds_matched_arr = match_histograms(image=np.array(input_img), reference=np.array(orig_sample_0), channel_axis=2)
            input_img = Image.fromarray(input_img_ds_matched_arr)

        input_img_tensor = (transforms.ToTensor()(input_img) * 2 - 1)  # normalize
        input_size = torch.tensor(input_img_tensor.shape[1:])
        input_img_batch = input_img_tensor.repeat(batch_size, 1, 1, 1).to(device)  # batchify and send to GPU

        final_results_folder = Path(str(self.results_folder / 'i2i_final_samples'))
        final_results_folder.mkdir(parents=True, exist_ok=True)
        final_img = None
        t_string = '_'.join(str(e) for e in custom_t)
        time = str(datetime.datetime.now()).replace(":", "_")

        if start_s > 0:  # starting scale has no mixing between blurry and clean images
            self.ema_model.gammas[start_s-1].clamp_(0, 0)
        samples_from_scales = []
        for i in range(self.n_scales-start_s):
            s = i + start_s
            ds_factor = self.scale_factor ** (self.n_scales - s - 1)
            cur_size = input_size/ds_factor
            cur_size = (int(cur_size[0].item()), int(cur_size[1].item()))

            if i == 0:
                samples_from_scales.append(self.ema_model.sample_via_scale(batch_size,
                                                                           input_img_batch,
                                                                           s=s,
                                                                           custom_t=custom_t[s],
                                                                           scale_mul=scale_mul,
                                                                           custom_image_size=cur_size), )
            else:
                samples_from_scales.append(self.ema_model.sample_via_scale(batch_size,
                                                                           samples_from_scales[i - 1],
                                                                           s=s,
                                                                           custom_t=custom_t[s],
                                                                           scale_mul=scale_mul,
                                                                           custom_image_size=cur_size),)
            final_img = (samples_from_scales[i] + 1) * 0.5
            input_file_name = input_file.rsplit(".", 1)[0]
            if i == self.n_scales-start_s - 1:
                input_img_batch_denorm = (input_img_batch + 1) * 0.5
                input_img_batch_denorm.clamp_(0.0, 1.0)
                final_img = mask_img * final_img + (1 - mask_img) * input_img_batch_denorm

            utils.save_image(final_img, str(final_results_folder / f'{input_file_name}_i2i_s_{start_s+i}_t_{t_string}_hist_{"on" if use_hist else "off"}_{time}.png'), nrow=4)
        if save_unbatched:
            final_results_folder = Path(str(self.results_folder / f'unbatched_i2i_s{start_s}_t_{t_string}_{time}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            for b in range(batch_size):
                utils.save_image(final_img[b], os.path.join(final_results_folder ,input_file + f'_out_b{b}_i2i.png'))

    def clip_sampling(self, clip_model, text_input, strength, sample_batch_size, custom_t_list=None,
                      guidance_sub_iters=None, quantile=0.8, stop_guidance=None, save_unbatched=False, scale_mul=(1,1), llambda=0, start_noise=True, image_name=''):
        if guidance_sub_iters is None:
            guidance_sub_iters = [*reversed(range(self.n_scales))]
        
        self.ema_model.clip_strength   = strength
        self.ema_model.clip_text       = text_input
        self.ema_model.text_embedds_hr = clip_model.get_text_embedding(text_input, template=get_augmentations_template('hr'))
        self.ema_model.text_embedds_lr = clip_model.get_text_embedding(text_input, template=get_augmentations_template('lr'))
        self.ema_model.clip_guided_sampling = True
        self.ema_model.guidance_sub_iters = guidance_sub_iters
        self.ema_model.quantile      = quantile
        self.ema_model.stop_guidance = stop_guidance
        self.ema_model.clip_model    = clip_model
        self.ema_model.clip_score    = []
        self.ema_model.llambda       = llambda
        strength_string = f'{strength}'
        gsi_string      = '_'.join(str(e) for e in guidance_sub_iters)
        n_aug           = self.ema_model.clip_model.cfg["n_aug"]
        desc = f"clip_{text_input.replace(' ', '_')}_n_aug{n_aug}_str_" + strength_string + "_gsi_" + gsi_string + \
               f'_ff{1-quantile}' + f'_{str(datetime.datetime.now()).replace(":", "_")}'

        if not start_noise:  # relevant for mode==clip_style_trans
            # start from last scale
            custom_scales          = [self.n_scales - 2, self.n_scales - 1]
            custom_image_size_idxs = [self.n_scales - 2, self.n_scales - 1]
            # custom_t_list = [self.ema_model.num_timesteps_ideal[-2], self.ema_model.num_timesteps_ideal[-1]]
            self.sample_scales(scale_mul=scale_mul,  # H,W
                               custom_sample=True,
                               custom_scales=custom_scales,
                               custom_image_size_idxs=custom_image_size_idxs,
                               image_name=image_name,
                               batch_size=sample_batch_size,
                               custom_t_list=custom_t_list,
                               desc=desc,
                               save_unbatched=save_unbatched,
                               start_noise=start_noise,
                               )
        else:  # relevant for mode==clip_style_gen or clip_content
            self.sample_scales(scale_mul=scale_mul,  # H,W
                               custom_sample=False,
                               image_name='',
                               batch_size=sample_batch_size,
                               custom_t_list=custom_t_list,
                               desc=desc,
                               save_unbatched=save_unbatched,
                               start_noise=start_noise,
                               )
        self.ema_model.clip_guided_sampling = False

    def clip_roi_sampling(self, clip_model, text_input, strength, sample_batch_size,
                          num_clip_iters=100, num_denoising_steps=2, clip_roi_bb=None, save_unbatched=False):

        text_embedds = clip_model.get_text_embedding(text_input, template=get_augmentations_template('lr'))
        strength_string = f'{strength}'
        n_aug = clip_model.cfg["n_aug"]
        desc = f"clip_roi_{text_input.replace(' ', '_')}_n_aug{n_aug}_str_" + strength_string + "_n_iters_" + str(num_clip_iters) + \
               f'_{str(datetime.datetime.now()).replace(":", "_")}'
        image = self.data_list[self.n_scales-1][0][0][None,:,:,:]
        image = image.repeat(sample_batch_size, 1, 1, 1)

        image_roi = image[:,:,clip_roi_bb[0]:clip_roi_bb[0]+clip_roi_bb[2],clip_roi_bb[1]:clip_roi_bb[1]+clip_roi_bb[3]].clone()

        image_roi.requires_grad_(True)
        image_roi_renorm = (image_roi + 1) * 0.5
        interm_results_folder = Path(str(self.ema_model.results_folder / f'interm_samples_clip_roi'))
        interm_results_folder.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(num_clip_iters)):
            clip_model.zero_grad()
            score = -clip_model.calculate_clip_loss(image_roi_renorm, text_embedds)
            clip_grad = torch.autograd.grad(score, image_roi, create_graph=False)[0]
            if self.ema_model.save_interm:
                utils.save_image((image_roi.clamp(-1., 1.) + 1) * 0.5,
                                 str(interm_results_folder / f'iter_{i}.png'),
                                 nrow=4)

            image_roi_prev_norm = torch.linalg.vector_norm(image_roi, dim=(1, 2, 3), keepdim=True)
            division_norm = torch.linalg.vector_norm(image_roi, dim=(1,2,3), keepdim=True) / torch.linalg.vector_norm(
                clip_grad, dim=(1,2,3), keepdim=True)
            image_roi_prev = image_roi
            image_roi = image_roi_prev + strength* division_norm * clip_grad
            image_roi_norm = torch.linalg.vector_norm(image_roi, dim=(1, 2, 3), keepdim=True)
            keep_norm = False
            if keep_norm:
                image_roi *= (image_roi_prev_norm) / (image_roi_norm)

            image_roi.clamp_(-1., 1.)
            image_roi_renorm = (image_roi + 1) * 0.5

        # insert patch into original image
        image[:, :, clip_roi_bb[0]:clip_roi_bb[0] + clip_roi_bb[2],clip_roi_bb[1]:clip_roi_bb[1] + clip_roi_bb[3]] = image_roi
        #
        final_image = self.ema_model.sample_via_scale(sample_batch_size,
                                                      image,
                                                      s=self.n_scales-1,
                                                      custom_t=num_denoising_steps,
                                                      scale_mul=(1,1))
        final_img_renorm = (final_image + 1) * 0.5
        final_results_folder = Path(str(self.ema_model.results_folder / f'final_samples'))
        final_results_folder.mkdir(parents=True, exist_ok=True)
        utils.save_image(final_img_renorm, str(final_results_folder / (desc + '.png')), nrow=4)

        if save_unbatched:
            final_results_folder = Path(str(self.results_folder / f'final_samples_unbatched_{desc}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            for b in range(sample_batch_size):
                utils.save_image(final_img_renorm[b], os.path.join(final_results_folder, f'{desc}_out_b{b}.png'))

    def roi_guided_sampling(self, custom_t_list=None, target_roi=None, roi_bb_list=None, save_unbatched=False, batch_size=4 ,scale_mul=(1, 1)):
        self.ema_model.roi_guided_sampling = True
        self.ema_model.roi_bbs = roi_bb_list
        target_bb = target_roi
        # create a corresponding downsampled patch for each scale
        for scale in range(self.n_scales):
            target_bb_rescaled = [int(bb_i / np.power(self.scale_factor, self.n_scales - scale - 1)) for bb_i in target_bb]
            self.ema_model.roi_target_patch.append(extract_patch(self.data_list[scale][0][0][None, :,:,:], target_bb_rescaled))

        self.sample_scales(scale_mul=scale_mul,  # H,W
                           custom_sample=False,
                           image_name='',
                           batch_size=batch_size,
                           custom_t_list=custom_t_list,
                           desc=f'roi_{str(datetime.datetime.now()).replace(":", "_")}',
                           save_unbatched=save_unbatched,
                           start_noise=True,
                           )
        self.ema_model.roi_guided_sampling = False
