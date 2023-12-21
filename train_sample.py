import torch
import numpy as np
import argparse
import os
import torchvision
from SinDDM.functions import create_img_scales
from SinDDM.models import MultiScaleGaussianDiffusion, SinDDMNet, NextNet, Unet, DiffusionBiSeNet
from SinDDM.trainer import MultiscaleTrainer
from text2live_util.clip_extractor import ClipExtractor
import time

mvtectAD = "./datasets/mvtec/"
mvtectTexture = ["grid", "carpet", "leather", "tile", "wood"]
mvtectObject1 = ["hazelnut", "bottle", "cable", "capsule",  ] # , 
mvtectObject2 = ["screw", "metal_nut", "pill", "toothbrush", ] #  
mvtectObject3 = ["transistor", "zipper"] #
mvtectAll     = mvtectTexture + mvtectObject1 + mvtectObject2 + mvtectObject3  


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scope",   help='choose training scope.',      default='single_image')
    parser.add_argument("--mode",    help='choose mode: train, sample,', default='train') # train  sample
    parser.add_argument("--augment", help='use data augmentation.',      default=0)
    
    # --------------------------------------------------------------------------------------------------

    # Dataset
    parser.add_argument("--dataset_folder", help='choose dataset folder.', default='./datasets/mvtec/cable/cut_outer_insulation/')
    parser.add_argument("--image_name",     help='choose image name.',     default=None)
    parser.add_argument("--results_folder", help='choose results folder.', default='./results/')
    parser.add_argument("--image_size",     help='choose image size.',     default=[256, 256])

    # Net
    parser.add_argument("--dim", help='widest channel dimension for conv blocks.', default=160, type=int)
    # diffusion params
    parser.add_argument("--scale_factor", help='downscaling step for each scale.', default=1.411, type=float) # 1.411 1.732# 
    # training params
    parser.add_argument("--timesteps",        help='total diffusion timesteps.',  default=100, type=int)
    parser.add_argument("--train_batch_size", help='batch size during training.', default=32,  type=int)
    parser.add_argument("--grad_accumulate",  help='gradient accumulation (bigger batches).', default=1, type=int)
    parser.add_argument("--train_num_steps",  help='total training steps.',       default=120001, type=int)  # 120001
    parser.add_argument("--save_sample_every",help='n. steps for checkpointing model.', default=1000, type=int) # 10000
    parser.add_argument("--avg_window",       help='window size for averaging loss (visualization only).', default=100, type=int)
    parser.add_argument("--train_lr",         help='starting lr.', default=1e-3, type=float)
    parser.add_argument("--sched_k_milestones", nargs="+", help='lr scheduler steps x 1000.', default=[20, 40, 70, 80, 90, 110], type=int)
    parser.add_argument("--load_milestone",   help='load specific milestone.', default=17, type=int)
    
    # sampling params
    parser.add_argument("--sample_batch_size", help='batch size during sampling.', default=16, type=int)
    parser.add_argument("--scale_mul",         help='image size retargeting modifier.', nargs="+", default=[1, 1], type=float)
    parser.add_argument("--sample_t_list",     nargs="+", help='Custom list of timesteps corresponding to each scale (except scale 0).', type=int)
    
    # device num
    parser.add_argument("--device_num",        help='use specific cuda device.', default=4, type=int)

    # DEV. params - do not modify
    parser.add_argument("--sample_limited_t", help='limit t in each scale to stop at the start of the next scale', action='store_true')
    parser.add_argument("--omega",            help='sigma=omega*max_sigma.', default=0, type=float)
    parser.add_argument("--loss_factor",      help='ratio between MSE loss and starting diffusion step for each scale.', default=1, type=float)

    args = parser.parse_args()
    args.train_num_steps    = int(args.train_num_steps * 5 / args.train_batch_size) 
    args.sched_k_milestones = [int(val * 5 / args.train_batch_size) for val in args.sched_k_milestones]

    return args


def print_save_params(args):
    
    # 对args的参数进行打印
    print('-----------------args-------------------')
    for k, v in vars(args).items():
        print(k, ':', v)
    print('----------------------------------------')


    # args的参数保存到文件中
    # 获取dataset_folder的最后两个文件夹
    folder_list = args.dataset_folder.rstrip('/')
    folder_list = folder_list.split('/')
    save_path   = os.path.join(args.results_folder, folder_list[-2], folder_list[-1], args.scope)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(str(k) + ':' + str(v) + '\n')
    
    return save_path

# ------------------------------------------------------------------------- #
def train_sample_models(args, save_path):
    device           = f"cuda:{args.device_num}"
    scale_mul        = (args.scale_mul[0], args.scale_mul[1])
    sched_milestones = [val * 1000 for val in args.sched_k_milestones] # 1000
    
    # set to true to save all intermediate diffusion timestep results
    save_interm    = True if args.mode == 'sample' else False
    save_unbatched = False
    channels       = 3
    create         = False if args.mode == 'sample' else True

    sizes, rescale_losses, scale_factor, n_scales = create_img_scales(args.dataset_folder, args.image_name,
                                                                      scale_factor=args.scale_factor,
                                                                      image_size=args.image_size,
                                                                      create=create,
                                                                      auto_scale=100000, # limit max number of pixels in image
                                                                      )

    model = SinDDMNet(dim=args.dim, multiscale=True, device=device, channels=channels,)    
    #model = Unet(dim=int(args.dim/8), channels=channels, dim_mults=(1, 2, 4, 4), device=device)
    #model = NextNet(in_channels= channels, out_channels=channels, depth=16, filters_per_layer=64, device=device)
    #model = DiffusionBiSeNet(in_channels=channels, detail_depth=16, dim=64, semantic_mults=(1,2,2), device=device)
    model.to(device)

    ms_diffusion = MultiScaleGaussianDiffusion(
        denoise_fn=model,
        save_interm=save_interm,
        results_folder=save_path, # for debug
        n_scales=n_scales,
        scale_factor=scale_factor,
        image_sizes=sizes,
        scale_mul=scale_mul,
        channels=channels,
        timesteps=args.timesteps,
        train_full_t=True,
        scale_losses=rescale_losses,
        loss_factor=args.loss_factor,
        loss_type='l1',
        betas=None,
        device=device,
        reblurring=True,
        sample_limited_t=args.sample_limited_t,
        omega=args.omega,
    ).to(device)

    if args.sample_t_list is None:
        sample_t_list = ms_diffusion.num_timesteps_ideal[1:]
    else:
        sample_t_list = args.sample_t_list

    print('-----------------sample_t_list is {}-------------------'.format(sample_t_list))

    ScaleTrainer = MultiscaleTrainer(
            ms_diffusion,
            folder=args.dataset_folder,
            n_scales=n_scales,
            scale_factor=scale_factor,
            image_sizes=sizes,
            train_batch_size=args.train_batch_size,
            train_lr=args.train_lr,
            train_num_steps=args.train_num_steps,  # total training steps
            gradient_accumulate_every=args.grad_accumulate,  # gradient accumulation steps
            ema_decay=0.995,  # exponential moving average decay
            fp16=False,  # turn on mixed precision training with apex
            save_sample_every=args.save_sample_every,
            avg_window=args.avg_window,
            sched_milestones=sched_milestones,
            results_folder=save_path,
            device=device,
        )

    if args.load_milestone > 0:
        ScaleTrainer.load(milestone=args.load_milestone, mode=args.mode)
    if args.mode == 'train':
        ScaleTrainer.train(args.augment)
        
        # Sample after training is complete
        ScaleTrainer.sample_scales(scale_mul=(1, 1),    # H,W
                                   custom_sample=True,
                                   image_name=args.image_name,
                                   batch_size=args.sample_batch_size,
                                   custom_t_list=sample_t_list
                                   )
    elif args.mode == 'sample':

        # # Sample
        ScaleTrainer.sample_scales(scale_mul=scale_mul,    # H,W
                                   custom_sample=True,
                                   image_name=args.image_name,
                                   batch_size=args.sample_batch_size,
                                   custom_t_list=sample_t_list,
                                   save_unbatched=save_unbatched,
                                   )
    elif args.mode == 'sample_eval':
        # # Sample eval
        ScaleTrainer.bat_sample_scales(scale_mul=scale_mul,    # H,W
                                       custom_sample=True,
                                       batch_size=args.sample_batch_size,
                                       custom_t_list=sample_t_list,
                                       total_num=1000,
                                       )
    
    else:
        raise NotImplementedError()


# ------------------------------------------------------------------------- #
def train_sample_all():     
    
    # mvtectTexture mvtectObject1 mvtectObject2 mvtectObject3
    for target in mvtectObject3:
        target_path      = os.path.join(mvtectAD, target)
        file_names       = os.listdir(target_path)

        for file_name in file_names:
            #if file_name == "good" or file_name == "tmp":
            #    continue

            args.dataset_folder = os.path.join(target_path, file_name) + '/'
            args.image_name     = '000.png'
            save_path           = print_save_params(args)
            time_start          = time.time()

            # 获取当前的时间，转化为yy-mm-dd hh:mm:ss的形式
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('-----------------start train models:{}, start_time:{} -------------------'.format(args.dataset_folder, cur_time))
            
            train_sample_models(args, save_path)

            time_last = time.strftime("%H:%M:%S", time.localtime(time.time() - time_start))
            print('-----------------end train models:{}, last_time:{} -------------------'.format(args.dataset_folder, time_last))

            #break
       
# ------------------------------------------------------------------------- #
def main():
    save_path = print_save_params(args)
    train_sample_models(args, save_path)


if __name__ == '__main__':
    print('----------------num devices: '+ str(torch.cuda.device_count()))
    args = parse_args()

    if args.mode == 'sample' or args.mode == 'sample_eval':
        main()
    elif args.mode == 'train': 
        train_sample_all()
    
    quit()
