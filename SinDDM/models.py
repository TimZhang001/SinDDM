"""
the DDPM model was originally based on
https://github.com/lucidrains/denoising-diffusion-pytorch
"""

from SinDDM.functions import *
from torch import nn
import math
from functools import partial
import torch.nn.functional as F
from torchvision import utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from einops import rearrange
from SinDDM.utils.modules import ConvNextBlock, Residual, PreNorm, Attention, LinearAttention, Upsample, Downsample
from SinDDM.utils.resize_right import resize

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# ---------------理论感受野为35*35------------------#
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# building block modules
class SinDDMConvBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),

        ) if exists(time_emb_dim) else None

        self.time_reshape = nn.Conv2d(time_emb_dim, dim, 1)
        self.ds_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        #self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
    
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            condition = self.time_reshape(condition)
            h = h + condition

        h = self.net(h)
        return h + self.res_conv(x)
    
# denoiser model
class SinDDMNet(nn.Module):
    def __init__(
            self,
            dim,
            out_dim=None,
            channels=3,
            with_time_emb=True,
            multiscale=False,
            device=None
    ):
        super().__init__()
        self.device = device
        self.channels = channels
        self.multiscale = multiscale

        if with_time_emb:
            time_dim = 32

            if multiscale:
                self.SinEmbTime  = SinusoidalPosEmb(time_dim)
                self.SinEmbScale = SinusoidalPosEmb(time_dim)
                self.time_mlp = nn.Sequential(
                    nn.Linear(time_dim * 2, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
            else:
                self.time_mlp = nn.Sequential(
                    SinusoidalPosEmb(time_dim),
                    nn.Linear(time_dim, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
        else:
            time_dim = None
            self.time_mlp = None

        half_dim = int(dim / 2)

        self.l1 = SinDDMConvBlock(channels, half_dim, time_emb_dim=time_dim) # 
        self.l2 = SinDDMConvBlock(half_dim, dim,      time_emb_dim=time_dim)
        self.l3 = SinDDMConvBlock(dim,      dim,      time_emb_dim=time_dim)
        self.l4 = SinDDMConvBlock(dim,      half_dim, time_emb_dim=time_dim)

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(half_dim, out_dim, 1)
        )

    def forward(self, x, time, scale=None):

        if exists(self.multiscale):
            scale_tensor = torch.ones(size=time.shape).to(device=self.device) * scale
            t = self.SinEmbTime(time)
            s = self.SinEmbScale(scale_tensor)
            t_s_vec = torch.cat((t, s), dim=1)
            cond_vec = self.time_mlp(t_s_vec)
        else:
            t = self.time_mlp(time) if exists(self.time_mlp) else None
            cond_vec = t

        x = self.l1(x, cond_vec)
        x = self.l2(x, cond_vec)
        x = self.l3(x, cond_vec)
        x = self.l4(x, cond_vec)

        return self.final_conv(x)

# --------------理论感受野为1+(N*10)--------------------#
# N=16 时，理论感受野为161，实际距离比较远的位置作用很小
class NextNet(nn.Module):
    """
    A backbone model comprised of a chain of ConvNext blocks, with skip connections.
    The skip connections are connected similar to a "U-Net" structure (first to last, middle to middle, etc).
    """
    def __init__(self, in_channels=3, out_channels=3, depth=16, filters_per_layer=64, device=None):
        """
        Args:
            in_channels (int):
                Number of input image channels.
            out_channels (int):
                Number of network output channels.
            depth (int):
                Number of ConvNext blocks in the network.
            filters_per_layer (int):
                Base dimension in each ConvNext block.
        """
        super().__init__()

        if isinstance(filters_per_layer, (list, tuple)):
            dims = filters_per_layer
        else:
            dims = [filters_per_layer] * depth

        time_dim    = dims[0]
        emb_dim     = time_dim
        self.depth  = 16
        self.layers = nn.ModuleList([])
        self.device = device

        # First block doesn't have a normalization layer
        self.layers.append(ConvNextBlock(in_channels, dims[0], emb_dim=emb_dim, norm=False))

        for i in range(1, math.ceil(self.depth / 2)):
            self.layers.append(ConvNextBlock(dims[i - 1], dims[i], emb_dim=emb_dim, norm=True))
        for i in range(math.ceil(self.depth / 2), depth):
            self.layers.append(ConvNextBlock(2 * dims[i - 1],  dims[i], emb_dim=emb_dim, norm=True))

        # After all blocks, do a 1x1 conv to get the required amount of output channels
        self.final_conv = nn.Conv2d(dims[depth - 1], out_channels, 1)

        # Encoder for positional embedding of timestep
        self.SinEmbTime  = SinusoidalPosEmb(time_dim)
        self.SinEmbScale = SinusoidalPosEmb(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim * 2, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

    def forward(self, x, time, scale=None):

        scale_tensor = torch.ones(size=time.shape).to(device=self.device) * scale
        t = self.SinEmbTime(time)
        s = self.SinEmbScale(scale_tensor)
        t_s_vec  = torch.cat((t, s), dim=1)
        cond_vec = self.time_mlp(t_s_vec)
        
        residuals = []
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, cond_vec)
            residuals.append(x)

        for layer in self.layers[math.ceil(self.depth / 2) : self.depth]:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, cond_vec)
        x = self.final_conv(x)

        return x
    
# --------------使用Unet结构，使用了Downsample和Attention 不存在感受野问题------------------#
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        device = None,
    ):
        super().__init__()
        self.channels = channels
        self.device   = device

        dims     = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out   = list(zip(dims[:-1], dims[1:]))
        time_dim = dim

        self.downs = nn.ModuleList([])
        self.ups   = nn.ModuleList([])
        num_resolutions = len(in_out)

        # ---- Downsampling blocks ----
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, emb_dim=time_dim, norm =ind != 0),
                ConvNextBlock(dim_out, dim_out, emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # ---- Middle block ----
        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, emb_dim=time_dim)
        self.mid_attn   = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=4, dim_head=32)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, emb_dim=time_dim)

        # ---- Upsampling blocks ----
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, emb_dim=time_dim),
                ConvNextBlock(dim_in, dim_in, emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # ---- Final convolutions ----
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

        # Encoder for positional embedding of timestep
        self.SinEmbTime  = SinusoidalPosEmb(time_dim)
        self.SinEmbScale = SinusoidalPosEmb(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim * 2, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

    def forward(self, x, time, scale=None):
        
        scale_tensor = torch.ones(size=time.shape).to(device=self.device) * scale
        t = self.SinEmbTime(time)
        s = self.SinEmbScale(scale_tensor)
        t_s_vec  = torch.cat((t, s), dim=1)
        cond_vec = self.time_mlp(t_s_vec)

        initial_shape = x.shape[-2:]
        h = []
        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, cond_vec)
            x = convnext2(x, cond_vec)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond_vec)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond_vec)

        for convnext, convnext2, attn, upsample in self.ups:
            res = h.pop()
            x_resize = resize(x, out_shape=res.shape[-2:])
            x = torch.cat((x_resize, res), dim=1)
            x = convnext(x, cond_vec)
            x = convnext2(x, cond_vec)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(resize(x, out_shape=initial_shape))

# --------------使用全局感受野网络+局部感受野网络合并的方式------------------#
class DetailedNextNet(nn.Module):
    def __init__(self, in_channels=3, depth=16, filters_per_layer=64):
        super(DetailedNextNet,self).__init__()
        
        dims        = [filters_per_layer] * depth
        emb_dim     = dims[0]
        self.depth  = depth
        self.layers = nn.ModuleList([])

        # First block doesn't have a normalization layer
        self.layers.append(ConvNextBlock(in_channels, dims[0], emb_dim=emb_dim, norm=False))

        # Add the rest of the blocks
        for i in range(1, math.ceil(self.depth / 2)):
            self.layers.append(ConvNextBlock(dims[i - 1], dims[i], emb_dim=emb_dim, norm=True))
        for i in range(math.ceil(self.depth / 2), depth):
            self.layers.append(ConvNextBlock(2 * dims[i - 1],  dims[i], emb_dim=emb_dim, norm=True))

    def forward(self, x, cond_vec):
        residuals = []
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, cond_vec)
            residuals.append(x)

        for layer in self.layers[math.ceil(self.depth / 2) : self.depth]:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, cond_vec)

        return x

class SemanticUnet(nn.Module):
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 4), channels = 3):
        super(SemanticUnet,self).__init__()
        
        dims   = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups   = nn.ModuleList([])
        num_resolutions = len(in_out)

        # encoder ---------------------------------------
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out,  emb_dim=dim, norm =ind != 0),
                ConvNextBlock(dim_out, dim_out, emb_dim=dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # middle ---------------------------------------
        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, emb_dim=dim)
        self.mid_attn   = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=4, dim_head=32)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, emb_dim=dim)

        # decoder---------------------------------------
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, emb_dim=dim),
                ConvNextBlock(dim_in, dim_in, emb_dim=dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

    def forward(self, x, cond_vec):
        initial_shape = x.shape[-2:]
        h = []

        # encoder ---------------------------------------
        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x,  cond_vec)
            x = convnext2(x, cond_vec)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # middle ---------------------------------------
        x = self.mid_block1(x, cond_vec)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond_vec)

        # decoder---------------------------------------
        for convnext, convnext2, attn, upsample in self.ups:
            res = h.pop()
            x_resize = resize(x, out_shape=res.shape[-2:])
            x = torch.cat((x_resize, res), dim=1)
            x = convnext(x, cond_vec)
            x = convnext2(x, cond_vec)
            x = attn(x)
            x = upsample(x)

        x = resize(x, out_shape=initial_shape)
        return x
        

class DiffusionBiSeNet(nn.Module):
    def __init__(self, in_channels=3, detail_depth=16, dim=32, semantic_mults=(1, 2, 4, 4), device = None):
        super(DiffusionBiSeNet, self).__init__()
        
        self.device   = device
        
        # detail branch
        self.nextnet  = DetailedNextNet(in_channels, detail_depth, dim)
        
        # semantic branch
        self.unet     = SemanticUnet(dim, semantic_mults, in_channels)
        
        # union branch
        self.nextnet_final = nn.Conv2d(dim, in_channels, 1)
        self.unet_final    = nn.Sequential(ConvNextBlock(dim, dim), 
                                           nn.Conv2d(dim, 3, 1),)
        
        # Encoder for positional embedding of timestep
        self.SinEmbTime  = SinusoidalPosEmb(dim)
        self.SinEmbScale = SinusoidalPosEmb(dim)
        self.time_mlp    = nn.Sequential(nn.Linear(dim * 2, dim * 4),
                                         nn.GELU(),
                                         nn.Linear(dim * 4, dim),)

    def forward(self, x, time, scale=None):
        scale_tensor = torch.ones(size=time.shape).to(device=self.device) * scale
        t = self.SinEmbTime(time)
        s = self.SinEmbScale(scale_tensor)
        t_s_vec  = torch.cat((t, s), dim=1)
        cond_vec = self.time_mlp(t_s_vec)

        if scale > 0:
            detail_feat = self.nextnet(x, cond_vec)
            x_nextnet   = self.nextnet_final(detail_feat)
            return x_nextnet
        else:
            semantic_feat = self.unet(x, cond_vec)
            x_unet        = self.unet_final(semantic_feat)
            return x_unet


# ------------------------------------------------------------------------
class MultiScaleGaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            save_interm=False,
            results_folder = '/Results',
            n_scales,
            scale_factor,
            image_sizes,
            scale_mul=(1, 1),
            channels=3,
            timesteps=100,
            train_full_t=False,
            scale_losses=None,
            loss_factor=1,
            loss_type='l1',
            betas=None,
            device=None,
            reblurring=True,
            sample_limited_t=False,
            omega=0,
    ):
        super().__init__()
        self.device = device
        self.save_interm = save_interm
        self.results_folder = Path(results_folder)
        self.channels = channels
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.image_sizes = ()
        self.scale_mul = scale_mul

        self.sample_limited_t = sample_limited_t
        self.reblurring = reblurring

        self.img_prev_upsample = None

        # CLIP guided sampling
        self.clip_guided_sampling = False
        self.guidance_sub_iters = None
        self.stop_guidance = None
        self.quantile = 0.8
        self.clip_model = None
        self.clip_strength = None
        self.clip_text = ''
        self.text_embedds = None
        self.text_embedds_hr = None
        self.text_embedds_lr = None
        self.clip_text_features = None
        self.clip_score = []
        self.clip_mask = None
        self.llambda = 0
        self.x_recon_prev = None

        # for clip_roi
        self.clip_roi_bb = []

        # omega tests
        self.omega = omega

        # ROI guided sampling
        self.roi_guided_sampling = False
        self.roi_bbs = []  # roi_bbs - list of [y,x,h,w]
        self.roi_bbs_stat = []  # roi_bbs_stat - list of [mean_tensor[1,3,1,1], std_tensor[1,3,1,1]]
        self.roi_target_patch = []

        for i in range(n_scales):  # flip xy->hw
            self.image_sizes += ((image_sizes[i][1], image_sizes[i][0]),)

        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        # self.num_timesteps_trained = int(timesteps_trained) # overwritten if scale_loss is given
        self.num_timesteps_trained = []
        self.num_timesteps_ideal = []
        self.num_timesteps_trained.append(self.num_timesteps)
        self.num_timesteps_ideal.append(self.num_timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',  to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',     to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',   to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        sigma_t = np.sqrt(1. - alphas_cumprod) / np.sqrt(alphas_cumprod) # sigma_t = sqrt_one_minus_alphas_cumprod_div_sqrt_alphas_cumprod

        # flag to force training of all the timesteps across all scales
        if scale_losses is not None:
            for i in range(n_scales - 1):
                self.num_timesteps_ideal.append(int(np.argmax(sigma_t > loss_factor * scale_losses[i])))
                if train_full_t:
                    self.num_timesteps_trained.append(int(timesteps))
                else:
                    self.num_timesteps_trained.append(self.num_timesteps_ideal[i+1])

        # gamma blur schedule
        gammas = torch.zeros(size=(n_scales - 1, self.num_timesteps), device=self.device)
        for i in range(n_scales - 1):
            gammas[i,:] = (torch.tensor(sigma_t, device=self.device) / (loss_factor * scale_losses[i])).clamp(min=0, max=1)

        self.register_buffer('gammas', gammas)

    # for roi_guided_sampling
    #
    def roi_patch_modification(self, x_recon, scale=0, eta=0.8):
        x_modified = x_recon
        for bb in self.roi_bbs:  # bounding box is of shape [y,x,h,w]
            bb = [int(bb_i / np.power(self.scale_factor, self.n_scales - scale - 1)) for bb_i in bb]
            bb_y, bb_x, bb_h, bb_w = bb
            target_patch_resize = F.interpolate(self.roi_target_patch[scale], size=(bb_h, bb_w))
            x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w] = eta * target_patch_resize + (1-eta) * x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]
        return x_modified

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, s, noise):

        x_recon_ddpm = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

        if not self.reblurring or s == 0:
            return x_recon_ddpm, x_recon_ddpm  # x_t_mix == x_tm1_mix at scale 0
        else:
            cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)
            x_tm1_mix = (x_recon_ddpm - extract(cur_gammas, t, x_recon_ddpm.shape) * self.img_prev_upsample) / (
                        1 - extract(cur_gammas, t, x_recon_ddpm.shape))
            x_t_mix = x_recon_ddpm # without subtraction of the blurry part
            return x_tm1_mix, x_t_mix


    def q_posterior(self, x_start, x_t_mix, x_t, t, s):
        if not self.reblurring or s == 0:
            # regular DDPM
            posterior_mean = (
                    extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                    extract(self.posterior_mean_coef2, t, x_t.shape) * x_t

            )
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        elif t[0] > 0:
            x_tm1_mix = x_start
            posterior_variance_low = torch.zeros(x_t.shape,
                                                 device=self.device)
            posterior_variance_high = 1 - extract(self.alphas_cumprod, t - 1, x_t.shape)
            omega = self.omega
            posterior_variance = (1-omega) * posterior_variance_low + omega * posterior_variance_high
            posterior_log_variance_clipped = torch.log(posterior_variance.clamp(1e-20,None))

            var_t = posterior_variance

            posterior_mean = extract(self.sqrt_alphas_cumprod, t-1, x_t.shape) * x_tm1_mix + \
                                    torch.sqrt(1-extract(self.alphas_cumprod, t-1, x_t.shape) - var_t) * \
                                    (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t_mix) / \
                                    extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        else:
            posterior_mean = x_start  # for t==0 no noise added
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.enable_grad()
    def p_mean_variance(self, x, t, s, clip_denoised: bool):
        pred_noise = self.denoise_fn(x, t, scale=s)
        x_recon, x_t_mix = self.predict_start_from_noise(x, t=t, s=s, noise=pred_noise)
        cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (x_recon.clamp(-1., 1.) + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'denoised_t-{t[0]:03}_s-{s}.png'),
                             nrow=4)
        # CLIP guidance
        if self.clip_guided_sampling and (self.stop_guidance <= t[0] or s < self.n_scales - 1) and self.guidance_sub_iters[s] > 0:
            if clip_denoised:
                x_recon.clamp_(-1., 1.)

            # preserve CLIP changes from previous iteration
            if self.clip_mask is not None:
                x_recon = x_recon * (1 - self.clip_mask) + (
                        (1 - self.llambda) * self.x_recon_prev + self.llambda * x_recon) * self.clip_mask
            x_recon.requires_grad_(True)  # for autodiff

            x_recon_renorm = (x_recon + 1) * 0.5
            for i in range(self.guidance_sub_iters[s]):  # can experiment with more than 1 iter. per timestep
                self.clip_model.zero_grad()
                # choose text embedding augmentation (High Res / Low Res)
                if s > 0:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.text_embedds_hr)
                else:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.text_embedds_lr)

                clip_grad = torch.autograd.grad(score, x_recon, create_graph=False)[0]

                # create CLIP mask depending on the strongest gradients locations
                if self.clip_mask is None:
                    clip_grad, clip_mask = thresholded_grad(grad=clip_grad, quantile=self.quantile)
                    self.clip_mask = clip_mask.float()

                if self.save_interm:
                    final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
                    final_results_folder.mkdir(parents=True, exist_ok=True)
                    final_mask = self.clip_mask.type(torch.float64)

                    utils.save_image(final_mask,
                                     str(final_results_folder / f'clip_mask_s-{s}.png'),
                                     nrow=4)
                    utils.save_image((x_recon.clamp(-1., 1.) + 1) * 0.5,
                                     str(final_results_folder / f'clip_out_s-{s}_t-{t[0]}_subiter_{i}.png'),
                                     nrow=4)

                #normalize gradients
                division_norm = torch.linalg.vector_norm(x_recon * self.clip_mask, dim=(1,2,3), keepdim=True) / torch.linalg.vector_norm(
                    clip_grad * self.clip_mask, dim=(1,2,3), keepdim=True)

                # update clean image
                x_recon += self.clip_strength * division_norm * clip_grad * self.clip_mask
                x_recon.clamp_(-1., 1.)

                # prepare for next sub-iteration
                x_recon_renorm = (x_recon + 1) * 0.5
                # plot score
                self.clip_score.append(score.detach().cpu())

            self.x_recon_prev = x_recon.detach()

            # plot clip loss
            plt.rcParams['figure.figsize'] = [16, 8]
            plt.plot(self.clip_score)
            plt.grid(True)
            # plt.ylim((0, 0.2))
            plt.savefig(str(self.results_folder / 'clip_score'))
            plt.clf()

        # ROI guided sampling
        elif self.roi_guided_sampling and (s < self.n_scales-1):
            x_recon = self.roi_patch_modification(x_recon, scale=s)

        # else normal sampling
        if int(s) > 0 and t[0] > 0 and self.reblurring:
            x_tm1_mix = extract(cur_gammas, t - 1, x_recon.shape) * self.img_prev_upsample + \
                        (1 - extract(cur_gammas, t - 1, x_recon.shape)) * x_recon  # mix blurred and orig
        else:
            x_tm1_mix = x_recon

        if clip_denoised:
            x_tm1_mix.clamp_(-1., 1.)
            x_t_mix.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_tm1_mix, x_t_mix=x_t_mix,
                                                                                  x_t=x, t=t, s=s,
                                                                                  )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, s, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, clip_denoised=clip_denoised)

        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        nonzero_mask_s = torch.tensor([True], device=self.device).float()

        return model_mean + nonzero_mask_s * nonzero_mask * (0.5 * model_log_variance).exp() * noise


    @torch.no_grad()
    def p_sample_loop(self, shape, s):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'input_noise_s-{s}.png'),
                             nrow=4)
        if self.sample_limited_t and s < (self.n_scales-1):
            t_min = self.num_timesteps_ideal[s+1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), s)
            if self.save_interm:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img,
                                 str(final_results_folder / f'output_t-{i:03}_s-{s}.png'),
                                 nrow=4)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, scale_0_size=None, s=0):
        """
        Sample from the first scale (without conditioning on a previous scale's output).
        """
        if scale_0_size is not None:
            image_size = scale_0_size
        else:
            image_size = self.image_sizes[0]
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size[0], image_size[1]), s=s)

    @torch.no_grad()
    def p_sample_via_scale_loop(self, batch_size, img, s, custom_t=None):
        device = self.betas.device
        if custom_t is None:
            total_t = self.num_timesteps_ideal[min(s, self.n_scales-1)]-1
        else:
            total_t = custom_t
        b = batch_size
        # store upsampled image from previous scale as x^s_tilda
        self.img_prev_upsample = img
        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'clean_input_s_{s}.png'),
                             nrow=4)
        img = self.q_sample(x_start=img, t=torch.Tensor.expand(torch.tensor(total_t, device=device), batch_size), noise=None)  # add noise

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'noisy_input_s_{s}.png'),
                             nrow=4)

        if self.clip_mask is not None:
            if s > 0:
                mul_size          = [int(self.image_sizes[s][0]* self.scale_mul[0]), int(self.image_sizes[s][1]* self.scale_mul[1])]
                self.clip_mask    = F.interpolate(self.clip_mask, size=mul_size, mode='bilinear')
                self.x_recon_prev = F.interpolate(self.x_recon_prev, size=mul_size, mode='bilinear')
            else:  # mask created at scale 0 is usually too noisy
                self.clip_mask = None

        if self.sample_limited_t and s < (self.n_scales - 1):
            t_min = self.num_timesteps_ideal[s + 1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, total_t)), desc='sampling loop time step', total=total_t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), s)
            if self.save_interm:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img,
                                 str(final_results_folder / f'output_t-{i:03}_s-{s}.png'),
                                 nrow=4)
        return img

    @torch.no_grad()
    def sample_via_scale(self, batch_size, img, s, scale_mul=(1, 1), custom_sample=False, custom_img_size_idx=0, custom_t=None, custom_image_size=None):
        """
        Sampling at a given scale s conditioned on the output of a previous scale.
        """
        if custom_sample:
            if custom_img_size_idx >= self.n_scales:  # extrapolate size
                size   = self.image_sizes[self.n_scales-1]
                factor = self.scale_factor ** (custom_img_size_idx + 1 - self.n_scales)
                size   = (int(size[0] * factor), int(size[1] * factor))
            else:
                size = self.image_sizes[custom_img_size_idx]
        else:
            size = self.image_sizes[s]
        
        image_size = (int(size[0] * scale_mul[0]), int(size[1] * scale_mul[1]))
        if custom_image_size is not None:  # force custom image size
            image_size = custom_image_size

        img = F.interpolate(img, size=image_size, mode='bilinear')
        return self.p_sample_via_scale_loop(batch_size, img, s, custom_t=custom_t)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, s, noise=None, x_orig=None, x_mask = None):
        b, c, h, w = x_start.shape
        noise   = default(noise, lambda: torch.randn_like(x_start))
        results = []

        if int(s) > 0:
            cur_gammas = self.gammas[s - 1].reshape(-1)
            x_mix   = extract(cur_gammas, t, x_start.shape) * x_start + (1 - extract(cur_gammas, t, x_start.shape)) * x_orig  # mix blurred and orig
            x_noisy = self.q_sample(x_start=x_mix, t=t, noise=noise)  # add noise
            x_recon = self.denoise_fn(x_noisy, t, s)

        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t, s)

        if self.loss_type == 'l1' or self.loss_type == 'l2':
            loss, results = self.cal_loss(noise, recon = x_recon, x_mask = x_mask, scale = s, weight=None)
        elif self.loss_type == 'l1_pred_img':
            if int(s) > 0:
                cur_gammas = self.gammas[s - 1].reshape(-1)
                if t[0]>0:
                    x_mix_prev = extract(cur_gammas, t-1, x_start.shape) * x_start + (1 - extract(cur_gammas, t-1, x_start.shape)) * x_orig  # mix blurred and orig
                else:
                    x_mix_prev = x_orig
            else:
                x_mix_prev = x_start
            loss = (x_mix_prev-x_recon).abs().mean()
            results['loss'] = loss
        else:
            raise NotImplementedError()

        return loss, results
    
    # -----------cal loss ----------------
    def cal_loss(self, noise, recon, x_mask= None, scale = 0, weight=None):
        results    = {}
        loss       = 0

        #-------------------------------------------------------------------------------
        # 定义权重
        if weight is None:
            if scale ==2:
                weight = dict({'fg': 0, 'bg':  0, 'image': 1})
            else:
                weight = dict({'fg': 0, 'bg':  0, 'image': 1})

        # 计算前景和背景的loss
        if weight['fg'] > 0:
            loss_fg       = self.cal_region_loss(recon, noise, x_mask, type='fg') * weight['fg']  
            loss          = loss + loss_fg
            results['fg'] = loss_fg.item()

        if weight['bg'] > 0:
            loss_bg       = self.cal_region_loss(recon, noise, x_mask, type='bg') * weight['bg']
            loss          = loss + loss_bg
            results['bg'] = loss_bg.item()
        
        if weight['image'] > 0:
            loss_image       = self.cal_region_loss(recon, noise, x_mask, type='image') * weight['image']
            loss             = loss + loss_image
            results['image'] = loss_image.item()

        # 返回loss
        results['loss'] = loss.item()
        return loss, results
     
    
    def cal_region_loss(self, recon, noise, mask, type='fg'):
        if type == 'fg':
            inner_mask = mask
        elif type == 'bg':
            inner_mask = 1 - mask
        else:
            inner_mask = torch.ones_like(mask)
        
        recon = recon * inner_mask.float()
        noise = noise * inner_mask.float()

        # cal diffusion loss
        if self.loss_type == 'l2':
            loss = F.mse_loss(noise.float(), recon.float(), reduction='sum')
        else:
            loss = (noise - recon).abs().sum()
        loss = loss / (self.channels * inner_mask.sum())

        return loss
 

    def forward(self, x, s, *args, **kwargs):
        if int(s) > 0:  # no deblurring in scale=0
            x_orig  = x[0]
            x_recon = x[1]
            x_mask  = x[2]
            b, c, h, w = x_orig.shape
            device   = x_orig.device
            img_size = self.image_sizes[s]
            # Tim 先不考虑分辨率的问题，因为使用了Large Crop 
            #assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x_recon, t, s, x_orig=x_orig, x_mask = x_mask, *args, **kwargs)

        else:

            b, c, h, w = x[0].shape
            device = x[0].device
            x_mask = x[2]
            img_size = self.image_sizes[s]
            # Tim 先不考虑分辨率的问题，因为使用了Large Crop 
            #assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x[0], t, s, x_mask = x_mask, *args, **kwargs)

