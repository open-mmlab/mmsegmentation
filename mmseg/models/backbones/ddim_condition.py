"""SAMPLING ONLY."""

import torch
from torch import nuclear_norm
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import cv2
from einops import rearrange

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from ldm.util import default

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from ldm.cldm import ControlNet

import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


@MODELS.register_module()
class DDIMSampler(BaseModule):
    def __init__(self, model, pretrained, schedule="linear", out_indices=(0, 1, 2, 3), **kwargs):
        super().__init__()
        self.model = create_model(model)
        self.model.load_state_dict(load_state_dict(pretrained, location='cuda'), strict=False)
        self.ddpm_num_timesteps = self.model.num_timesteps
        self.schedule = schedule
        self.out_indices = out_indices
        self.make_schedule(ddim_num_steps=20, ddim_eta=1, verbose=False)
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        for param in self.parameters():
            param.requires_grad = False
    
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        """
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule, **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img], 'cross_attn_maps': [None], 'pseudo_masks': None}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            img, pred_x0, layers_cross_attn_maps = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            # if index % log_every_t == 0 or index == total_steps - 1:
            if True:
                intermediates['x_inter'].append(img.cpu())
                intermediates['pred_x0'].append(pred_x0.cpu())
                intermediates['cross_attn_maps'].append(layers_cross_attn_maps)
            
            if i + 2 == kwargs['control_start_step']:
                pseudo_masks = self.get_pseudo_mask(intermediates['cross_attn_maps'][-1:], noun_idx=[6] * img.shape[0], shape=img.shape)
                intermediates['pseudo_masks'] = pseudo_masks
                pseudo_masks = pseudo_masks.to(self.model.betas.device)
                cond["c_concat"] = [pseudo_masks]
        
        if intermediates['pseudo_masks'] is None:
            pseudo_masks = self.get_pseudo_mask(intermediates['cross_attn_maps'][-1:], noun_idx=[6] * img.shape[0], shape=img.shape)
            intermediates['pseudo_masks'] = pseudo_masks

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output, layers_cross_attn_maps = self.model.apply_model(x, t, c)
        else:
            model_t, layers_cross_attn_maps = self.model.apply_model(x, t, c)
            model_uncond, _ = self.model.apply_model(x, t, unconditional_conditioning)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0, layers_cross_attn_maps

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
    
    
    @torch.no_grad()
    def sample_create_image_mask(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None, 
               **kwargs
               ):
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        """
        
        
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        self.make_schedule(ddim_num_steps=kwargs['control_start_step'], ddim_eta=eta, verbose=verbose)
                
        contour_masks, foreground_masks = self.first_step(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule, **kwargs
                                                    )
        
        conditioning["c_concat"] = [contour_masks.to(self.model.betas.device)]
        unconditional_conditioning["c_concat"] = [contour_masks.to(self.model.betas.device)]
        
        self.make_schedule(ddim_num_steps=S - kwargs['control_start_step'], ddim_eta=eta, verbose=verbose)
        samples = self.second_step(conditioning, size,
                            callback=callback,
                            img_callback=img_callback,
                            quantize_denoised=quantize_x0,
                            mask=mask, x0=x0,
                            ddim_use_original_steps=False,
                            noise_dropout=noise_dropout,
                            temperature=temperature,
                            score_corrector=score_corrector,
                            corrector_kwargs=corrector_kwargs,
                            x_T=x_T,
                            log_every_t=log_every_t,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            dynamic_threshold=dynamic_threshold,
                            ucg_schedule=ucg_schedule, **kwargs
                            )
        intermediates = {"pseudo_masks": foreground_masks, 'contour_masks': contour_masks}
        
        return samples, intermediates
    
    @torch.no_grad()
    def first_step(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        
        intermediates = {'cross_attn_maps': []}
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]
                
                
            img, pred_x0, layers_cross_attn_maps = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            
            intermediates['cross_attn_maps'].append(layers_cross_attn_maps)
        
        contour_masks, foreground_masks = self.get_pseudo_mask(intermediates['cross_attn_maps'], noun_idx=[6] * img.shape[0], shape=img.shape)

        return contour_masks, foreground_masks
    
    
    @torch.no_grad()
    def second_step(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img], 'cross_attn_maps': [None], 'pseudo_masks': None}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]
                
                
            img, pred_x0, layers_cross_attn_maps = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

        return img
    
    
    @torch.no_grad()
    def get_pseudo_mask(self, all_cross_attn_maps, noun_idx, shape):
        noun_idx = torch.tensor(noun_idx)
        B, C, H, W = shape
        pseudo_masks = torch.zeros(B, 1, H, W)
        for layers_cross_attn_maps in all_cross_attn_maps:
            for cross_attn_maps in layers_cross_attn_maps:
                for cross_attn_map in cross_attn_maps:
                    cross_attn_map = cross_attn_map.cpu()
                    cross_attn_map = rearrange(cross_attn_map, '(b h) (H W) N -> b h N H W', h=8, H=int(np.sqrt(cross_attn_map.shape[1])))
                    cross_attn_map = cross_attn_map.sum(dim=1)                    
                    cross_attn_map = torch.gather(input=cross_attn_map, dim=1, index=noun_idx.reshape(B, 1, 1).repeat(1, *cross_attn_map.shape[2:]).unsqueeze(1))
                    cross_attn_map = F.interpolate(cross_attn_map, size=shape[2:], mode='bilinear')
                    pseudo_masks = pseudo_masks + cross_attn_map
        pseudo_masks = (pseudo_masks - pseudo_masks.min()) / (pseudo_masks.max() - pseudo_masks.min()) / 2        
        
        pseudo_masks = F.interpolate(pseudo_masks, size=(512, 512), mode='bilinear')
        
        # fixed threshold
        # pseudo_masks[pseudo_masks >= 0.5] = 1
        # pseudo_masks[pseudo_masks < 0.5] = 0
        
        binary_masks = (pseudo_masks.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
        
        def get_max_connected(m):
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(cnt) for cnt in contours]
            max_idx = np.argmax(areas)                
            contour_mask = np.zeros_like(m)
            cv2.drawContours(contour_mask, contours, max_idx, 255, 4)
            contour_mask = nms(contour_mask, 127, 3.0)
            contour_mask = cv2.GaussianBlur(contour_mask, (0, 0), 3.0)
            contour_mask[contour_mask > 4] = 255
            contour_mask[contour_mask < 255] = 0
            
            foreground_mask = np.zeros_like(m)
            cv2.drawContours(foreground_mask, contours, max_idx, 255, -1)
            
            return contour_mask, foreground_mask

        binary_masks = [cv2.threshold(bm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for bm in binary_masks]
        contour_masks = []
        foreground_masks = []
        for bm in binary_masks:
            # from skimage.filters import threshold_otsu
            # thresh = threshold_otsu(bm)
            # bm[bm < thresh] = 0
            # bm[bm >= thresh] = 255
            cm, fm = get_max_connected(bm)
            contour_masks.append(cm)
            foreground_masks.append(fm)
        
        contour_masks = np.concatenate([cm[None, ...] for cm in contour_masks], axis=0)
        contour_masks = torch.from_numpy(contour_masks / 255).unsqueeze(1).repeat(1, 3, 1, 1)
        foreground_masks = np.concatenate([fm[None, ...] for fm in foreground_masks], axis=0)
        foreground_masks = torch.from_numpy(foreground_masks / 255).unsqueeze(1)
        return contour_masks.float(), foreground_masks.float()
    
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
    @torch.no_grad()
    def forward(self, x):
        encoder_posterior = self.model.encode_first_stage(x)
        z = self.model.get_first_stage_encoding(encoder_posterior).detach()
        noise = torch.randn_like(z).to(z)
        t = 50 * z.new_ones(z.shape[0], dtype=torch.int64)
        x_noisy = self.q_sample(x_start=z, t=t, noise=noise)
        
        cond = {"c_concat": None, "c_crossattn": [self.model.get_learned_conditioning([''] * z.shape[0])]}
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        
        features = self.model.model.diffusion_model.forward_features(x=x_noisy, timesteps=t, context=cond_txt, control=None)
        outs = []
        for i, feat in enumerate(features):
            if i in self.out_indices:
                outs.append(feat)
        return outs
    