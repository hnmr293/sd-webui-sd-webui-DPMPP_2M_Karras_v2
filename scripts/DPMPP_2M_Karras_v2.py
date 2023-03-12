import sys
import functools
import torch
from torch import nn
from tqdm import trange

import k_diffusion.sampling # type: ignore
from modules import shared
from modules import sd_samplers, sd_samplers_common
import modules.sd_samplers_kdiffusion as K

@torch.no_grad()
def sample_dpmpp_2m_test(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    
    # cf. https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        
        t_min = min(sigma_fn(t_next), sigma_fn(t))
        t_max = max(sigma_fn(t_next), sigma_fn(t))

        if old_denoised is None or sigmas[i + 1] == 0:
            x = (t_min / t_max) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])

            h_min = min(h_last, h)
            h_max = max(h_last, h)
            r = h_max / h_min

            h_d = (h_max + h_min) / 2
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_min / t_max) * x - (-h_d).expm1() * denoised_d

        old_denoised = denoised
    return x

class KDiffusionSamplerLocal(K.KDiffusionSampler):
    
    def __init__(
        self,
        funcname: str,
        original_funcname: str,
        func,
        sd_model: nn.Module
    ):
        # here we do not call super().__init__() 
        # because target function is not in k_diffusion.sampling
        
        denoiser = k_diffusion.external.CompVisVDenoiser if sd_model.parameterization == "v" else k_diffusion.external.CompVisDenoiser

        self.model_wrap = denoiser(sd_model, quantize=shared.opts.enable_quantization)
        self.funcname = funcname
        self.func = func
        self.extra_params = K.sampler_extra_params.get(original_funcname, [])
        self.model_wrap_cfg = K.CFGDenoiser(self.model_wrap)
        self.sampler_noises = None
        self.stop_at = None
        self.eta = None
        self.config = None
        self.last_latent = None

        self.conditioning_key = sd_model.model.conditioning_key # type: ignore


def add_dpmpp_2m_test():
    original = [ x for x in K.samplers_k_diffusion if x[0] == 'DPM++ 2M Karras' ][0]
    o_label, o_constructor, o_aliases, o_options = original
    
    label = o_label + ' test'
    funcname = sample_dpmpp_2m_test.__name__
    
    def constructor(model: nn.Module):
        return KDiffusionSamplerLocal(funcname, o_constructor, sample_dpmpp_2m_test, model)
    
    aliases = [ x + '_test' for x in o_aliases ]
    
    options = { **o_options }
    
    data = sd_samplers_common.SamplerData(label, constructor, aliases, options)
    
    sd_samplers.all_samplers.append(data)


def update_samplers():
    sd_samplers.set_samplers()
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}


def hook(fn):
    
    @functools.wraps(fn)
    def f(*args, **kwargs):
        old_samplers, mode, *rest = args
        
        if mode not in ['txt2img', 'img2img']:
            print(f'unknown mode: {mode}', file=sys.stderr)
            return fn(*args, **kwargs)
        
        update_samplers()
        
        new_samplers = (
            sd_samplers.samplers if mode == 'txt2img' else
            sd_samplers.samplers_for_img2img
        )
        
        return fn(new_samplers, mode, *rest, **kwargs)
    
    return f


# register new sampler
add_dpmpp_2m_test()
update_samplers()


# hook Sampler textbox creation
from modules import ui

ui.create_sampler_and_steps_selection = hook(ui.create_sampler_and_steps_selection)
