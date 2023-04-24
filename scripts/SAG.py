import math
import os
from inspect import isfunction
from typing import Tuple

import gradio as gr
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

import modules.scripts as scripts
from modules import shared
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import (
    AfterCFGCallbackParams,
    CFGDenoisedParams,
    CFGDenoiserParams,
    on_cfg_after_cfg,
    on_cfg_denoised,
    on_cfg_denoiser,
)
from scripts import xyz_grid_support_sag

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
_DEFAULT_SCALE = 0.75
_DEFAULT_MASK_THRESH = 1.0


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class LoggedSelfAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attn_probs = None

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type="cuda"):
                q, k = q.float(), k.float()
                sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        self.attn_probs = sim

        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


def xattn_forward_log(self, x, context=None, mask=None):
    global current_selfattn_map

    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION == "fp32":
        with torch.autocast(enabled=False, device_type="cuda"):
            q, k = q.float(), k.float()
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
    else:
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
    del q, k

    if exists(mask):
        mask = rearrange(mask, "b ... -> b (...)")
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, "b j -> (b h) () j", h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    self.attn_probs = sim
    current_selfattn_map = sim

    out = einsum("b i j, b j d -> b i d", sim, v)
    out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
    out = self.to_out(out)
    return out


current_selfattn_map = []


def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


class Script(scripts.Script):
    enabled: bool = False
    guidance_scale: float = _DEFAULT_SCALE
    mask_threshold: float = _DEFAULT_MASK_THRESH
    callbacks_added: bool = False

    original_selfattn_forward: callable = None
    xin = None
    batch_size: int = 1
    degraded_pred = None
    degraded_pred_compensation = None
    uncond_pred = None
    unet_kwargs = {}

    def __init__(self):
        pass

    def title(self):
        return "Self Attention Guidance"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def denoiser_callback(self, params: CFGDenoiserParams):
        if self.enabled is not True:
            return

        # logging current uncond size for cond/uncond output separation
        self.batch_size = params.text_uncond.shape[0]
        # logging current input for eps calculation later
        self.xin = params.x[-(self.batch_size) :]

        # logging necessary information for SAG pred
        current_uncond_emb = params.text_uncond
        current_sigma = params.sigma
        current_image_cond_in = params.image_cond
        self.unet_kwargs = {
            "sigma": current_sigma[-(self.batch_size) :],
            "image_cond": current_image_cond_in[-(self.batch_size) :],
            "text_uncond": current_uncond_emb,
        }

    def denoised_callback(self, params: CFGDenoisedParams):
        if self.enabled is not True:
            return
        global current_selfattn_map
        # output from DiscreteEpsDDPMDenoiser is already pred_x0
        uncond_output = params.x[-(self.batch_size) :]
        original_latents = uncond_output
        self.uncond_pred = uncond_output

        # Produce attention mask
        # We're only interested in the last (self.batch_size)*head_count slices of logged self-attention map
        attn_map = current_selfattn_map[-(self.batch_size * 8) :]
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = 8

        middle_layer_latent_size = [math.ceil(latent_h / 8), math.ceil(latent_w / 8)]

        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > self.mask_threshold
        attn_mask = (
            attn_mask.reshape(b, middle_layer_latent_size[0], middle_layer_latent_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # Blur according to the self-attention mask
        degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)
        renoised_degraded_latent = degraded_latents - (uncond_output - self.xin)

        # get predicted x0 in degraded direction
        self.degraded_pred_compensation = uncond_output - degraded_latents
        if shared.sd_model.model.conditioning_key == "crossattn-adm":
            condition_dict = {
                "c_crossattn": self.unet_kwargs["text_uncond"],
                "c_adm": self.unet_kwargs["image_cond"],
            }
        else:
            condition_dict = {
                "c_crossattn": self.unet_kwargs["text_uncond"],
                "c_concat": [self.unet_kwargs["image_cond"]],
            }

        self.degraded_pred = params.inner_model(
            renoised_degraded_latent,
            self.unet_kwargs["sigma"],
            cond=condition_dict,
        )

    def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams):
        if self.enabled is not True:
            return

        params.x += (
            self.uncond_pred - (self.degraded_pred + self.degraded_pred_compensation)
        ) * self.guidance_scale
        params.output_altered = True

    def ui(self, is_img2img):
        with gr.Accordion("Self Attention Guidance", open=False, elem_id="sag_accordion"):
            enabled = gr.Checkbox(label="Enabled", default=False, elem_id="sag_enabled")
            scale = gr.Slider(
                label="Scale", minimum=-2.0, maximum=10.0, step=0.01, value=0.75, elem_id="sag_scale"
            )
            mask_threshold = gr.Slider(
                label="Mask Threshold", minimum=0.0, maximum=2.0, step=0.01, value=1.0, elem_id="sag_mask_th"
            )
        return [enabled, scale, mask_threshold]

    def process(
        self, p: StableDiffusionProcessing, enabled: bool, scale: float, mask_threshold: float, **kwargs
    ):
        if enabled is True:
            self.enabled = enabled
            self.mask_threshold = mask_threshold
            self.guidance_scale = scale

            # save original forward function for later restoration
            org_attn_module = (
                shared.sd_model.model.diffusion_model.middle_block._modules["1"]
                .transformer_blocks._modules["0"]
                .attn1
            )
            self.original_selfattn_forward = org_attn_module.forward

            # replace target self attention module in unet with ours
            org_attn_module.forward = xattn_forward_log.__get__(org_attn_module, org_attn_module.__class__)

            # add extra parameters to the generation metadata
            p.extra_generation_params["SAG Guidance Scale"] = self.guidance_scale
            p.extra_generation_params["SAG Mask Threshold"] = self.mask_threshold

        else:
            self.enabled = False

        if self.callbacks_added is not True:
            on_cfg_denoiser(self.denoiser_callback)
            on_cfg_denoised(self.denoised_callback)
            on_cfg_after_cfg(self.cfg_after_cfg_callback)
            self.callbacks_added = True

        return

    def postprocess(self, p, processed, enabled: bool, scale: float, mask_threshold: float):
        if enabled is True:
            # restore original self attention module forward function
            attn_module = (
                shared.sd_model.model.diffusion_model.middle_block._modules["1"]
                .transformer_blocks._modules["0"]
                .attn1
            )
            attn_module.forward = self.original_selfattn_forward
            # Reset module storage state
            self.clear_state()
        return

    def clear_state(self):
        global current_selfattn_map
        current_selfattn_map = []
        self.original_selfattn_forward = None
        self.xin = None
        self.batch_size = 1
        self.degraded_pred = None
        self.degraded_pred_compensation = None
        self.uncond_pred = None
        self.unet_kwargs = {}


xyz_grid_support_sag.initialize(Script)
