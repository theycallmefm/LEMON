# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MultiControlNet module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import sys
from dataclasses import dataclass
from typing import Union

import torch
from rich.console import Console
from torch import Tensor, nn
from jaxtyping import Float

CONSOLE = Console(width=120)
import torch.nn.functional as F

try:
    from diffusers import (
        DDIMScheduler,
        StableDiffusionInstructPix2PixPipeline,
        ControlNetModel,
        StableDiffusionControlNetPipeline
    )
    from transformers import logging

except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)

logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"


CONTROLNET_SOURCE = "fusing/stable-diffusion-v1-5-controlnet-normal"

CONTROLNET_NORMAL_SOURCE = "lllyasviel/control_v11p_sd15_normalbae"
CONTROLNET_IP2P_SOURCE = "lllyasviel/control_v11e_sd15_ip2p"
CONTROLNET_DEPTH_SOURCE = "lllyasviel/sd-controlnet-depth"

@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class MultiControlNet(nn.Module):
    """ControlNet implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=False, args = None) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision
        self.controlnet_names = args.controlnet
        controlnets = []
        out = "Using"
        for name in args.controlnet:
            if name == "ip2p":
                controlnets.append(ControlNetModel.from_pretrained(CONTROLNET_IP2P_SOURCE, torch_dtype=torch.float16))
            elif name =="normal":
                controlnets.append(ControlNetModel.from_pretrained(CONTROLNET_NORMAL_SOURCE, torch_dtype=torch.float16))
            elif name =="depth":
                controlnets.append(ControlNetModel.from_pretrained(CONTROLNET_DEPTH_SOURCE, torch_dtype=torch.float16))
            out += " "+name
        print(out)

        self.nets = nn.ModuleList(controlnets).to(device)
        self.conditioning_scale = []
        for condition in args.conditioning_scale:
            self.conditioning_scale.append(condition)
        #self.conditioning_scale = [args.cfg_ip2p, args.cfg_normal]


        #self.nets = nn.ModuleList([normal_controlnet]).to(device)
        #self.conditioning_scale = [args.cfg_normal]
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained( SD_SOURCE, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16)
        #pipe = StableDiffusionControlNetPipeline.from_pretrained( SD_SOURCE, controlnet=[normal_controlnet], safety_checker=None, torch_dtype=torch.float16)
        #TODO check controlnet scheduler
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        pipe.scheduler.set_timesteps(100)
        """timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order."""
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()
        self.nets.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            pipe.unet.float()
            pipe.vae.float()
            self.nets.float()
        # else:
        #     if self.device.index:
        #         pipe.enable_model_cpu_offload(self.device.index)
        #     else:
        #         pipe.enable_model_cpu_offload(0)

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae


    def edit_image(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        rendered_image: Float[Tensor, "BS 3 H W"],
        gt_cond: Float[Tensor, "BS 3 H W"],
        normal_cond: Float[Tensor, "BS 3 H W"],
        depth_cond: Float[Tensor, "BS 3 H W"],
        heatmap: Float[Tensor, "BS 1 H W"],
        guidance_scale: float = 7.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98,
        noise = None,
        latent_weight: float = 0.9,
        threshold: float = 0.7
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        #mask = heatmap >= threshold  # Shape: (BS, 1, H, W)

        # Apply the mask to the rendered_image

        
        with torch.no_grad():
            # prepare image and image_cond latents
            #mask_broadcast = mask.expand_as(rendered_image).to(torch.float16)
            #masked_image = mask_broadcast * rendered_image + (1 - mask_broadcast) * gt_cond
            render_latents = self.imgs_to_latent(rendered_image)
            if latent_weight < 1.0:
                gt_latents = self.imgs_to_latent(gt_cond)
                #gt_latents = self.imgs_to_latent(normal_cond)
                #mask = F.interpolate(heatmap, size=(render_latents.shape[2], render_latents.shape[3]), mode='bilinear', align_corners=False).to(torch.float16)
                #latents = latent_weight * mask + (1 - mask) * gt_latents
                latents = latent_weight * render_latents + (1 - latent_weight) * gt_latents
            else:
                latents = render_latents
            #gt_latents = self.prepare_image_latents(image_cond)
            #image_cond_latents = self.prepare_image_latents(image_cond)
            #latents = render_latents
            #latents = gt_latents
            # add noise
        if noise == None:
            noise = torch.randn_like(latents)      
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore
        #latents = noise
        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        controlnet_cond = []
        for name in self.controlnet_names: #TODO change input gt_cond, normal_cond to conds, make this more abstract
            if name == "ip2p":
                #rendered_image = self.mask_image(heatmap, rendered_image, threshold) if heatmap!=None else rendered_image
                controlnet_cond.append(rendered_image)
            elif name == "normal":
                #normal_cond = self.mask_image(heatmap, normal_cond, threshold) if heatmap!=None else normal_cond
                controlnet_cond.append(normal_cond)
            elif name == "depth" and depth_cond!= None: #TODO fix this
                #depth_cond = self.mask_image(heatmap, depth_cond, threshold) if heatmap!=None else depth_cond
                controlnet_cond.append(depth_cond)
        
        #controlnet_cond = [rendered_image, normal_cond]
        #controlnet_cond = [normal_cond]
        
        for i, t in enumerate(self.scheduler.timesteps):

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 2)
                #latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                for k, (image, scale, controlnet) in enumerate(zip(controlnet_cond, self.conditioning_scale, self.nets)):
                    down_samples, mid_sample = controlnet(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=image,
                        conditioning_scale=scale,
                        return_dict=False
                    )

                    # merge samples
                    if k == 0:
                        down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
                    else:
                        down_block_res_samples = [
                            samples_prev + samples_curr
                            for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                        ]
                        mid_block_res_sample += mid_sample
                
   
                noise_pred = self.unet(latent_model_input, 
                                       t, 
                                       encoder_hidden_states=text_embeddings,
                                        down_block_additional_residuals = down_block_res_samples,
                                        mid_block_additional_residual=mid_block_res_sample,
                                        return_dict = False
                )[0]
                #noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        return decoded_img

    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def mask_image(self, heatmap, img, threshold):
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        mask = heatmap >= threshold  # Shape: (BS, 1, H, W)

        # Broadcast the mask to match the shape of rendered_image
        mask_broadcast = mask.expand_as(img)  # Shape: (BS, 3, H, W)

        # Apply the mask to the rendered_image
        masked_img = img * mask_broadcast
        return masked_img
    
    def apply_mask_to_latent(latent, mask):
        """
        Apply a binary mask to the latent tensor.

        Args:
            latent (torch.Tensor): The latent tensor of shape (BS, 4, H_latent, W_latent).
            mask (torch.Tensor): The binary mask tensor of shape (BS, 1, H_mask, W_mask).

        Returns:
            torch.Tensor: The masked latent tensor of shape (BS, 4, H_latent, W_latent).
        """
        # Ensure mask is of shape (BS, 1, H_mask, W_mask)
        assert mask.shape[1] == 1, "Mask should have a single channel"

        # Resize mask to match the spatial dimensions of the latent tensor
        mask_resized = F.interpolate(mask, size=(latent.shape[2], latent.shape[3]), mode='bilinear', align_corners=False)

        # Expand mask to match the shape of the latent tensor
        mask_expanded = mask_resized.expand(latent.shape[0], latent.shape[1], latent.shape[2], latent.shape[3])

        # Apply the mask to the latent tensor
        masked_latent = latent * mask_expanded

        return masked_latent

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError

class ControlNet(nn.Module):
    """ControlNet implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=False, args = None) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        
        self.controlnet = ControlNetModel.from_pretrained(CONTROLNET_IP2P_SOURCE, torch_dtype=torch.float16)

        pipe = StableDiffusionControlNetPipeline.from_pretrained( SD_SOURCE, controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16)
        #TODO check controlnet scheduler
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()
        self.controlnet.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            pipe.unet.float()
            pipe.vae.float()
            self.controlnet.float()
        else:
            if self.device.index:
                pipe.enable_model_cpu_offload(self.device.index)
            else:
                pipe.enable_model_cpu_offload(0)

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae


    def edit_image(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        rendered_image: Float[Tensor, "BS 3 H W"],
        original_image: Float[Tensor, "BS 3 H W"],
        normal_image: Float[Tensor, "BS 3 H W"],
        guidance_scale: float = 7.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98,
        noise = None
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            render_latents = self.imgs_to_latent(rendered_image)
            #gt_latents = self.imgs_to_latent(image_cond)
            #image_cond_latents = self.prepare_image_latents(image_cond)
            #latents = 0.8 * render_latents + 0.2 * gt_latents
            latents = render_latents

        # add noise
        noise = torch.randn_like(latents)
        #latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore
        #latents = noise
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore
        condition = original_image

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 2)
                

                #print("image_cond_latents shape: ",image_cond_latents.shape)
                down_block_res_samples, mid_block_res_sample  = self.controlnet(latent_model_input, 
                                                                               t, 
                                                                               encoder_hidden_states=text_embeddings, 
                                                                               controlnet_cond = condition,
                                                                               conditioning_scale = 1.0,
                                                                               return_dict = False)
                
                
                #latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                noise_pred = self.unet(latent_model_input, 
                                       t, 
                                       encoder_hidden_states=text_embeddings,
                                        down_block_additional_residuals = down_block_res_samples,
                                        mid_block_additional_residual=mid_block_res_sample,
                                        return_dict = False
                )[0]
                
                #noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


            # noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            # noise_pred = (
            #     noise_pred_uncond
            #     + guidance_scale * (noise_pred_text - noise_pred_image)
            #     + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            # )

            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        return decoded_img

    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError

class InstructPix2Pix(nn.Module):
    """InstructPix2Pix implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=False) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            pipe.unet.float()
            pipe.vae.float()
        else:
            if self.device.index:
                pipe.enable_model_cpu_offload(self.device.index)
            else:
                pipe.enable_model_cpu_offload(0)

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae


    def edit_image(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        image: Float[Tensor, "BS 3 H W"],
        image_cond: Float[Tensor, "BS 3 H W"],
        normal: Float[Tensor, "BS 3 H W"],
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            rendered_latents = self.imgs_to_latent(image)
            normal_latents = self.imgs_to_latent(normal)
            image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        noise = torch.randn_like(rendered_latents)
        
        # Normalize normal latets [0,1] (because of torch rand_like is [0,1)]
        #normal_latents = (normal_latents - torch.min(normal_latents)) / (torch.max(normal_latents) - torch.min(normal_latents))
        #noise = normal_latents
        #normalize noise
        #noise = (noise - torch.min(noise)) / (torch.max(noise) - torch.min(noise))
        #noise = normal_latents
        latents = self.scheduler.add_noise(rendered_latents, noise, self.scheduler.timesteps[0])  # type: ignore

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        return decoded_img

    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError