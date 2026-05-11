from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.utils import is_torch_xla_available
from energies import reconstruction_energy_latent


RF_INVERSION_ROOT = "/home/Wu_25R8111/RF-Inversion"
if RF_INVERSION_ROOT not in sys.path:
    sys.path.insert(0, RF_INVERSION_ROOT)

from pipeline_flux_rf_inversion import RFInversionFluxPipeline  # noqa: E402


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class HRecRFInversionPipeline(RFInversionFluxPipeline):
    @staticmethod
    def predict_x0_from_linear_rf_path(
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        t_scalar: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """
        Path-based clean estimate under the linear RF path x_t = (1 - t) x_0 + t x_1.

        If v_theta(x_t, t) approximates the path derivative x_1 - x_0, then:
            x_hat_0(x_t, t) = x_t - t * v_theta(x_t, t)
        """
        if not torch.is_tensor(t_scalar):
            t_scalar = torch.tensor(t_scalar, device=x_t.device, dtype=x_t.dtype)
        t_scalar = t_scalar.to(device=x_t.device, dtype=x_t.dtype)
        while t_scalar.ndim < x_t.ndim:
            t_scalar = t_scalar.view(*t_scalar.shape, 1)
        return x_t - t_scalar * v_t

    @torch.no_grad()
    def invert(
        self,
        image,
        source_prompt: str = "",
        source_guidance_scale: float = 0.0,
        num_inversion_steps: int = 28,
        strength: float = 1.0,
        gamma: float = 0.5,
        return_source_trajectory: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        timesteps: List[int] = None,
        dtype: Optional[torch.dtype] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dtype = dtype or self.text_encoder.dtype
        batch_size = 1
        self._joint_attention_kwargs = joint_attention_kwargs
        num_channels_latents = self.transformer.config.in_channels // 4

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        device = self._execution_device

        image_latents, _ = self.encode_image(image, height=height, width=width, dtype=dtype)
        image_latents, latent_image_ids = self.prepare_latents_inversion(
            batch_size, num_channels_latents, height, width, dtype, device, image_latents
        )

        sigmas = np.linspace(1.0, 1 / num_inversion_steps, num_inversion_steps)
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inversion_steps = retrieve_timesteps(
            self.scheduler,
            num_inversion_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        timesteps, sigmas, num_inversion_steps = self.get_timesteps(num_inversion_steps, strength)

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=source_prompt,
            prompt_2=source_prompt,
            device=device,
        )

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], source_guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None

        y_t = image_latents
        y_1 = torch.randn_like(y_t)
        n_steps = len(sigmas)
        source_trajectory = [y_t.clone()] if return_source_trajectory else None

        with self.progress_bar(total=n_steps - 1) as progress_bar:
            for i in range(n_steps - 1):
                t_i = torch.tensor(i / n_steps, dtype=y_t.dtype, device=device)
                timestep = torch.tensor(t_i, dtype=y_t.dtype, device=device).repeat(batch_size)

                u_t_i = self.transformer(
                    hidden_states=y_t,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                u_t_i_cond = (y_1 - y_t) / (1 - t_i)
                u_hat_t_i = u_t_i + gamma * (u_t_i_cond - u_t_i)
                y_t = y_t + u_hat_t_i * (sigmas[i] - sigmas[i + 1])

                if return_source_trajectory:
                    source_trajectory.append(y_t.clone())
                progress_bar.update()

        if return_source_trajectory:
            return y_t, image_latents, latent_image_ids, torch.stack(source_trajectory, dim=0)
        return y_t, image_latents, latent_image_ids

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        inverted_latents: Optional[torch.FloatTensor] = None,
        image_latents: Optional[torch.FloatTensor] = None,
        latent_image_ids: Optional[torch.FloatTensor] = None,
        source_trajectory: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 1.0,
        decay_eta: bool = False,
        eta_decay_power: float = 1.0,
        rec_guidance_scale: float = 0.0,
        strength: float = 1.0,
        start_timestep: float = 0,
        stop_timestep: float = 0.25,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        enable_sde: bool = True,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            prompt_2,
            inverted_latents,
            image_latents,
            latent_image_ids,
            height,
            width,
            start_timestep,
            stop_timestep,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        do_rf_inversion = inverted_latents is not None

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        if do_rf_inversion:
            latents = inverted_latents
        else:
            latents, latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        if do_rf_inversion:
            start_timestep = int(start_timestep * num_inference_steps)
            stop_timestep = min(int(stop_timestep * num_inference_steps), num_inference_steps)
            timesteps, sigmas, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if do_rf_inversion:
            y_0 = image_latents.clone()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            x0_pred = None
            rec_energy = None
            for i, t in enumerate(timesteps):
                if do_rf_inversion:
                    t_i = 1 - t / 1000

                if self.interrupt:
                    continue

                latents_dtype = latents.dtype
                if do_rf_inversion:
                    latents_for_grad = latents.detach().clone()
                    if rec_guidance_scale > 0.0:
                        latents_for_grad.requires_grad_(True)

                    timestep = t.expand(latents_for_grad.shape[0]).to(latents_for_grad.dtype)
                    with torch.enable_grad() if rec_guidance_scale > 0.0 else torch.no_grad():
                        noise_pred = self.transformer(
                            hidden_states=latents_for_grad,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]

                        v_t = -noise_pred
                        x0_pred = self.predict_x0_from_linear_rf_path(latents_for_grad, v_t, t_i)
                        rec_energy = reconstruction_energy_latent(x0_pred, y_0)

                    if rec_guidance_scale > 0.0:
                        grad_rec = torch.autograd.grad(rec_energy, latents_for_grad)[0]
                        grad_rec = grad_rec.detach().to(dtype=latents_dtype)
                    else:
                        grad_rec = torch.zeros_like(latents)

                    v_t_cond = (y_0 - latents) / (1 - t_i)
                    eta_t = eta if start_timestep <= i < stop_timestep else 0.0
                    if decay_eta:
                        eta_t = eta_t * (1 - i / num_inference_steps) ** eta_decay_power
                    v_edit_t = v_t + eta_t * (v_t_cond - v_t)

                    if source_trajectory is not None and rec_guidance_scale > 0.0:
                        ref_idx = min(i, source_trajectory.shape[0] - 1)
                        source_ref_t = source_trajectory[ref_idx].to(device=device, dtype=latents.dtype)
                        v_rec_t = -rec_guidance_scale * (latents - source_ref_t) - rec_guidance_scale * grad_rec
                    elif rec_guidance_scale > 0.0:
                        v_rec_t = -rec_guidance_scale * grad_rec
                    else:
                        v_rec_t = torch.zeros_like(latents)

                    v_hat_t = v_edit_t + v_rec_t

                    if not enable_sde:
                        latents = latents + v_hat_t * (sigmas[i] - sigmas[i + 1])
                    else:
                        if i == 0:
                            drift = v_hat_t
                            diffusion_coeff = 0
                        else:
                            drift = 2 * v_hat_t - latents / (1 - sigmas[i])
                            diffusion_coeff = (
                                2 * sigmas[i] / (1 - sigmas[i]) * (sigmas[i] - sigmas[i + 1])
                            ).sqrt()
                        latents = latents + (sigmas[i] - sigmas[i + 1]) * drift + diffusion_coeff * torch.randn_like(
                            latents
                        )
                else:
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
