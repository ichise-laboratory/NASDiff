# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : TimeSeriesImputation
# @File          : architecture.py
# @Time          : 2024/9/18 16:46
# @Author        : SY.M
# @Software      : PyCharm
from math import floor
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn

from model.NASDiff.denoising_network import Denoising_Network, Frequency_Decomposition, LowRank_Reconstrution

from pypots.nn.functional import calc_mae



class NASDiff(nn.Module):
    def __init__(self,
                 # internal modules hyper-parameters
                 d_embedding: int,
                 d_hidden: int,
                 dropout: float,
                 qkv: int,
                 h: int,
                 N: int,
                 d_feature: int,
                 seq_len: int,
                 low_dim: int,
                 flimit: float,
                 mask: bool = True,
                 pe: bool = True,
                 fre_decom: str = 'slice',
                 fixed_last_step: Union[bool, int] = False,
                 betas: tuple = (1e-4, 0.02),
                 n_T: int = 10,
                 trajectory_len: int = 5,
                 trajectory_num: int = 1,
                 trajectory_sample_num: int = 10,
                 consistency: bool = False,
                 lowrank: bool = False,
                 frequency: bool = False,
                 shortcut: bool = False,
                 last_step: str = None,
                 model_num: str = None,) -> object:
        super(NASDiff, self).__init__()

        if d_feature > 1:
            low_dim = floor(low_dim * d_feature)
            if low_dim < 3:
                low_dim = 3
        else:
            low_dim = 1


        self.lowrank_recon = LowRank_Reconstrution(low_dim=low_dim,
                                                     second_dim=seq_len,
                                                  third_dim=d_feature,
                                                  d_embedding=d_embedding,
                                                  d_hidden=d_hidden,
                                                  q=qkv, k=qkv, v=qkv, h=h,
                                                  mask=mask,
                                                  dropout=dropout)

        self.frequency_decom = Frequency_Decomposition(second_dim=seq_len, dropout=dropout, flimit=flimit, fre_decom=fre_decom)

        self.denoising = Denoising_Network(seq_len=seq_len,
                                            d_feature=d_feature,
                                            d_embedding=d_embedding,
                                            d_hidden=d_hidden,
                                            d_output=d_feature,
                                            dropout=dropout,
                                            q=qkv,
                                            k=qkv,
                                            v=qkv,
                                            h=h,
                                            N=N,
                                            n_T=n_T,
                                            mask=mask,
                                            pe=pe)

        self.num_feature = d_feature
        self.seq_len = seq_len
        self.size_T = (seq_len, d_feature)
        self.size_F = (d_feature, seq_len)
        assert model_num in ['art_only', 'default'], 'model_num error!'
        self.model_num = model_num
        self.last_step = last_step
        self.fixed_last_step = fixed_last_step

        for k, v in self.ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        # Paper-facing names; the original implementation names are kept as aliases below.
        self.diffusion_steps = n_T
        self.short_trajectory_length = trajectory_len
        self.train_trajectory_count = trajectory_num
        self.inference_trajectory_count = trajectory_sample_num
        self.n_T = self.diffusion_steps
        self.trajectory_len = self.short_trajectory_length
        self.trajectory_num = self.train_trajectory_count
        self.trajectory_sample_num = self.inference_trajectory_count
        self.consistency = consistency
        self.lowrank = lowrank
        self.frequency = frequency
        self.fre_decom = fre_decom
        self.shortcut = shortcut
        self.loss_mse = nn.MSELoss()

    @staticmethod
    def ddpm_schedules(beta1, beta2, T):
        '''
        Precompute alpha terms for each diffusion step with a linear beta schedule.
        :param beta1: lower beta bound
        :param beta2: upper beta bound
        :param T: total number of diffusion steps
        '''
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    @staticmethod
    def _combine_frequency_components(base, high=None, overlap=None):
        """Recompose low-frequency output with high-frequency residual and slice overlap."""
        output = base
        if high is not None:
            while high.dim() < output.dim():
                high = high.unsqueeze(1)
            output = output + high
        if overlap is not None:
            while overlap.dim() < output.dim():
                overlap = overlap.unsqueeze(1)
            output = output - overlap
        return output

    def diffusion_forward(self,
                          x_expect,
                          condition_mask,
                          condition_replace,
                          _ts=None,
                          model_num=None,
                          noise=None):
        if _ts is None:
            _ts = torch.randint(1, self.n_T + 1, (x_expect.shape[0],)).to(x_expect.device)  # t ~ Uniform(0, n_T)

        if noise is None:
            noise = torch.randn_like(x_expect)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None] * x_expect
                + self.sqrtmab[_ts, None, None] * noise
        )

        if model_num == 'art_only':
            x_t = torch.where(condition_mask == 1, condition_replace, x_t)

        return x_t, noise, _ts

    def forward(self,
                 inputs: dict,
                 stage: str = 'train',):

        assert stage in ['train', 'val', 'test'], f'Require "stage" in ["train", "val", "test"] but got {stage}!'

        (X_T,
         condition_label_T,
         condition_x_T,
         condition_mask_T) = (inputs['X'],
                         inputs['X'],
                         inputs['X'],
                         inputs['missing_mask'])

        output: [str, torch.Tensor] = dict(
            noise=None,
            noise_pred=None,
            generation=None,
            imputed_result=None,)

        origin_high_f = None
        overlap = None
        lowrank_x = X_T
        if self.frequency:
            if self.fre_decom == 'slice':
                lowrank_x, origin_high_f, overlap = self.frequency_decom(x=X_T, x_intact=inputs['X_intact'],)
            elif self.fre_decom == 'fixed':
                lowrank_x, origin_high_f = self.frequency_decom(x=X_T, x_intact=inputs['X_intact'],)
            elif self.fre_decom == 'all':
                lowrank_x = self.frequency_decom(x=X_T, x_intact=inputs['X_intact'],)


        if self.lowrank:
            lowrank_x = self.lowrank_recon(X=lowrank_x)

        noise, noise_pred, imputed_result = self.forward_process(
            X=lowrank_x,
            condition_x=condition_x_T,
            condition_mask=condition_mask_T,
            condition_replace=condition_x_T,
            consistency=self.consistency,
        )

        output['noise'] = noise
        output['noise_pred'] = noise_pred

        if self.consistency:
            if self.shortcut:
                if self.frequency:
                    output['generation'] = self._combine_frequency_components(
                        imputed_result,
                        high=origin_high_f,
                        overlap=overlap,
                    )
                    output['imputed_result'] = self._combine_frequency_components(
                        torch.mean(imputed_result, dim=1, keepdim=False),
                        high=origin_high_f,
                        overlap=overlap,
                    )
                else:
                    output['generation'] = imputed_result
                    output['imputed_result'] = torch.mean(imputed_result, dim=1, keepdim=False)
            else:
                if self.frequency:
                    output['generation'] = self._combine_frequency_components(
                        imputed_result,
                        high=origin_high_f,
                        overlap=overlap,
                    )
                    output['imputed_result'] = self._combine_frequency_components(
                        imputed_result,
                        high=origin_high_f,
                        overlap=overlap,
                    )
                else:
                    output['generation'] = imputed_result
                    output['imputed_result'] = imputed_result


        return output

    def forward_process(self,
                        X,
                        condition_x,
                        condition_mask,
                        condition_replace,
        consistency=False,
                        ):
        """
        Sample diffusion steps and noise during training.
        """

        x_t, noise, _ts = self.diffusion_forward(x_expect=X,
                                                 condition_mask=condition_mask,
                                                 model_num=self.model_num,
                                                 condition_replace=condition_replace,
                                                )

        noise_pred = self.denoising(X=x_t, t=_ts)

        sampled_x_0 = None
        if consistency:
            # sampling
            sampled_x_0 = self.sample_for_impute(X=X,
                                                  condition_x=condition_x,
                                                  condition_mask=condition_mask,
                                                  condition_replace=condition_replace,)

        return noise, noise_pred, sampled_x_0

    def sample_for_impute(self,
                          X,
                          condition_x,
                          condition_mask,
                          condition_replace,
                          n_sample=1):

        self.denoising.eval()

        B = condition_x.shape[0]
        size = condition_mask[0].shape

        if self.shortcut:
            generated_trajectory = self.random_trajectory(B, sample_num=n_sample, trajectory_num=self.trajectory_num).to(condition_x.device)  # [B * n_sample, n_T]
        else:
            generated_trajectory = torch.arange(self.n_T, -1, -1)

        # generate x_T and conditional
        if self.shortcut:
            x_i = torch.randn(B * n_sample * self.trajectory_num, *size).float().to(condition_x.device)
            condition_x = condition_x.repeat_interleave(n_sample * self.trajectory_num, dim=0)
            condition_mask = condition_mask.repeat_interleave(n_sample * self.trajectory_num, dim=0)
            condition_replace = condition_replace.repeat_interleave(n_sample * self.trajectory_num, dim=0)
        else:
            x_i = torch.randn(B * n_sample, *size).float().to(condition_x.device)
            condition_x = condition_x.repeat_interleave(n_sample, dim=0)
            condition_mask = condition_mask.repeat_interleave(n_sample, dim=0)
        device = condition_x.device

        if self.shortcut:
            for i in range(self.trajectory_len):
                t_is = generated_trajectory[:, i].to(torch.int32)

                if i != self.trajectory_len - 1:
                    next_step = generated_trajectory[:, i + 1].to(torch.int32)
                    eta = 0.1
                    bar_alpha_t = self.alphabar_t[t_is]
                    bar_alpha_tm1 = self.alphabar_t[next_step]
                    assert torch.all(t_is > next_step), \
                        f"Invalid alpha order: t {t_is}, alpha_tm1 {bar_alpha_tm1}, alpha_t {bar_alpha_t}" \
                        f"Invalid alpha order: t {next_step}, alpha_tm1 {bar_alpha_tm1}, alpha_t {bar_alpha_t}"
                    sigma_t = eta * torch.sqrt((1 - bar_alpha_tm1) / (1 - bar_alpha_t)) * torch.sqrt(
                        1 - bar_alpha_t / bar_alpha_tm1)
                    z_2 = torch.randn(self.trajectory_num * B * n_sample, *size).to(device) * (t_is > 1)[:, None, None]

                    x_i = torch.where(condition_mask == 1, condition_replace, x_i)
                    eps_2 = self.denoising(X=x_i, t=t_is)

                    pred_x0 = (x_i - self.sqrtmab[t_is, None, None] * eps_2) / self.sqrtab[t_is, None, None]
                    # with random noise
                    # alpha_bar_next = self.alphabar_t[next_step]  # ᾱ_{t'}
                    # sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
                    # coeff_eps = torch.sqrt(1 - alpha_bar_next - sigma_t ** 2)
                    # x_i = (
                    #         sqrt_alpha_bar_next[:, None, None] * pred_x0
                    #         + coeff_eps[:, None, None] * eps_2
                    #         + sigma_t[:, None, None] * z_2
                    # )
                    ## x_i = self.sqrtab[next_step][:, None, None] * pred_x0 + self.sqrtmab[next_step][:, None, None] * eps_2 + sigma_t[:, None, None] * z_2
                    # No random noise
                    x_i = self.sqrtab[next_step][:, None, None] * pred_x0 + self.sqrtmab[next_step][:, None, None] * eps_2
                else:  # the last step before 0

                    x_i = torch.where(condition_mask == 1, condition_replace, x_i)
                    eps_2 = self.denoising(X=x_i, t=t_is)

                    if self.last_step == 'DDPM':
                        x_i = self.oneover_sqrta[t_is] * (x_i - eps_2 * self.mab_over_sqrtmab[t_is])
                    elif self.last_step == 'DDIM':
                        x_i = (x_i - self.sqrtmab[t_is, None, None] * eps_2) / self.sqrtab[t_is, None, None]
        else:
            # x_i = torch.randn(n_sample * B, *size).float().to(device)
            inter_states = [x_i.reshape(B, -1, *size), ]  # item shape [B, n_sampling_time, T, F]

            for i in range(self.n_T, 0, -1):
                t_is = torch.tensor([i]).repeat(n_sample * B, 1, 1).to(device)

                # x_i = condition_mask * condition_replace + (1 - condition_mask) * x_i


                z_2 = torch.randn(n_sample * B, *size).to(device) if i > 1 else 0

                # eps_2 = module(X=x_i, t=t_is)
                if self.consistency:
                    x_i = torch.where(condition_mask == 1, condition_replace, x_i)
                    eps_2 = self.denoising(X=x_i, t=t_is)

                x_i = self.oneover_sqrta[i] * (x_i - eps_2 * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z_2

                inter_states.append(x_i.reshape(B, -1, *size))

        if self.shortcut:
            x_i = torch.mean(x_i.reshape(B, n_sample, self.trajectory_num, *size), dim=1, keepdim=False)
        else:
            x_i = torch.mean(x_i.reshape(B, n_sample, *size), dim=1, keepdim=False)
        self.denoising.train()

        return x_i

    def predict(self,
               inputs,
               used_module='TF1',
               trajectory_sample_num=1,
               **kwargs):

        # sample times during inference
        n_sample = kwargs.get('test_sampling_times', 1)
        # get shape and device
        device = inputs['X'].device
        B = inputs['X'].shape[0]
        size = inputs['missing_mask'].shape[1:]
        X_T = inputs['X']
        condition_mask = inputs['missing_mask']
        condition_x = inputs['X']

        self.denoising.eval()
        self.frequency_decom.eval()
        self.lowrank_recon.eval()

        trajectory_sample_num = self.trajectory_sample_num

        origin_high_f = None
        overlap = None
        lowrank_x = condition_x
        if self.frequency:
            if self.fre_decom == 'slice':
                lowrank_x, origin_high_f, overlap = self.frequency_decom(x=condition_x, x_intact=inputs['X_intact'], )
            elif self.fre_decom == 'fixed':
                lowrank_x, origin_high_f = self.frequency_decom(x=condition_x, x_intact=inputs['X_intact'], )
            elif self.fre_decom == 'all':
                lowrank_x = self.frequency_decom(x=condition_x, x_intact=inputs['X_intact'], )
        # if self.lowrank:
        #     lowrank_x = self.lowrank_recon(X=lowrank_x)

        condition = condition_x

        # construct inputs
        if self.shortcut:
            X_T = inputs['X'].to(device).repeat_interleave(n_sample * trajectory_sample_num, dim=0)
            condition_mask = inputs['missing_mask'].repeat_interleave(n_sample * trajectory_sample_num, dim=0)
            condition_x = condition.repeat_interleave(n_sample * trajectory_sample_num, dim=0)
        else:
            X_T = inputs['X'].to(device).repeat_interleave(n_sample, dim=0)
            condition_mask = inputs['missing_mask'].repeat_interleave(n_sample, dim=0)
            condition_x = condition.repeat_interleave(n_sample, dim=0)

        with torch.no_grad():
            # initial X_T
            x_i = torch.randn(B * n_sample * trajectory_sample_num, *size).float().to(condition_x.device)

            if self.shortcut:
                # generate random trajectory
                generated_trajectory = self.random_trajectory(B, sample_num=n_sample, trajectory_num=trajectory_sample_num).to(device)
                # generated_trajectory = torch.linspace(self.n_T, 0, self.trajectory_len, dtype=torch.long,).to(device).unsqueeze(0).expand(B * n_sample * trajectory_sample_num, -1)
                # generated_trajectory = torch.Tensor([10, 7, 3, 2, 1, 0]).long().unsqueeze(0).expand(B * n_sample * trajectory_sample_num, -1).to(device)
            else:
                generated_trajectory = torch.arange(self.n_T, -1, -1)

            if self.shortcut:
                for i in range(self.trajectory_len):
                    t_is = generated_trajectory[:, i].to(torch.int32)

                    if i != self.trajectory_len - 1:
                        next_step = generated_trajectory[:, i + 1].to(torch.int32)
                        eta = 0.1
                        bar_alpha_t = self.alphabar_t[t_is]
                        bar_alpha_tm1 = self.alphabar_t[next_step]
                        sigma_t = eta * torch.sqrt((1 - bar_alpha_tm1) / (1 - bar_alpha_t)) * torch.sqrt(
                            1 - bar_alpha_t / bar_alpha_tm1)
                        z_2 = torch.randn(n_sample * B * trajectory_sample_num, *size).to(condition_x.device) * (t_is > 1)[:, None, None]
                        x_i = torch.where(condition_mask == 1, condition_x, x_i)
                        eps_2 = self.denoising(X=x_i, t=t_is)
                        pred_x0 = (x_i - self.sqrtmab[t_is, None, None] * eps_2) / self.sqrtab[t_is, None, None]

                        # with random noise
                        # x_i = self.sqrtab[next_step][:, None, None] * pred_x0 + self.sqrtmab[next_step][:, None, None] * eps_2 + sigma_t[:, None, None] * z_2

                        # with no random noise
                        x_i = self.sqrtab[next_step][:, None, None] * pred_x0 + self.sqrtmab[next_step][:, None, None] * eps_2
                    else:  # the last step before 0

                        x_i = torch.where(condition_mask == 1, condition_x, x_i)
                        eps_2 = self.denoising(X=x_i, t=t_is)

                        if self.last_step == 'DDPM':
                            x_i = self.oneover_sqrta[t_is] * (x_i - eps_2 * self.mab_over_sqrtmab[t_is])
                        elif self.last_step == 'DDIM':
                            x_i = (x_i - self.sqrtmab[t_is, None, None] * eps_2) / self.sqrtab[t_is, None, None]
            else:
                inter_states = [x_i.reshape(B, -1, *size), ]  # item shape [B, n_sampling_time, T, F]

                for i in range(self.n_T, 0, -1):
                    t_is = torch.tensor([i]).repeat(n_sample * B, 1, 1).to(device)

                    # x_i = condition_mask * condition_x + (1 - condition_mask) * x_i

                    z_2 = torch.randn(n_sample * B, *size).to(device) if i > 1 else 0

                    # eps_2 = module(X=x_i, t=t_is)
                    x_i = torch.where(condition_mask == 1, condition_x, x_i)
                    eps_2 = self.denoising(X=x_i, t=t_is)

                    x_i = self.oneover_sqrta[i] * (x_i - eps_2 * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z_2

                    inter_states.append(x_i.reshape(B, -1, *size))

            imputed_n_samples = x_i.reshape(B, -1, *size)  # [B, sample_num, L, F]
            x_i = torch.mean(imputed_n_samples, dim=1, keepdim=False)  # [B, L, F]

            if self.frequency:
                x_i = self._combine_frequency_components(x_i, high=origin_high_f, overlap=overlap)
                imputed_n_samples = self._combine_frequency_components(
                    imputed_n_samples,
                    high=origin_high_f,
                    overlap=overlap,
                )
        self.denoising.train()
        self.frequency_decom.train()
        self.lowrank_recon.train()

        output = dict(imputed_result=x_i,
                      imputed_n_samples=imputed_n_samples)

        return output

    def random_trajectory(self, batch, sample_num, trajectory_num):
        trajectories = []
        if isinstance(self.fixed_last_step, bool) and self.fixed_last_step == False:  # no settings to last step, i.e. randomly
            for i in range(batch):
                    for k in range(trajectory_num):
                        idx = np.random.choice(
                            range(1, self.n_T),
                            self.trajectory_len - 1,
                            replace=False
                        )
                        trajectory = np.sort(idx)[::-1].copy()
                        trajectories.append(torch.tensor(trajectory, dtype=torch.float32))

        elif isinstance(self.fixed_last_step, int):  # fixed last step, generally 1
            for i in range(batch):
                    for k in range(trajectory_num):
                        idx = np.random.choice(
                            range(2, self.n_T),
                            self.trajectory_len - 2,
                            replace=False
                        )
                        trajectory = np.sort(idx)[::-1].copy()
                        trajectories.append(torch.tensor(trajectory, dtype=torch.float32))

        # [batch * sample_num * trajectory_num, trajectory_len]
        trajectories = torch.stack(trajectories, dim=0)
        trajectories = trajectories.reshape(batch, trajectory_num, -1)
        trajectories = trajectories.unsqueeze(1).repeat(1, sample_num, 1, 1)
        trajectories = trajectories.reshape(batch * sample_num * trajectory_num, -1)

        col_n_T = torch.full((trajectories.size(0), 1), self.n_T, dtype=trajectories.dtype)

        if isinstance(self.fixed_last_step, bool):
            trajectories = torch.cat([col_n_T, trajectories], dim=1)  # (batch*trajectory_num, trajectory_len+1)
        elif isinstance(self.fixed_last_step, int):
            col_1 = torch.full((trajectories.size(0), 1), self.fixed_last_step, dtype=trajectories.dtype)
            trajectories = torch.cat([col_n_T, trajectories, col_1], dim=1)  # (batch*trajectory_num, trajectory_len+1)

        return trajectories


    def loss_func(self,
                      outputs: dict,
                      inputs: dict,
                      calc_func: Callable = calc_mae,
                      ) -> torch.Tensor:
        (noise,
         noise_pred,
         imputed_result,
         ) = (outputs['noise'],
                outputs['noise_pred'],
                outputs['generation'],
                )

        if self.shortcut:
            (X_intact_T,
             X_T,
             missing_mask_T,
             indicating_mask_T) = (inputs['X_intact'].unsqueeze(1).expand(-1, self.trajectory_num, -1, -1),
                                   inputs['X'].unsqueeze(1).expand(-1, self.trajectory_num, -1, -1),
                                   inputs['missing_mask'].unsqueeze(1).expand(-1, self.trajectory_num, -1, -1),
                                   inputs['indicating_mask'].unsqueeze(1).expand(-1, self.trajectory_num, -1, -1))
        else:
            (X_intact_T,
             X_T,
             missing_mask_T,
             indicating_mask_T) = (inputs['X_intact'],
                                   inputs['X'],
                                   inputs['missing_mask'],
                                   inputs['indicating_mask'])
            self.trajectory_num = 1


        ones = torch.ones_like(noise).to(X_T.device)
        noise_loss = calc_func(noise_pred, noise, ones)
        integrated_loss = noise_loss

        if self.consistency:
            ORT_loss = calc_func(imputed_result, X_T, missing_mask_T)
            MIT_loss = calc_func(imputed_result, X_intact_T,  indicating_mask_T)

            integrated_loss += MIT_loss + ORT_loss

        return integrated_loss
