import numpy as np
import torch
import torch.nn as nn

from pypots.nn.modules.csdi.layers import CsdiDiffusionModel


class CSDI(nn.Module):
    def __init__(
        self,
        n_feature,
        n_layers,
        n_heads,
        n_channels,
        d_target,
        d_time_embedding,
        d_feature_embedding,
        d_diffusion_embedding,
        is_unconditional,
        n_diffusion_steps,
        schedule,
        beta_start,
        beta_end,
        n_sampling_times,
    ):
        super().__init__()

        self.d_target = d_target
        self.d_time_embedding = d_time_embedding
        self.n_feature = n_feature
        self.d_feature_embedding = d_feature_embedding
        self.is_unconditional = is_unconditional
        self.n_channels = n_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.n_sampling_times = n_sampling_times

        d_side = d_time_embedding + d_feature_embedding
        if self.is_unconditional:
            d_input = 1
        else:
            d_side += 1  # for conditional mask
            d_input = 2

        self.diff_model = CsdiDiffusionModel(
            n_diffusion_steps,
            d_diffusion_embedding,
            d_input,
            d_side,
            n_channels,
            n_heads,
            n_layers,
        )

        self.embed_layer = nn.Embedding(
            num_embeddings=n_feature,
            embedding_dim=d_feature_embedding,
        )

        # parameters for diffusion models
        if schedule == "quad":
            self.beta = np.linspace(beta_start**0.5, beta_end**0.5, self.n_diffusion_steps) ** 2
        elif schedule == "linear":
            self.beta = np.linspace(beta_start, beta_end, self.n_diffusion_steps)
        else:
            raise ValueError(f"The argument schedule should be 'quad' or 'linear', but got {schedule}")

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer("alpha_torch", torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1))

    @staticmethod
    def time_embedding(pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(pos.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=pos.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        device = observed_tp.device
        time_embed = self.time_embedding(observed_tp, self.d_time_embedding)  # (B,L,emb)
        time_embed = time_embed.to(device)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(self.n_feature).to(device))  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,emb+d_feature_embedding)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def forward_val(self, inputs):

        (observed_data,
         cond_mask,
         indicating_mask,
         observed_tp) = (inputs['X_intact'],
                         inputs['missing_mask'],
                         inputs['indicating_mask'],
                         inputs['observed_tp'])
        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_sum = 0
        for t in range(self.n_diffusion_steps):  # calculate loss for all t
            predicted_noise = self.predict_noise(observed_data=observed_data,
                                       cond_mask=cond_mask,
                                       indicating_mask=indicating_mask,
                                       side_info=side_info,
                                                 set_t=t)
            predicted_noise['indicating_mask'] = indicating_mask
            loss = self.loss_func(outputs=predicted_noise, inputs=inputs)
            loss_sum += loss.detach().cpu().numpy()
        loss_sum /= self.n_diffusion_steps

        return dict(val_loss=loss_sum)

    def forward(self, inputs, set_t=-1):

        (observed_data,
         observed_mask,
         indicating_mask,
         observed_tp) = (inputs['X_intact'],
                         inputs['missing_mask'],
                         inputs['indicating_mask'],
                         inputs['observed_tp'])
        # Reviewer-release behavior: CSDI uses the fixed artificial missing mask
        # produced by data_processing, rather than dynamically sampling a new mask.
        cond_mask = observed_mask
        indicating_mask = indicating_mask

        # indicating_mask = observed_mask - cond_mask
        side_info = self.get_side_info(observed_tp, cond_mask)


        output = self.predict_noise(observed_data=observed_data,
                           cond_mask=cond_mask,
                           indicating_mask=indicating_mask,
                           side_info=side_info)
        output['indicating_mask'] = indicating_mask
        return output

    def predict_noise(self, observed_data, cond_mask, indicating_mask, side_info, set_t=-1):

        B, K, L = observed_data.shape
        device = observed_data.device
        if self.training != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(device)
        else:
            t = torch.randint(0, self.n_diffusion_steps, [B]).to(device)

        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diff_model(total_input, side_info, t)  # (B,K,L)

        output = dict(predicted_noise=predicted,
                      noise=noise,)

        return output

    def loss_func(self, outputs, inputs):

        (predicted_noise,
         indicating_mask,
         noise) = (outputs['predicted_noise'],
                   outputs['indicating_mask'],
                              outputs['noise'])

        # indicating_mask = inputs['indicating_mask']

        residual = (noise - predicted_noise) * indicating_mask
        num_eval = indicating_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

    def predict(self, inputs, **kwargs):
        stage = kwargs.get('stage')
        assert stage in ['test', 'val'], f"{stage} is not a valid stage"

        if stage == 'test':
            output = self.impute(inputs, **kwargs)
        elif stage == 'val':
            output = self.forward_val(inputs)

        return output

    def impute(self, inputs, **kwargs):

        (observed_data,
         cond_mask,
         observed_tp) = (inputs['X'],
                         inputs['missing_mask'],
                         inputs['observed_tp'])
        side_info = self.get_side_info(observed_tp, cond_mask)
        n_sampling_times = kwargs.get('test_sampling_times', 1)
        cond_mask = cond_mask.repeat_interleave(n_sampling_times, dim=0)
        side_info = side_info.repeat_interleave(n_sampling_times, dim=0)

        B, K, L = observed_data.shape
        device = observed_data.device
        imputed_samples = torch.zeros(B, n_sampling_times, K, L).to(device)

        observed_data = observed_data.repeat_interleave(n_sampling_times, dim=0)

        self.diff_model.eval()
        self.embed_layer.eval()
        with torch.no_grad():
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.n_diffusion_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)
            if 'fixed_noise' in inputs.keys():
                current_sample = inputs['fixed_noise'][:, :, :].to(device)
            else:
                current_sample = torch.randn(B * n_sampling_times, K, L).float().to(device)

            for t in range(self.n_diffusion_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diff_model(diff_input, side_info, torch.tensor([t]).to(device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples = current_sample.reshape(B, n_sampling_times, K, L)

            output = dict(imputed_result=torch.mean(imputed_samples, dim=1, keepdim=False),  # [batch, F, T]
                          imputed_n_samples=imputed_samples,  # [batch, n_sample, F, T]
                          )

        return output
