import torch
from torch import nn
import math

class Denoising_Network(nn.Module):
    def __init__(self,
                 seq_len: int,
                 d_feature: int,
                 d_output: int,
                 d_embedding: int = 512,
                 d_hidden: int = 2048,
                 dropout: float = 0.1,
                 q: int = 64,
                 k: int = 64,
                 v: int = 64,
                 h: int = 8,
                 N: int = 4,
                 n_T: int = 10,
                 pe=True,
                 mask=True):
        super(Denoising_Network, self).__init__()
        self.n_T = n_T
        self.d_feature = d_feature
        self.seq_len = seq_len

        self.norm = nn.BatchNorm1d(num_features=seq_len)
        self.input_layer_T = Input_Layer(second_dim=seq_len, third_dim=d_feature, pe=pe,
                                         d_embedding=d_embedding, dropout=dropout)
        self.input_layer_F = Input_Layer(second_dim=d_feature, third_dim=seq_len, pe=pe,
                                         d_embedding=d_embedding, dropout=dropout)

        self.encoder_stack_T = nn.ModuleList([Encoder(d_embedding=d_embedding,
                                                      d_hidden=d_hidden,
                                                      dropout=dropout,
                                                      q=q, k=k, v=v, h=h, mask=mask) for _ in range(N)])

        self.encoder_stack_F = nn.ModuleList([Encoder(d_embedding=d_embedding,
                                                      d_hidden=d_hidden,
                                                      dropout=dropout,
                                                      q=q, k=k, v=v, h=h, mask=mask) for _ in range(N)])

        self.output_layer = Output_layer_Fuse(d_feature=d_feature,
                                              seq_len=seq_len,
                                              d_output=d_output,
                                              d_embedding=d_embedding,
                                              d_hidden=d_hidden,
                                              )

    def forward(self, X, t,):
        X_F = torch.cat(torch.split(X, split_size_or_sections=self.d_feature, dim=-1), dim=1).transpose(-1, -2)

        X_T = self.input_layer_T(X)
        X_F = self.input_layer_F(X_F)

        for encoder in self.encoder_stack_F:
            X_F, score_F = encoder(query=X_F, key=X_F, value=X_F)
        for encoder in self.encoder_stack_T:
            X_T, score_T = encoder(query=X_T, key=X_T, value=X_T)

        X = self.output_layer(X_T=X_T, X_F=X_F, ts=t / self.n_T)

        return X


class Output_layer_Fuse(nn.Module):
    def __init__(self,
                 d_feature: int,
                 seq_len: int,
                 d_output: int,
                 d_embedding: int = 512,
                 d_hidden: int = 2048,):
        super().__init__()

        self.d_output = d_output

        self.timeembed1 = EmbedFC(1, d_output)

        self.output_linear_T = torch.nn.Sequential(
            nn.Linear(in_features=d_embedding, out_features=d_feature),
        )
        self.output_linear_F = torch.nn.Sequential(
            nn.Linear(in_features=d_embedding, out_features=seq_len),
        )
        self.weights = torch.nn.Sequential(
            nn.Linear(in_features=d_embedding * (d_feature + seq_len),
                      out_features=2),
        )
        self.bias = torch.nn.Sequential(
            nn.Linear(in_features=d_feature, out_features=d_hidden),
            nn.ReLU(),
            nn.Linear(in_features=d_hidden, out_features=d_feature),
        )

    def forward(self, X_T, X_F, ts):

        X_cat = torch.cat([X_F, X_T], dim=1)
        weight = self.weights(X_cat.reshape(X_cat.size(0), -1)).unsqueeze(1)

        X_T = self.output_linear_T(X_T)
        X_F = self.output_linear_F(X_F)

        # gap = X_T - X_F.permute(0, 2, 1)
        # bias = self.bias(gap)

        X = (torch.mul(X_T, weight[:, :, 0:1]) +
             torch.mul(X_F.transpose(-1, -2), weight[:, :, 1:2]))

        step_info = self.timeembed1(ts).reshape(-1, 1, self.d_output)

        X = X + step_info

        return X


def position_encode(x):
    pe = torch.ones_like(x[0])
    position = torch.arange(0, x.shape[1]).unsqueeze(-1)
    temp = torch.Tensor(range(0, x.shape[-1], 2))
    temp = temp * -(math.log(10000) / x.shape[-1])
    temp = torch.exp(temp).unsqueeze(0)
    temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
    pe[:, 0::2] = torch.sin(temp)
    pe[:, 1::2] = torch.cos(temp)
    return pe


class Input_Layer(torch.nn.Module):
    def __init__(self, second_dim, third_dim, d_embedding, dropout, pe=True):
        super(Input_Layer, self).__init__()
        self.token_embedding = TokenEmbedding(second_dim=second_dim, third_dim=third_dim,
                                              d_embedding=d_embedding, dropout=dropout)
        self.pe = pe

    def forward(self, x_ori_art):
        x_ori_art = self.token_embedding(x_ori_art)
        residual = x_ori_art
        if self.pe:
            x_ori_art += position_encode(x_ori_art)
        return x_ori_art


class TokenEmbedding(torch.nn.Module):
    def __init__(self, second_dim, third_dim, d_embedding, dropout):
        super(TokenEmbedding, self).__init__()
        self.embedding = torch.nn.Linear(in_features=third_dim, out_features=d_embedding)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, d_embedding, d_hidden, q, k, v, h, dropout, mask=True):
        super(Encoder, self).__init__()
        self.MHA = MultiHeadAttention(d_embedding=d_embedding, q=q, k=k, v=v, h=h, mask=mask)
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.layernorm_1 = torch.nn.LayerNorm(d_embedding)
        self.layernorm_2 = torch.nn.LayerNorm(d_embedding)

        self.feedforward = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                               torch.nn.Conv1d(in_channels=d_embedding, out_channels=d_hidden,
                                                               kernel_size=1),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(dropout),
                                               torch.nn.Conv1d(in_channels=d_hidden, out_channels=d_embedding,
                                                               kernel_size=1))

    def forward(self, query, key, value):
        residual = query
        x, score = self.MHA(query, key, value)
        x = self.dropout(x)
        y = x = self.layernorm_1(x + residual)

        y = self.feedforward(y.transpose(-1, -2)).transpose(-1, -2)
        x = self.layernorm_2(x + y)

        return x, score


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_embedding, q, k, v, h, mask=True):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = torch.nn.Linear(in_features=d_embedding, out_features=h * q)
        self.W_K = torch.nn.Linear(in_features=d_embedding, out_features=h * k)
        self.W_V = torch.nn.Linear(in_features=d_embedding, out_features=h * v)

        self.mask = mask
        self.h = h
        self.inf = (-1 * torch.ones(1) * torch.inf + 1)

        self.out_linear = torch.nn.Linear(v * h, d_embedding)

    def forward(self, query, key, value):
        Q = torch.cat(torch.chunk(self.W_Q(query), self.h, dim=-1), dim=0)
        K = torch.cat(torch.chunk(self.W_K(key), self.h, dim=-1), dim=0)
        V = torch.cat(torch.chunk(self.W_V(value), self.h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2))

        if self.mask and score.shape[-1] > 1:
            mask = torch.ones_like(score[0])
            mask_1 = mask.tril(diagonal=-1)
            mask_2 = mask.triu(diagonal=1)
            mask = (mask_1 + mask_2).to(score.device)
            score = torch.where(mask > 0, score, self.inf.to(score.device))

        score = torch.softmax(score, dim=-1)

        attention = torch.cat(torch.chunk(torch.matmul(score, V), self.h, dim=0), dim=-1)

        out = self.out_linear(attention)

        return out, score


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.model(x)

class Frequency_Decomposition(nn.Module):
    def __init__(self,
                 second_dim,
                 flimit,
                 fre_decom,
                 dropout: float = 0.2,
                 ):
        super(Frequency_Decomposition, self).__init__()

        assert fre_decom in ['slice', 'fixed', 'all']
        self.fre_decom = fre_decom
        self.fixed_flimit = flimit

        if fre_decom in ['slice', 'all']:  # the proposed multi-scale frequency decomposition
            self.embedding_conv_high_pre = nn.Sequential(
                nn.Conv1d(in_channels=second_dim, out_channels=second_dim, kernel_size=3, padding=1),
                nn.Dropout(dropout),
            )

            self.embedding_conv_high = nn.Sequential(
                nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1),
                nn.Dropout(dropout),
            )
            self.embedding_conv_rest = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
                nn.Dropout(dropout),
            )
            self.embedding_conv_overlap = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
                nn.Dropout(dropout),
            )
        elif fre_decom == 'fixed':  # use the fixed boundary of low- and high frequency
            self.embedding_conv_high = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1),
                nn.Dropout(dropout),
            )
            self.embedding_conv_rest = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1),
                nn.Dropout(dropout),
            )

    def forward(self, x, x_intact=None):
        if self.fre_decom == 'slice':
            rest_f, high_f, overlap = self.slice_frequency(x, x_intact)
            return rest_f, high_f, overlap

        elif self.fre_decom == 'fixed':  # use the fixed boundary of low- and high frequency
            rest_f, high_f = self.fixed_frequency(x)
            return rest_f, high_f

        elif self.fre_decom == 'all':
            rest_f, high_f, overlap = self.slice_frequency(x, x_intact)
            x = rest_f + high_f - overlap
            return x


    def slice_frequency(self, x, x_intact=None):
        B, L, F = x.shape

        self.flimit_h1 = 0.3
        self.flimit_h2 = 0.35
        self.flimit_h3 = 0.4
        self.flimit_h4 = 0.45

        self.flimit_l1 = 0.05
        self.flimit_l2 = 0.1
        self.flimit_l3 = 0.15
        self.flimit_l4 = 0.2
        self.flimit_l5 = 0.25
        self.flimit_l6 = 0.3

        # time dimension
        _, t_rest_f1 = self.get_f(x, self.flimit_h4, self.flimit_l1)  # [B, L, F]
        _, t_rest_f2 = self.get_f(x, self.flimit_h4, self.flimit_l2)  # [B, L, F]
        t_high_f1, t_rest_f3 = self.get_f(x, self.flimit_h1, self.flimit_l3)  # [B, L, F]
        t_high_f2, t_rest_f4 = self.get_f(x, self.flimit_h2, self.flimit_l4)  # [B, L, F]
        t_high_f3, t_rest_f5 = self.get_f(x, self.flimit_h3, self.flimit_l5)  # [B, L, F]
        t_high_f4, t_rest_f6 = self.get_f(x, self.flimit_h4, self.flimit_l6)  # [B, L, F]


        t_high_f1 = t_high_f1 - t_high_f2
        t_high_f2 = t_high_f2 - t_high_f3
        t_high_f3 = t_high_f3 - t_high_f4

        t_rest_f6 = t_rest_f6 - t_rest_f5
        t_rest_f5 = t_rest_f5 - t_rest_f4
        t_rest_f4 = t_rest_f4 - t_rest_f3
        t_rest_f3 = t_rest_f3 - t_rest_f2
        t_rest_f2 = t_rest_f2 - t_rest_f1

        # feature dimension
        _, f_rest_f1 = self.get_f(x.transpose(-1, -2), self.flimit_h1, self.flimit_l1)  # [B, F, L]
        _, f_rest_f2 = self.get_f(x.transpose(-1, -2), self.flimit_h1, self.flimit_l2)  # [B, F, L]
        f_high_f1, f_rest_f3 = self.get_f(x.transpose(-1, -2), self.flimit_h1, self.flimit_l3)  # [B, F, L]
        f_high_f2, f_rest_f4 = self.get_f(x.transpose(-1, -2), self.flimit_h2, self.flimit_l4)  # [B, F, L]
        f_high_f3, f_rest_f5 = self.get_f(x.transpose(-1, -2), self.flimit_h3, self.flimit_l5)  # [B, F, L]
        f_high_f4, f_rest_f6 = self.get_f(x.transpose(-1, -2), self.flimit_h4, self.flimit_l6)  # [B, F, L]

        f_high_f1 = f_high_f1 - f_high_f2
        f_high_f2 = f_high_f2 - f_high_f3
        f_high_f3 = f_high_f3 - f_high_f4

        f_rest_f6 = f_rest_f6 - f_rest_f5
        f_rest_f5 = f_rest_f5 - f_rest_f4
        f_rest_f4 = f_rest_f4 - f_rest_f3
        f_rest_f3 = f_rest_f3 - f_rest_f2
        f_rest_f2 = f_rest_f2 - f_rest_f1

        high_f = torch.stack([
            t_rest_f5,
            t_rest_f6,
            t_high_f1, t_high_f2, t_high_f3, t_high_f4,
            f_rest_f5.transpose(-1, -2),
            f_rest_f6.transpose(-1, -2),
            f_high_f1.transpose(-1, -2), f_high_f2.transpose(-1, -2), f_high_f3.transpose(-1, -2), f_high_f4.transpose(-1, -2), ], dim=1)
        rest_f = torch.stack([t_rest_f1, t_rest_f2, t_rest_f3, t_rest_f4, t_rest_f5, t_rest_f6,
                              t_high_f1,
                              t_high_f2,
                              f_rest_f1.transpose(-1, -2), f_rest_f2.transpose(-1, -2), f_rest_f3.transpose(-1, -2), f_rest_f4.transpose(-1, -2),
                              f_rest_f5.transpose(-1, -2), f_rest_f6.transpose(-1, -2),
                              f_high_f1.transpose(-1, -2),
                              f_high_f2.transpose(-1, -2)
                              ], dim=1)
        overlap = torch.stack([
                               t_rest_f5,
                               t_rest_f6,
                               t_high_f1,
                               t_high_f2,
                               f_rest_f5.transpose(-1, -2),
                               f_rest_f6.transpose(-1, -2),
                               f_high_f1.transpose(-1, -2),
                               f_high_f2.transpose(-1, -2)
        ], dim=1)



        high_f = self.embedding_conv_high_pre(high_f.reshape(-1, L, F)).reshape(B, -1, L, F)
        high_f = self.embedding_conv_high(high_f).squeeze(1).reshape(B, L, F)

        rest_f = self.embedding_conv_rest(rest_f).squeeze(1).reshape(B, L, F)

        overlap = self.embedding_conv_overlap(overlap).squeeze(1).reshape(B, L, F)

        return rest_f, high_f, overlap

    def fixed_frequency(self, x):
        fixed_b = self.fixed_flimit
        t_high_f, t_rest_f = self.get_f(x, fixed_b, fixed_b)  # [B, L, F]
        f_high_f, f_rest_f = self.get_f(x.transpose(-1, -2), fixed_b, fixed_b)  # [B, F, L]

        high_f = torch.stack([t_high_f, f_high_f.transpose(-1, -2)], dim=1)
        rest_f = torch.stack([t_rest_f, f_rest_f.transpose(-1, -2)], dim=1)

        high_f = self.embedding_conv_high(high_f).squeeze(1)
        rest_f = self.embedding_conv_rest(rest_f).squeeze(1)

        return high_f, rest_f


    def get_f(self, x, flimit_h, flimit_l,):
        high_f_list = []
        rest_f_list = []

        if x.shape[1] == 1:
            xf = torch.fft.rfft(x, dim=1).float().to(x.device)
            return xf, xf
        for j in range(x.shape[2]):
            x_j = x[:, :, j]
            xf = torch.fft.rfft(x_j, dim=-1)
            pass_f = (torch.abs((torch.fft.rfftfreq(x_j.shape[1]))).to(x.device) > flimit_h).float().to(x.device)
            pass_l = (torch.abs((torch.fft.rfftfreq(x_j.shape[1]))).to(x.device) < flimit_l).float().to(x.device)
            high_f = torch.fft.irfft(xf * pass_f, n=x_j.shape[1], dim=-1)
            rest_f = torch.fft.irfft(xf * pass_l, n=x_j.shape[1], dim=-1)
            high_f_list.append(high_f)
            rest_f_list.append(rest_f)

        high_f = torch.stack(high_f_list, dim=2)
        rest_f = torch.stack(rest_f_list, dim=2)
        high_f_x = torch.flip(high_f, [1])
        high_f_x = torch.cat([high_f_x[:, -1:, :], high_f_x[:, :-1, :]], dim=1)
        rest_f_x = torch.flip(rest_f, [1])
        rest_f_x = torch.cat([rest_f_x[:, -1:, :], rest_f_x[:, :-1, :]], dim=1)

        return high_f_x, rest_f_x


class LowRank_Reconstrution(nn.Module):
    def __init__(self,
                 low_dim: int,
                 second_dim: int,
                 third_dim: int,
                 d_embedding: int = 512,
                 d_hidden: int = 2048,
                 q: int = 64,
                 k: int = 64,
                 v: int = 64,
                 h: int = 8,
                 mask: bool = True,
                 dropout: float = 0.1,
                 ) -> object:
        super(LowRank_Reconstrution, self).__init__()

        self.init_lowrank_q = nn.Parameter(torch.randn(low_dim, d_embedding))
        self.init_lowrank_q_f = nn.Parameter(torch.randn(second_dim, d_embedding))

        self.fuse_high_f = nn.Sequential(
            torch.nn.Conv1d(in_channels=third_dim * 2, out_channels=d_hidden, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=d_hidden, out_channels=d_hidden, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=d_hidden, out_channels=d_embedding, kernel_size=1),
        )

        self.up_embedding = nn.Linear(in_features=third_dim, out_features=d_embedding)
        self.up_embedding_f = nn.Linear(in_features=third_dim, out_features=d_embedding)
        self.down_embedding = nn.Linear(in_features=d_embedding, out_features=third_dim)
        self.down_embedding_f = nn.Linear(in_features=d_embedding, out_features=third_dim)

        self.downsamplingAtten_f = MultiHeadAttention(d_embedding=d_embedding, q=q, k=k, v=v, h=h, mask=mask)
        self.upsampingAtten_f = MultiHeadAttention(d_embedding=d_embedding, q=q, k=k, v=v, h=h, mask=mask)

        self.downsamplingAtten = MultiHeadAttention(d_embedding=d_embedding, q=q, k=k, v=v, h=h, mask=mask)
        self.upsampingAtten = MultiHeadAttention(d_embedding=d_embedding, q=q, k=k, v=v, h=h, mask=mask)

        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.layernorm_1 = torch.nn.LayerNorm(d_embedding)
        self.layernorm_2 = torch.nn.LayerNorm(d_embedding)

        self.feedforward = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                               torch.nn.Conv1d(in_channels=d_embedding, out_channels=d_hidden,
                                                               kernel_size=1),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(dropout),
                                               torch.nn.Conv1d(in_channels=d_hidden, out_channels=d_embedding,
                                                               kernel_size=1))


    def forward(self, X):

        x = self.up_embedding(X)

        low_rank_x, _ = self.downsamplingAtten(query=self.init_lowrank_q, key=x, value=x)
        reconstructed_x, _ = self.upsampingAtten(query=x, key=self.init_lowrank_q, value=low_rank_x)

        residual = x

        x = self.dropout(reconstructed_x)
        x = self.layernorm_1(x + residual)
        # x = x + residual

        residual = x
        x = self.dropout(self.feedforward(x.transpose(-1, -2))).transpose(-1, -2)
        x = self.layernorm_2(x + residual)
        # x = x + residual

        out = self.down_embedding(x)

        return out
