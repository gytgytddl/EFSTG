import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Chomp_T(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp_T, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[..., :-self.chomp_size].contiguous()
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return norm * self.weight


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_of_week_idx = (x[..., 1] * 6).round().long().clamp(0, 6)
        time_of_day_idx = (x[..., 2] * 287).round().long().clamp(0, 287)
        tem_emb = (self.time_day[time_of_day_idx] + self.time_week[day_of_week_idx]).permute(0, 3, 2, 1)
        return tem_emb


class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.w1 = nn.Conv2d(in_features, out_features, 1)
        self.w2 = nn.Conv2d(in_features, out_features, 1)
        self.w3 = nn.Conv2d(out_features, out_features, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(self.drop(gate))


class EulerFrequencyModulation(nn.Module):
    def __init__(self, channels, seq_len):
        super().__init__()
        self.num_freqs = seq_len // 2 + 1
        self.complex_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=(1, 5), padding=(0, 2))
        self.delta_theta = nn.Parameter(torch.empty(1, channels, 1, self.num_freqs))
        nn.init.uniform_(self.delta_theta, -math.pi, math.pi)
        self.channel_mix = nn.Conv2d(channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        B, C, N, T = x.shape
        x_freq = torch.fft.rfft(x.float(), dim=-1, norm='ortho')
        mag_real_imag = torch.cat([x_freq.real, x_freq.imag], dim=1)

        freq_filter = self.complex_conv(mag_real_imag)
        mag_scale = torch.sigmoid(freq_filter[:, :C, :, :])
        dynamic_phase_shift = torch.tanh(freq_filter[:, C:, :, :]) * math.pi

        mag = torch.abs(x_freq) * mag_scale
        phase_modulated = torch.angle(x_freq) + self.delta_theta + dynamic_phase_shift

        x_mod = torch.polar(mag, phase_modulated)
        x_out = torch.fft.irfft(x_mod, n=T, dim=-1, norm='ortho')
        return (self.channel_mix(x_out) + x).to(x.dtype)


class SpectrumAwareGraphConv(nn.Module):
    def __init__(self, channels, num_nodes, seq_len, top_k=15):
        super().__init__()
        self.top_k = top_k
        self.gcn_conv = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.ln = nn.LayerNorm([channels, num_nodes, seq_len])
        self.spec_projection = nn.Linear(seq_len // 2 + 1, 32)

    def forward(self, x):
        B, C, N, T = x.shape
        residual = x
        with torch.amp.autocast('cuda', enabled=False):
            x_freq = torch.fft.rfft(x.float(), norm='ortho', dim=-1)
            mag_feat = torch.abs(x_freq).mean(dim=1)

            mag_feat = self.spec_projection(mag_feat)
            mag_feat = F.normalize(mag_feat, p=2, dim=-1)

            dist = torch.cdist(mag_feat, mag_feat, p=2.0)

            sigma = torch.std(dist, dim=(1, 2), keepdim=True) + 1e-5
            adj = torch.exp(-(dist ** 2) / (2 * (sigma ** 2)))

            topk_val, topk_idx = torch.topk(adj, k=self.top_k, dim=-1)
            sparse_adj = torch.zeros_like(adj).scatter_(2, topk_idx, topk_val)

            deg = sparse_adj.sum(dim=-1, keepdim=True) + 1e-6
            deg_inv_sqrt = deg.pow(-0.5)
            norm_adj = deg_inv_sqrt * sparse_adj * deg_inv_sqrt.transpose(1, 2)

        out = torch.einsum('bnm, bcmt -> bcnt', norm_adj.to(x.dtype), x)
        return self.ln(self.gcn_conv(out) + residual), norm_adj


class TemporalConvNet_SiLU(nn.Module):
    def __init__(self, features, kernel_size, dropout=0.2, levels=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(levels):
            dilation_size = 2 ** i
            padding_size = (kernel_size - 1) * dilation_size
            self.layers.append(nn.Sequential(
                nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding_size)),
                Chomp_T(padding_size),

                nn.SiLU(),
                nn.Dropout(dropout)
            ))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class TSGAT_Optimized(nn.Module):
    def __init__(self, device, input_dim, num_nodes, channels, real_adj=None, granularity=288, dropout=0.2, ablation_mode=None, tcn_levels=3, top_k=15, **kwargs):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.input_len = 12
        self.output_len = 12
        hidden_dim = channels * 2

        if real_adj is not None:
            self.register_buffer('real_adj', real_adj)
        else:
            self.real_adj = None
        self.graph_alpha = nn.Parameter(torch.tensor(-4.0))

        self.skip_conv = nn.Conv2d(in_channels=1, out_channels=self.output_len, kernel_size=(1, self.input_len))
        self.residual_proj = nn.Conv2d(1, self.output_len, (1, 1)) if self.input_len == self.output_len else None

        self.start_conv = nn.Conv2d(input_dim, channels, kernel_size=(1, 1))
        self.Temb = TemporalEmbedding(granularity, channels)
        self.node_emb = nn.Parameter(torch.empty(1, channels, num_nodes, 1))
        nn.init.kaiming_uniform_(self.node_emb, a=math.sqrt(5))

        self.temporal_conv = TemporalConvNet_SiLU(hidden_dim, kernel_size=3, levels=tcn_levels, dropout=dropout)

        self.node_emb_1 = nn.Parameter(torch.randn(num_nodes, 32) * 0.01)
        self.node_emb_2 = nn.Parameter(torch.randn(32, num_nodes) * 0.01)

        self.ln_local = RMSNorm(hidden_dim)
        self.euler = EulerFrequencyModulation(hidden_dim, self.input_len)
        self.dynamic_gcn = SpectrumAwareGraphConv(hidden_dim, num_nodes, self.input_len, top_k=top_k)
        self.ln_global = RMSNorm(hidden_dim)

        self.fusion_gate = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)

        self.dec_time = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, self.input_len))
        self.swiglu = SwiGLU(hidden_dim, hidden_dim)
        self.end_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, self.output_len, kernel_size=(1, 1))
        )
        self.identity_weight = nn.Parameter(torch.tensor(0.5))
        self.gc_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        info_dict = {}

        x_target = x[:, 0:1, :, :]

        # 1. 编码输入
        x_in = torch.cat([self.start_conv(x) + self.node_emb, self.Temb(x.permute(0, 3, 2, 1))], dim=1)

        # 2. 时域卷积 (TCN)
        if self.ablation_mode == 'w/o_TCN':
            h_local = x_in
        else:
            h_local = self.temporal_conv(x_in)

        # 3. 静态图卷积 (GC)
        raw_adj = torch.mm(self.node_emb_1, self.node_emb_2) / math.sqrt(32.0)
        adp_adj = F.softmax(raw_adj, dim=-1)
        info_dict['static_adj'] = adp_adj  # 收集自适应静态图
        use_real_adj = self.real_adj is not None and self.ablation_mode != 'w/o_PhyGraph'

        if use_real_adj:
            alpha = torch.sigmoid(self.graph_alpha)
            combined_adj = alpha * self.real_adj + (1 - alpha) * adp_adj
            graph_out = torch.einsum('vw, bcwt -> bcvt', combined_adj, h_local)
        else:
            graph_out = torch.einsum('vw, bcwt -> bcvt', adp_adj, h_local)

        if self.ablation_mode == 'w/o_GC':
            graph_out = graph_out * 0.0
        else:
            graph_out = graph_out * self.gc_gate

        h_local = self.ln_local(h_local + graph_out + x_in)

        # 4. 频域分支 (Freq)
        if self.ablation_mode == 'w/o_Euler':
            freq_input = h_local
        else:
            freq_input = self.euler(h_local)

        h_freq, dynamic_adj = self.dynamic_gcn(freq_input)
        info_dict['dynamic_adj'] = dynamic_adj

        if self.ablation_mode == 'w/o_Freq':
            h_final = h_local
        else:
            h_freq = self.ln_global(h_freq + h_local)
            # 5. 门控融合
            gate = torch.sigmoid(self.fusion_gate(torch.cat([h_local, h_freq], dim=1)))
            info_dict['fusion_gate'] = gate
            h_final = gate * h_local + (1 - gate) * h_freq

        # 6. 深层解码
        out_deep = self.dec_time(h_final)
        out_deep = self.swiglu(out_deep)
        out_deep = self.end_conv(out_deep)

        # 7. 线性直连 (Skip)
        x_target = x[:, 0:1, :, :]
        out_skip = self.skip_conv(x_target)
        identity = x_target[:, :, :, -1:]
        if self.ablation_mode == 'w/o_Skip':
            final_out = out_deep
        else:
            final_out = out_deep + out_skip + self.identity_weight * identity

        return final_out, info_dict

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()]) / 1e6