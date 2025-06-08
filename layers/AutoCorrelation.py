import torch
import torch.nn as nn
import math

# This module implements the AutoCorrelation mechanism used in the Autoformer model for time-series forecasting.
class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor                         # Controls number of top-k patterns in delay aggregation
        self.scale = scale                           # Not used directly in this code (could be for scaling attn)
        self.mask_flag = mask_flag                   # Masking flag for future use
        self.output_attention = output_attention     # If True, output the attention weights
        self.dropout = nn.Dropout(attention_dropout) # Dropout for regularization

    def time_delay_agg_training(self, values, corr):
        """
        Efficient autocorrelation aggregation for training phase.
        values: [batch, head, channel, length]
        corr:   [batch, head, channel, length]
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # Select top-k time delays by autocorrelation score
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)   # Mean over head and channel: [batch, length]
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]  # [top_k]
        # Gather weights for these delays
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)                 # Normalize weights (probabilities)
        # Aggregate time-delayed values according to autocorrelation weights
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)  # Roll (shift) values by delay amount
            delays_agg = delays_agg + pattern * \
                (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        Efficient autocorrelation aggregation for inference phase.
        Allows per-sample delay selection (using top-k for each batch element).
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # Create index tensor for all time steps
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # Find top-k delay positions for each sample
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [batch, length]
        weights, delay = torch.topk(mean_value, top_k, dim=-1)   # [batch, top_k]
        tmp_corr = torch.softmax(weights, dim=-1)                # Normalize
        # Duplicate values along last dimension for circular shifts
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            # Gather values at the delayed positions
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard (non-optimized) autocorrelation aggregation for analysis.
        Each sample, head, channel can have unique delays.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # Build index tensor
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)  # Each batch, head, channel: [top_k, length]
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        # Input shapes:
        # queries: [batch, length, heads, embed_dim]
        # keys, values: [batch, length, heads, embed_dim]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        # Ensure keys/values are the same length as queries
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        # Phase 1: Period-based dependencies via frequency domain
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)
        # Phase 2: Time delay aggregation (different for train/inference)
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        # Optionally return attention map
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)

# A wrapper layer for integrating the AutoCorrelation block into deep architectures
class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)     # Dimension per head
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation        # Instance of AutoCorrelation
        # Linear projections for query, key, value
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model) # Final output projection
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        # [batch, seq_len, model_dim]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # Project input to multiple heads and split
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # Apply AutoCorrelation mechanism
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1) # Merge heads
        return self.out_projection(out), attn # Final linear projection
