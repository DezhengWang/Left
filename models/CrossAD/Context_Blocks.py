import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class ContextNet(nn.Module):
    def __init__(self, router, querys, extractor):
        super().__init__()
        self.router = router
        self.querys = querys
        self.extractor = extractor
    
    def forward(self, x_enc, local_repr, mask=None):
        q_indices = self.router(x_enc)
        q = torch.einsum('bn,nqd->bqd', q_indices, self.querys)                                         # q: [bs x query_len x d_model]
        
        query_latent_distances, context = self.extractor(q, local_repr, mask)
        return query_latent_distances, context


class Router(nn.Module):
    def __init__(self, seq_len, n_vars, n_query, topk=5):
        super().__init__()
        self.k = topk
        self.fc = nn.Sequential(nn.Flatten(-2), nn.Linear(seq_len*n_vars, n_query))

    def forward(self, x):
        bs, t, c = x.shape
        # fft
        x_freq = torch.fft.rfft(x, dim=1, n=t)
        # topk
        _, indices = torch.topk(x_freq.abs(), self.k, dim=1)    # indices: [bs x k x c]                             
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing="ij")
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        mask = torch.zeros_like(x_freq, dtype=torch.bool)       # mask: [bs x f x c]  
        mask[index_tuple] = True
        x_freq[~mask] = torch.tensor(0.0+0j, device=x_freq.device)
        # ifft
        x = torch.fft.irfft(x_freq, dim=1, n=t)
        # mlp
        logits = self.fc(x)                                     # logits: [bs x n_query]
        # gumbel softmax
        q_indices = F.gumbel_softmax(logits, tau=1, hard=True)  # q_indices: [bs x n_query]

        return q_indices

 
class Extractor(nn.Module):
    def __init__(self, layers, 
                 context_size=64, query_len=5, d_model=128, decay=0.99, epsilon=1e-5
                ):
        super().__init__()
        # context
        self.context_size = context_size
        self.query_len = query_len
        self.d_model = d_model
        self.register_buffer("context", torch.randn(context_size, query_len, d_model))                  # context: [N x query_len x d_model]
        self.register_buffer("ema_count", torch.ones(context_size))   
        self.register_buffer("ema_dw", torch.zeros(context_size, query_len, d_model))                
        self.decay = decay
        self.epsilon = epsilon
        # extractor
        self.extractor = nn.ModuleList(layers)
              
    def update_context(self, q):
        # q: [bs x query_len x d_model]

        _, q_len, d = q.shape
        q_flat = q.reshape(-1, q_len*d)                                                                 # [bs x query_len*d_model]
        g_flat = self.context.reshape(-1, q_len*d)                                                      # [N x query_len*d_model]
        N, D = g_flat.shape

        distances = (
            torch.sum(q_flat**2, dim=1, keepdim=True) + 
            torch.sum(g_flat**2, dim=1) -
            2 * torch.matmul(q_flat, g_flat.t())
        )                                                                                               # [bs x N] soft
        # distances = torch.sum((q_flat.unsqueeze(1)-g_flat.unsqueeze(0))**2, dim=-1)                   
        indices = torch.argmin(distances.float(), dim=-1)                                               # [bs] 
        encodings = F.one_hot(indices, N).float()                                                       # [bs x N] hard
        q_context = torch.einsum("bn,nqd->bqd", [encodings, self.context])                              # [bs x query_len x d_model]
        q_hat = torch.einsum("bn,bqd->nqd", [encodings, q])                                             # [N x query_len x d_model]
        
        # query_latent_distances
        query_latent_distances = torch.mean(F.mse_loss(q_context.detach(), q, reduction="none"), dim=(1, 2))    # [bs]

        if self.training:
            with torch.no_grad():
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)   # [N]
                n = torch.sum(self.ema_count)
                self.ema_count = (self.ema_count + self.epsilon) / (n + D * self.epsilon) * n
                
                dw = torch.einsum("bn,bqd->nqd", [encodings, q])                                                # [N x query_len x d_model]
                self.ema_dw = self.decay * self.ema_dw + (1 - self.decay) * dw                                  # [N x query_len x d_model]
                self.context = self.ema_dw / self.ema_count.unsqueeze(-1).unsqueeze(-1)
        return query_latent_distances, q_hat
               
    def concat_context(self, context):
        return context.view(-1, self.d_model)                                                       # [N*query_len x d_model]

    def forward(self, q, local_repr, mask=None):
        # q: [bs, query_len, d_model]
        # local_repr: [bs, ms_t, d_model]
        for layer in self.extractor:
            q = layer(q, local_repr, mask)                                                          # [bs x query_len x d_model]

        query_latent_distances, q_hat = self.update_context(q) 
        context = self.concat_context(q_hat+self.context.detach()-q_hat.detach())                   # context: [N*query_len x d_model]
        return query_latent_distances, context                                                      # N,query_len,d - bs,q_query_len,d_model
        

class ExtractorLayer(nn.Module):
    def __init__(self, cross_attention, d_model, d_ff=None, norm="batchnorm", dropout=0.1, activation="relu"):
        super().__init__()

        d_ff = d_ff or 4 * d_model
        # attention
        self.cross_attention = cross_attention
        # ffn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # norm
        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, q, local_repr, mask=None):
        # q: [bs, query_len, d_model]
        # local_repr: [bs, ms_t, d_model]

        # cross_attention
        q = q + self.dropout(self.cross_attention(
            q, local_repr, local_repr,
            attn_mask=mask
        )[0])
        q = self.norm1(q)                                                       # q: [bs x query_len x d_model]

        # ffn
        y = self.dropout(self.activation(self.conv1(q.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))                        # y: [bs x query_len x d_model]

        return self.norm2(q + y)