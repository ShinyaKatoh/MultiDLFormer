import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary
    
class DSconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, 
                                        out_channels=in_ch,
                                        kernel_size=kernel_size, 
                                        groups=in_ch, 
                                        stride=stride,
                                        padding=(kernel_size - 1) // 2)
        
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, 
                                        out_channels=out_ch,
                                        kernel_size=1, 
                                        padding='same')
        
    def forward(self, x):
        
        x = rearrange(x, 'B L C -> B C L')
        
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        
        x = rearrange(x, 'B C L -> B L C')
       
        return  x
    
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=8000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
    def forward(self, x):
        # xの長さに応じてエンベディングをスライス
        encoding = self.encoding.to(x.device)
        return x + encoding[:, :x.size(1), :]
    
class InputLayer(nn.Module): 
    def __init__(self, in_channels, emb_dim, kernel_size, stride, h, w):
        super().__init__() 
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = emb_dim,
                              kernel_size = (1, kernel_size),
                              stride = (1, stride),
                              padding=(0, kernel_size//2-1),
                              bias=False)

        # クラストークン 
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        
        self.pos_emb = PositionalEncoding(emb_dim=emb_dim)
        
        # self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x):
        
        x = self.conv(x)
        x = rearrange(x, 'B C H W -> B (H W) C')
        # x = self.ln(x)
        
        x = torch.cat([self.cls_token.repeat(repeats=(x.size(0),1,1)), x], dim=1)
        x = self.pos_emb(x)
        
        return x
 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 channels,
                 num_heads,
                 dropout_ratio):
        super().__init__()
        
        self.linear_q = nn.Linear(channels, channels, bias=False)
        self.linear_k = nn.Linear(channels, channels, bias=False)
        self.linear_v = nn.Linear(channels, channels, bias=False)
        
        self.ln_q = nn.LayerNorm(channels)
        self.ln_k = nn.LayerNorm(channels)
        self.ln_v = nn.LayerNorm(channels)
        
        self.ln_kv = nn.LayerNorm(channels)
        
        self.head = num_heads
        self.head_ch = channels // num_heads
        self.sqrt_dh = self.head_ch**0.5 
        
        self.attn_drop = nn.Dropout(dropout_ratio)

        self.w_o = nn.Linear(channels, channels, bias=False)
        self.w_drop = nn.Dropout(dropout_ratio)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.reducer = DSconv(channels, channels, 7, 4)
        
    def forward(self, x):
        
        cls_token = x[:, :1, :]     # (B, 1, C)
        tokens = x[:, 1:, :]
        
        tokens = self.reducer(tokens)
        tokens = self.ln_kv(tokens)
        
        # print(tokens.shape)
        
        kv = torch.cat([cls_token, tokens], dim=1)
        # print(x.shape)
        q = self.linear_q(x)
        # print(q.shape)
        k = self.linear_k(kv)
        v = self.linear_v(kv)
            
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)
        
        q = rearrange(q, 'B L (h C) -> B h L C', h=self.head)
        k = rearrange(k, 'B L (h C) -> B h L C', h=self.head)
        v = rearrange(v, 'B L (h C) -> B h L C', h=self.head)
        
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=False)
        
        # k_T = k.transpose(2, 3)
        
        # dots = (q @ k_T) / self.sqrt_dh
        # attn = self.softmax(dots)
        # attn = self.attn_drop(attn)
        # out = attn @ v
        
        out = rearrange(out, 'B h L C -> B L (h C)')
        
        out = self.w_o(out) 
        out = self.w_drop(out)
        
        with torch.no_grad():
            # attention map = softmax(q @ k^T / sqrt(d))
            # scale = q.shape[-1] ** 0.5
            # attn_map = torch.matmul(q, k.transpose(-2, -1)) / scale
            # attn_map = F.softmax(attn_map, dim=-1)
            k_T = k.transpose(2, 3)
            dots = (q @ k_T) / self.sqrt_dh
            attn = self.softmax(dots)
            attn = self.attn_drop(attn)
            # out = attn @ v
        
        return out, attn
    
class MixFFN(nn.Module):
    def __init__(self,
                 emb_dim,
                 expantion_ratio:int=4):
        super().__init__()
        self.linear1 = nn.Conv1d(emb_dim, 
                                 emb_dim, 
                                 kernel_size = 1)
        
        self.linear2 = nn.Conv1d(emb_dim * expantion_ratio, 
                                 emb_dim, 
                                 kernel_size = 1)
        
        self.conv = nn.Conv1d(in_channels=emb_dim, 
                              out_channels=emb_dim * expantion_ratio, 
                              kernel_size=3, 
                              groups=emb_dim,
                              padding='same')
        
        self.gelu = nn.GELU()

    def forward(self, x):
     
        x1 = x[:,:1]
        x2 = x[:,1:]
        
        x2 = rearrange(x2, 'B L C -> B C L')
        
        x2 = self.linear1(x2)
    
        x2 = self.conv(x2)
        x2 = self.gelu(x2)
        
        x2 = self.linear2(x2)
       
        x2 = rearrange(x2, 'B C L -> B L C')
        
        out = torch.cat((x1, x2), dim=1)
        
        return out
    
class FFN(nn.Module):
    def __init__(self, channels, dropout_ratio):
        super().__init__()
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)
        
        self.drop1 = nn.Dropout(dropout_ratio)
        self.drop2 = nn.Dropout(dropout_ratio)
        
        self.gelu = nn.GELU()

    def forward(self,x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

    
class MHSAtoFFN(nn.Module):
    def __init__(self,
                 emb_dim,
                 head_num,
                 dropout_ratio):
        
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(emb_dim,
                                           head_num,
                                           dropout_ratio)
      
        self.ffn = FFN(emb_dim, 
                        dropout_ratio)
        
        self.mixffn1 = MixFFN(emb_dim)
        self.mixffn2 = MixFFN(emb_dim)
        
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ln3 = nn.LayerNorm(emb_dim)
        self.ln4 = nn.LayerNorm(emb_dim)
       
    def forward(self, x):
        
        residual_mhsa = x
        mhsa_input = self.ln1(x)
        mhsa_output, attn = self.mhsa(mhsa_input)
        mhsa_output2 = mhsa_output + residual_mhsa
        
        residual_mixffn = mhsa_output2
        mixffn_input = self.ln2(mhsa_output2)
        mixffn_output = self.mixffn1(mixffn_input) + residual_mixffn
        
        residual_mixffn2 = mixffn_output
        mixffn_input2 = self.ln3(mixffn_output)
        mixffn_output2 = self.mixffn2(mixffn_input2) + residual_mixffn2
       
        residual_ffn = mixffn_output2
        ffn_input = self.ln4(mixffn_output)
        ffn_output = self.ffn(ffn_input) + residual_ffn
     
        return ffn_output
    
class ViTEncoderBlock(nn.Module):
    def __init__(self, 
                 emb_dim,
                 head_num,
                 dropout_ratio,
                 block_num):
        super().__init__()
       
        self.Encoder = nn.Sequential(*[MHSAtoFFN(emb_dim,
                                                 head_num,
                                                 dropout_ratio)
                                       for _ in range(block_num)])
        
    def forward(self, x):
        x = self.Encoder(x)
        return x
    
class MLP(nn.Module):
    def __init__(self,
                 emb_dim,
                 dropout_ratio,
                 class_num):
        super().__init__()
        self.linear = nn.Linear(emb_dim, class_num)
        self.softmax = nn.Softmax(dim=1)
        self.ln = nn.LayerNorm(emb_dim)
    
    def forward(self, x):
        x = self.ln(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x
    
class Model(nn.Module): 
    def __init__(self, 
                 in_channels:int=3, 
                 class_num:int=3, 
                 emb_dim:int=64, 
                 h:int=10,
                 w:int=6000,
                 kernel_size:int=50,
                 stride:int=25,
                 num_blocks:int=7, 
                 head_num:int=4, 
                 dropout_ratio:float=0.3):
        super().__init__()
        
        self.input = InputLayer(in_channels, 
                                emb_dim, 
                                kernel_size,
                                stride, 
                                h,
                                w)

        self.encoder = ViTEncoderBlock(emb_dim,
                                       head_num,
                                       dropout_ratio,
                                       num_blocks)
         
        self.mlp = MLP(emb_dim, dropout_ratio, class_num)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
       
        out = self.input(x)
        
        out = self.encoder(out)

        cls_token = out[:,0]

        pred = self.mlp(cls_token)

        return pred

    
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    model = Model().to('cpu')
    summary(model, input_size=(32, 3, 10, 6000))