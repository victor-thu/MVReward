import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from ..BLIP.blip import BLIP_Pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Calculate the path relative to the current file location
# current_directory = os.path.dirname(os.path.abspath(__file__))
# config_path = os.path.join(current_directory, '../../config/med_config.json')

current_file = Path(__file__).resolve()
config_path = current_file.parent.parent.parent / 'configs' / 'med_config.json'
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.positional_embeddings = nn.Embedding(max_len, embed_size)

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(x.size(0), seq_length)
        pos_embeddings = self.positional_embeddings(positions)
        return x + pos_embeddings
    
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        bs = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(bs, value_len, self.heads, self.head_dim)
        keys = keys.reshape(bs, key_len, self.heads, self.head_dim)
        queries = query.reshape(bs, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            bs, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1536),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 384),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 64),
            # nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, input):
        return self.layers(input)
    
class Scorer(nn.Module):
    def __init__(self, feature_dim):
        super(Scorer, self).__init__()
        self.positional_embedding = PositionalEmbedding(feature_dim, 12)
        self.attention = SelfAttention(feature_dim, heads=8)
        self.LayerNorm = nn.LayerNorm(feature_dim, eps=1e-12)
        # self.pooling = nn.AdaptiveAvgPool1d(1)
        self.mlp = MLP(feature_dim*12)  # Single output for score

    def forward(self, x, mask=None):
        # print(f'x{x.shape}')
        x = self.positional_embedding(x)
        attn_output = self.attention(x, x, x, mask)
        # print(f'attn_output{attn_output.shape}')
        residual_block = x + attn_output
        mv_embed = self.LayerNorm(residual_block).view(x.shape[0], -1)
        # print(f'mv_embed{mv_embed.shape}')
        # pooled_output = self.pooling(attn_output.permute(0, 2, 1)).squeeze(-1)
        score = self.mlp(mv_embed)
        # print(f'score{score.shape}')
        return score

class mvreward(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.blip = BLIP_Pretrain(image_size=224, vit='base', med_config=config_path)
        self.preprocess = _transform(224)
        self.scorer = Scorer(feature_dim=768)

        self.prior_mean = 0.0369387455284595
        self.prior_std = 1.9023002386093140

    def rate_mvimgs(self, input_view, generated_views):
        
        # encode data
        b, _, _, _ = input_view.shape
        f, _, _, _ = generated_views.shape
        n = int(f / b)
        
        # just for test
        # reorder_pattern = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
        # generated_views = generated_views[reorder_pattern, :, :, :].to(self.device)

        combined_views = torch.cat((input_view, generated_views), dim=0).to(self.device) 
        combined_embeds = self.blip.visual_encoder(combined_views)   # [n + b, seq_len, feature_dim]
        input_embeds = combined_embeds[:b, :].unsqueeze(1).expand(-1, n, -1, -1).contiguous().view(b * n , -1, combined_embeds.shape[-1])  # [b*n, seq_length, feature_dim]
        view_embeds = combined_embeds[b:, :].to(self.device)
        atts_mask = torch.ones((view_embeds.size()[:-1]), dtype=torch.long).to(self.device)
        mixed_embed = self.blip.text_encoder(inputs_embeds = input_embeds,
                                            encoder_hidden_states = view_embeds,
                                            encoder_attention_mask = atts_mask,
                                            return_dict = True,
                                           ).last_hidden_state # [b * n, seq_len, feature_dim]
        mixed_embed = mixed_embed[:, 0, :]  # [b * n, feature_dim]
        mixed_embed = mixed_embed.contiguous().view(b, n, -1)  
            
        # score data
        reward = self.scorer(mixed_embed)
        reward = (reward - self.prior_mean) / self.prior_std
        
        return reward
    
    