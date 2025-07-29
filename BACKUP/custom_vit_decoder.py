import torch
import torch.nn as nn
import math
import config

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=config.IMAGE_SIZE[0], patch_size=config.PATCH_SIZE, in_channels=3, embed_dim=config.EMBED_DIM):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, n_patches + 1, embed_dim]
        x = x + self.pos_embed
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key=None, value=None, mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        qkv = self.qkv(query).reshape(batch_size, seq_len_q, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = qkv[0]  # [batch_size, num_heads, seq_len_q, head_dim]
        qkv_k = self.qkv(key).reshape(batch_size, seq_len_k, 3, self.num_heads, self.head_dim)
        qkv_k = qkv_k.permute(2, 0, 3, 1, 4)
        k = qkv_k[1]  # [batch_size, num_heads, seq_len_k, head_dim]
        v = qkv_k[2]  # [batch_size, num_heads, seq_len_k, head_dim]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len_q, seq_len_k]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len_q, self.embed_dim)
        out = self.proj(out)
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=config.IMAGE_SIZE[0], patch_size=config.PATCH_SIZE, in_channels=3, embed_dim=config.EMBED_DIM, num_layers=12, num_heads=12, ff_dim=3072):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.patch_embed(x)  # [batch_size, 197, embed_dim]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None):
        x = self.norm1(tgt + self.dropout(self.self_attn(tgt, mask=tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, key=memory, value=memory)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size=config.VOCAB_SIZE, embed_dim=config.DEC_EMBED_DIM, num_layers=6, num_heads=8, ff_dim=2048, max_seq_len=config.MAX_SEQ_LEN):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tgt, memory, tgt_mask=None):
        batch_size, seq_len = tgt.size()
        x = self.token_embed(tgt) + self.pos_embed[:, :seq_len, :]  # [batch_size, seq_len, embed_dim]
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        x = self.norm(x)
        x = self.proj(x)  # [batch_size, seq_len, vocab_size]
        return x.contiguous()  # Ensure contiguous output

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size=config.VOCAB_SIZE, img_size=config.IMAGE_SIZE[0], patch_size=config.PATCH_SIZE, embed_dim=config.EMBED_DIM, dec_embed_dim=config.DEC_EMBED_DIM):
        super().__init__()
        self.encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.projection = nn.Linear(embed_dim, dec_embed_dim)
        self.decoder = TransformerDecoder(vocab_size=vocab_size, embed_dim=dec_embed_dim, max_seq_len=config.MAX_SEQ_LEN)
        
    def forward(self, images, captions, tgt_mask=None):
        memory = self.encoder(images)  # [batch_size, 197, embed_dim]
        memory = self.projection(memory)  # [batch_size, 197, dec_embed_dim]
        outputs = self.decoder(captions, memory, tgt_mask)  # [batch_size, seq_len, vocab_size]
        return outputs

    def generate(self, images, tokenizer, max_length=config.MAX_SEQ_LEN, device=config.DEVICE):
        self.eval()
        with torch.no_grad():
            memory = self.encoder(images)
            memory = self.projection(memory)
            batch_size = images.size(0)
            captions = torch.ones(batch_size, 1, dtype=torch.long, device=device) * tokenizer.bos_id()
            for _ in range(max_length - 1):
                mask = torch.triu(torch.ones(captions.size(1), captions.size(1), device=device), diagonal=1).bool()
                outputs = self.decoder(captions, memory, mask)
                next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
                captions = torch.cat([captions, next_token], dim=1)
                if (next_token == tokenizer.eos_id()).all():
                    break
            return captions

