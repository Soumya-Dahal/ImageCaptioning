import torch
import torch.nn as nn
from torchvision.models import vit_b_16
import config

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(weights='IMAGENET1K_V1')
        self.vit.heads = nn.Identity()

    def forward(self, images):
        return self.vit(images)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.MAX_SEQ_LEN, d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, tgt_seq, memory, tgt_mask=None, tgt_key_padding_mask=None):
        seq_len = tgt_seq.size(1)
        if seq_len > self.pos_embedding.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max positional embedding length")
        pos_embed = self.pos_embedding[:, :seq_len, :]
        tgt = self.embedding(tgt_seq) + pos_embed
        tgt = tgt.transpose(0, 1)
        memory = memory.unsqueeze(0)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.transpose(0, 1)
        logits = self.fc_out(output)
        return logits

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ViTEncoder()
        self.decoder = TransformerDecoder(vocab_size)

    def forward(self, images, tgt_seq, tgt_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(images)
        logits = self.decoder(tgt_seq, memory, tgt_mask, tgt_key_padding_mask)
        return logits

    def generate_caption(self, image, tokenizer, max_length=config.MAX_SEQ_LEN, device=config.DEVICE):
        self.eval()
        with torch.no_grad():
            image = image.to(device)
            memory = self.encoder(image)
            batch_size = image.size(0)
            tgt_seq = torch.full((batch_size, 1), tokenizer.bos_id(), dtype=torch.long, device=device)
            for _ in range(max_length):
                logits = self.decoder(tgt_seq, memory)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
                if (next_token == tokenizer.eos_id()).all():
                    break
            return tgt_seq