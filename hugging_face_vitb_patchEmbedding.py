from transformers import ViTModel
import torch

# Load pretrained ViT-Base model
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

# Extract PatchEmbedding weights
patch_embed_state_dict = {
    'projection.weight': model.embeddings.patch_embeddings.projection.weight,
    'projection.bias': model.embeddings.patch_embeddings.projection.bias,
    'cls_token': model.embeddings.cls_token,
    'pos_embed': model.embeddings.position_embeddings
}

# Save to file
torch.save(patch_embed_state_dict, 'pretrained_patch_embed_hf_vit_b16.pth')
print("Saved pretrained PatchEmbedding weights to pretrained_patch_embed_hf_vit_b16.pth")
