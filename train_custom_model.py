import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_vit_decoder import ImageCaptioningModel
import config
try:
    from sentencepiece import SentencePieceProcessor
except ImportError as e:
    print(f"SentencePiece import error: {e}")
    raise
from dataset import CocoDataset
from utils import compute_bleu
import os
import torch.cuda.amp as amp
from tqdm import tqdm

def create_causal_mask(seq_len, device):
    """Create a causal mask for the decoder."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

def decode_without_special_tokens(tokenizer, token_ids):
    """Decode token IDs, removing special tokens."""
    special_tokens = {tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()}
    # Filter out special tokens
    filtered_tokens = [tid for tid in token_ids if tid not in special_tokens]
    # Decode remaining tokens
    return tokenizer.decode(filtered_tokens)

def validate_model(model, val_dataloader, tokenizer, device):
    """Validate the model and compute BLEU-4 score."""
    model.eval()
    bleu_scores = []
    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, desc="Validating", leave=False)
        for images, captions in progress_bar:
            images, captions = images.to(device), captions.to(device)
            generated = model.generate(images, tokenizer, max_length=config.MAX_SEQ_LEN, device=device)
            for gen, ref in zip(generated, captions):
                gen_text = decode_without_special_tokens(tokenizer, gen.cpu().tolist())
                ref_text = decode_without_special_tokens(tokenizer, ref.cpu().tolist())
                if gen_text and ref_text:
                    bleu = compute_bleu(ref_text, gen_text)
                    bleu_scores.append(bleu)
        progress_bar.close()
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

def get_latest_checkpoint(checkpoint_dir, prefix):
    """Find the checkpoint with the highest epoch number."""
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    epochs = [int(f.split('_epoch_')[1].split('.pth')[0]) for f in checkpoint_files if 'epoch' in f]
    if not epochs:
        return None
    latest_epoch = max(epochs)
    return os.path.join(checkpoint_dir, f"{prefix}_epoch_{latest_epoch}.pth")

def train_model(model, train_dataloader, val_dataloader, tokenizer, epochs=config.PRETRAIN_EPOCHS, device=config.DEVICE, lr=config.LEARNING_RATE, is_finetune=False):
    """Train the model with mixed precision and save checkpoints."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=config.WARMUP_STEPS)
    scaler = amp.GradScaler(enabled=config.USE_AMP)
    
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    best_bleu = 0.0
    start_epoch = 0

    checkpoint_path = None
    prefix = 'finetune' if is_finetune else 'pretrain'
    if is_finetune and os.path.exists(config.PRETRAINED_MODEL_PATH):
        checkpoint_path = config.PRETRAINED_MODEL_PATH
    elif not is_finetune:
        checkpoint_path = get_latest_checkpoint(config.MODEL_SAVE_PATH, prefix)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_bleu = checkpoint.get('bleu', 0.0)
        print(f"Resumed from checkpoint: {checkpoint_path}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for batch_idx, (images, captions) in enumerate(progress_bar):
            images, captions = images.to(device), captions.to(device)
            if batch_idx == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: images shape: {images.shape}, captions shape: {captions.shape}")
            optimizer.zero_grad()
            
            seq_len = captions.size(1)
            mask = create_causal_mask(seq_len - 1, device)
            
            with amp.autocast(enabled=config.USE_AMP):
                outputs = model(images, captions[:, :-1], mask)
                if batch_idx == 0:
                    print(f"Outputs shape: {outputs.shape}, contiguous: {outputs.is_contiguous()}")
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        progress_bar.close()
        avg_loss = total_loss / len(train_dataloader)
        bleu_score = validate_model(model, val_dataloader, tokenizer, device)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, BLEU-4: {bleu_score:.4f}")
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'bleu': bleu_score
        }
        torch.save(checkpoint, os.path.join(config.MODEL_SAVE_PATH, f'{prefix}_epoch_{epoch+1}.pth'))
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save(checkpoint, os.path.join(config.MODEL_SAVE_PATH, f'best_{prefix}.pth'))
            if not is_finetune:
                torch.save(checkpoint, config.PRETRAINED_MODEL_PATH)
        
        scheduler.step()

if __name__ == "__main__":
    try:
        tokenizer = SentencePieceProcessor(model_file=config.TOKENIZER_MODEL_PATH)
    except Exception as e:
        print(f"Tokenizer initialization error: {e}")
        raise
    
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomCrop(config.IMAGE_SIZE[0]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CocoDataset(
        annotations_file=config.ANNOTATIONS_PATH,
        images_dir=config.IMAGES_PATH,
        tokenizer=tokenizer,
        max_seq_len=config.MAX_SEQ_LEN,
        transform=train_transform,
        max_samples=1000
    )
    val_dataset = CocoDataset(
        annotations_file=config.VAL_ANNOTATIONS_PATH,
        images_dir=config.VAL_IMAGES_PATH,
        tokenizer=tokenizer,
        max_seq_len=config.MAX_SEQ_LEN,
        transform=val_transform,
        max_samples=1000
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = ImageCaptioningModel(
        vocab_size=config.VOCAB_SIZE,
        img_size=config.IMAGE_SIZE[0],
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM,
        dec_embed_dim=config.DEC_EMBED_DIM
    )
    
    train_model(model, train_dataloader, val_dataloader, tokenizer, epochs=config.PRETRAIN_EPOCHS, lr=config.LEARNING_RATE, is_finetune=False)

