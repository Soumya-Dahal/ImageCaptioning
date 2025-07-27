import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_vit_decoder import ImageCaptioningModel
import config
import sentencepiece as spm
from dataset import CocoDataset
from utils import compute_bleu
import os
import torch.cuda.amp as amp

def create_causal_mask(seq_len, device):
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

def validate_model(model, val_dataloader, tokenizer, device):
    model.eval()
    bleu_scores = []
    with torch.no_grad():
        for images, captions in val_dataloader:
            images, captions = images.to(device), captions.to(device)
            generated = model.generate(images, tokenizer, max_length=config.MAX_SEQ_LEN, device=device)
            for gen, ref in zip(generated, captions):
                gen_text = tokenizer.decode(gen.cpu().tolist(), skip_special_tokens=True)
                ref_text = tokenizer.decode(ref.cpu().tolist(), skip_special_tokens=True)
                if gen_text and ref_text:
                    bleu = compute_bleu(ref_text, gen_text)
                    bleu_scores.append(bleu)
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

def finetune_model(model, train_dataloader, val_dataloader, tokenizer, epochs=config.FINETUNE_EPOCHS, device=config.DEVICE):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    optimizer = optim.AdamW(model.parameters(), lr=config.FINETUNE_LR, weight_decay=0.01)
    scaler = amp.GradScaler(enabled=config.USE_AMP)
    
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    best_bleu = 0.0

    # Load pre-trained weights
    if os.path.exists(config.PRETRAINED_MODEL_PATH):
        checkpoint = torch.load(config.PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained weights from {config.PRETRAINED_MODEL_PATH}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, captions) in enumerate(train_dataloader):
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()
            
            seq_len = captions.size(1)
            mask = create_causal_mask(seq_len - 1, device)
            
            with amp.autocast(enabled=config.USE_AMP):
                outputs = model(images, captions[:, :-1], mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), captions[:, 1:].view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:  # More frequent logging for small dataset
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        bleu_score = validate_model(model, val_dataloader, tokenizer, device)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, BLEU-4: {bleu_score:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'bleu': bleu_score
        }
        torch.save(checkpoint, os.path.join(config.MODEL_SAVE_PATH, f'finetune_domain_epoch_{epoch+1}.pth'))
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save(checkpoint, os.path.join(config.MODEL_SAVE_PATH, 'best_finetune_domain.pth'))
        
        # No scheduler for fine-tuning (short training)

if __name__ == "__main__":
    # Tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=config.TOKENIZER_MODEL_PATH)
    
    # Transforms
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
    
    # Domain-specific dataset (update paths)
    DOMAIN_ANNOTATIONS = os.path.join(config.DATA_PATH, "domain_specific", "annotations.json")
    DOMAIN_IMAGES = os.path.join(config.DATA_PATH, "domain_specific", "images")
    DOMAIN_VAL_ANNOTATIONS = os.path.join(config.DATA_PATH, "domain_specific", "val_annotations.json")
    DOMAIN_VAL_IMAGES = os.path.join(config.DATA_PATH, "domain_specific", "val_images")
    
    train_dataset = CocoDataset(
        annotations_file=DOMAIN_ANNOTATIONS,
        images_dir=DOMAIN_IMAGES,
        tokenizer=tokenizer,
        max_seq_len=config.MAX_SEQ_LEN,
        transform=train_transform
    )
    val_dataset = CocoDataset(
        annotations_file=DOMAIN_VAL_ANNOTATIONS,
        images_dir=DOMAIN_VAL_IMAGES,
        tokenizer=tokenizer,
        max_seq_len=config.MAX_SEQ_LEN,
        transform=val_transform
    )
    
    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = ImageCaptioningModel(vocab_size=config.VOCAB_SIZE)
    
    # Fine-tune
    finetune_model(model, train_dataloader, val_dataloader, tokenizer, epochs=config.FINETUNE_EPOCHS, device=config.DEVICE)