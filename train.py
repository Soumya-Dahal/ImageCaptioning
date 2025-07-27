import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dataset import CocoDataset
from tokenizer import SentencePieceTokenizer
from models import ImageCaptioningModel
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import os
import config
import validate

def train():
    tokenizer = SentencePieceTokenizer(config.TOKENIZER_MODEL_PATH)
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CocoDataset(
        config.ANNOTATIONS_PATH, config.IMAGES_PATH, tokenizer, config.MAX_SEQ_LEN, transform
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

    model = ImageCaptioningModel(config.VOCAB_SIZE).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / config.WARMUP_STEPS, 1.0))
    scaler = GradScaler()

    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [TRAIN]")

        for images, captions in loop:
            images, captions = images.to(config.DEVICE), captions.to(config.DEVICE)
            tgt_input, tgt_output = captions[:, :-1], captions[:, 1:]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(config.DEVICE)
            tgt_key_padding_mask = (tgt_input == tokenizer.pad_id()).to(config.DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(images, tgt_input, tgt_mask, tgt_key_padding_mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(avg_loss=total_loss / (loop.n + 1))

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        val_loss = validate.validate_model(model, tokenizer)
        print(f"Epoch {epoch+1}: Validation Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving new best model (val_loss={val_loss:.4f})")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss
            }, os.path.join(config.MODEL_SAVE_PATH, "best_checkpoint.pth"))

        if (epoch + 1) % 5 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss
            }, os.path.join(config.MODEL_SAVE_PATH, f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train()