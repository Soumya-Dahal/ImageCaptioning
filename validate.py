import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from dataset import CocoDataset
from tokenizer import SentencePieceTokenizer
from models import ImageCaptioningModel
from torchvision import transforms
import torch.nn as nn
import os
import config

def validate_model(model, tokenizer):
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = CocoDataset(
        config.VAL_ANNOTATIONS_PATH, config.VAL_IMAGES_PATH, tokenizer, config.MAX_SEQ_LEN, transform
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    total_loss = 0.0

    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validating [VAL]")
        for images, captions in loop:
            images, captions = images.to(config.DEVICE), captions.to(config.DEVICE)
            tgt_input, tgt_output = captions[:, :-1], captions[:, 1:]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(config.DEVICE)
            tgt_key_padding_mask = (tgt_input == tokenizer.pad_id()).to(config.DEVICE)

            with autocast():
                logits = model(images, tgt_input, tgt_mask, tgt_key_padding_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    return total_loss / len(val_loader)

if __name__ == "__main__":
    tokenizer = SentencePieceTokenizer(config.TOKENIZER_MODEL_PATH)
    model = ImageCaptioningModel(config.VOCAB_SIZE).to(config.DEVICE)
    checkpoint = torch.load(os.path.join(config.MODEL_SAVE_PATH, "best_checkpoint.pth"), map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    val_loss = validate_model(model, tokenizer)
    print(f"Validation Loss: {val_loss:.4f}")