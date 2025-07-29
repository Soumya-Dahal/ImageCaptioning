import torch
import torch.nn as nn
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
from tqdm import tqdm
import os

def decode_without_special_tokens(tokenizer, token_ids):
    """Decode token IDs, removing special tokens."""
    special_tokens = {tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()}
    filtered_tokens = [tid for tid in token_ids if tid not in special_tokens]
    return tokenizer.decode(filtered_tokens)

def validate_model(model, val_dataloader, tokenizer, device):
    """Validate the model and compute BLEU-4 score."""
    model.eval()
    bleu_scores = []
    debug_count = 0
    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, desc="Validating", leave=True)
        for images, captions in progress_bar:
            images, captions = images.to(device), captions.to(device)
            try:
                generated = model.generate(images, tokenizer, max_length=config.MAX_SEQ_LEN, device=device)
                for gen, ref in zip(generated, captions):
                    gen_text = decode_without_special_tokens(tokenizer, gen.cpu().tolist())
                    ref_text = decode_without_special_tokens(tokenizer, ref.cpu().tolist())
                    if debug_count < 10:  # Print first 10 captions
                        print(f"Sample {debug_count+1}: Generated='{gen_text}', Reference='{ref_text}'")
                        debug_count += 1
                    if gen_text and ref_text:
                        bleu = compute_bleu(ref_text, gen_text)
                        if bleu > 0.0:
                            bleu_scores.append(bleu)
                        else:
                            print(f"Zero BLEU-4 for: Generated='{gen_text}', Reference='{ref_text}'")
                    else:
                        print(f"Empty text: Generated='{gen_text}', Reference='{ref_text}'")
            except Exception as e:
                print(f"Validation error in batch: {e}")
        progress_bar.close()
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    print(f"Average BLEU-4: {avg_bleu:.4f}, Valid scores: {len(bleu_scores)}")
    return avg_bleu

def get_latest_checkpoint(checkpoint_dir, prefix="pretrain"):
    """Find the latest epoch checkpoint."""
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"{prefix}_epoch_") and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])

def evaluate_checkpoint(checkpoint_path=None):
    """Evaluate a checkpoint and return BLEU-4 score."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    try:
        tokenizer = SentencePieceProcessor(model_file=config.TOKENIZER_MODEL_PATH)
    except Exception as e:
        print(f"Tokenizer initialization error: {e}")
        raise

    # Load validation dataset
    val_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = CocoDataset(
        annotations_file=config.VAL_ANNOTATIONS_PATH,
        images_dir=config.VAL_IMAGES_PATH,
        tokenizer=tokenizer,
        max_seq_len=config.MAX_SEQ_LEN,
        transform=val_transform,
        max_samples=5000
    )
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Load model
    model = ImageCaptioningModel(
        vocab_size=config.VOCAB_SIZE,
        img_size=config.IMAGE_SIZE[0],
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM,
        dec_embed_dim=config.DEC_EMBED_DIM,
        pretrained_path='pretrained_patch_embed_hf_vit_b16.pth'
    )
    model.to(device)

    # Determine checkpoint to use
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, 'best_pretrain.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Best checkpoint not found: {checkpoint_path}")
            checkpoint_path = get_latest_checkpoint(config.MODEL_SAVE_PATH, prefix="pretrain")
            if not checkpoint_path:
                print(f"No epoch checkpoints found in {config.MODEL_SAVE_PATH}")
                return

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        epoch = checkpoint.get('epoch', 0)
        bleu_score = checkpoint.get('bleu', None)
        print(f"Loaded checkpoint: {checkpoint_path}, Epoch: {epoch}, Saved BLEU-4: {bleu_score if bleu_score is not None else 'N/A'}")
    except Exception as e:
        print(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return

    # Evaluate model
    bleu_score = validate_model(model, val_dataloader, tokenizer, device)
    print(f"Evaluated BLEU-4 score: {bleu_score:.4f}")

if __name__ == "__main__":
    evaluate_checkpoint()
