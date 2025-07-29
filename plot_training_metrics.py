import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import config
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_vit_decoder import ImageCaptioningModel
from dataset import CocoDataset
from utils import compute_bleu
from tqdm import tqdm
from sentencepiece import SentencePieceProcessor

def decode_without_special_tokens(tokenizer, token_ids):
    """Decode token IDs, removing special tokens."""
    special_tokens = {tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()}
    filtered_tokens = [tid for tid in token_ids if tid not in special_tokens]
    return tokenizer.decode(filtered_tokens)

def validate_model(model, val_dataloader, tokenizer, device, return_individual=False):
    """Validate model and return BLEU-4 scores."""
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
                    if debug_count < 10:  # Print first 10 captions for debugging
                        print(f"Sample {debug_count+1}: Generated='{gen_text}', Reference='{ref_text}'")
                        debug_count += 1
                    if gen_text and ref_text:
                        bleu = compute_bleu(ref_text, gen_text)
                        if bleu > 0.0:  # Skip invalid BLEU scores
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
    return bleu_scores if return_individual else avg_bleu

def extract_metrics(checkpoint_dir, prefix="pretrain"):
    """Extract loss and BLEU-4 from all epoch checkpoints."""
    losses = []
    bleu_scores = []
    epochs = []
    latest_checkpoint = None
    max_epoch = 0
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"{prefix}_epoch_") and f.endswith('.pth')]
    checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
    
    for file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, file)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', None)
            bleu = checkpoint.get('bleu', None)
            if loss is not None and bleu is not None and bleu >= 0 and not np.isnan(loss) and not np.isnan(bleu):
                epochs.append(epoch)
                losses.append(loss)
                bleu_scores.append(bleu)
                print(f"Checkpoint {file}: Epoch {epoch}, Loss: {loss:.4f}, BLEU-4: {bleu:.4f}")
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_checkpoint = checkpoint_path
            else:
                print(f"Skipping {file}: Invalid loss ({loss}) or BLEU ({bleu})")
        except Exception as e:
            print(f"Failed to load {file}: {e}")
    
    return epochs, losses, bleu_scores, latest_checkpoint

def get_learning_rate_schedule(epochs, lr=config.LEARNING_RATE, warmup_steps=config.WARMUP_STEPS, steps_per_epoch=None):
    """Reconstruct learning rate schedule."""
    if steps_per_epoch is None:
        steps_per_epoch = 20000 // 8  # 20,000 images / batch_size=8
    total_steps = steps_per_epoch * epochs
    lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr_factor = 0.1 + 0.9 * (step / warmup_steps)  # Linear warmup from 0.1*lr to lr
        else:
            lr_factor = 1.0  # Constant after warmup
        lrs.append(lr * lr_factor)
    
    epoch_lrs = [sum(lrs[i * steps_per_epoch:(i + 1) * steps_per_epoch]) / steps_per_epoch for i in range(epochs)]
    return epoch_lrs

def plot_metrics(epochs, losses, bleu_scores, learning_rates, bleu_distribution, checkpoint_dir):
    """Generate evaluation plots."""
    # 1. Training Loss vs. Epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.title('Training Loss vs. Epoch (Encoder-Only, 20,000 Images)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'training_loss_vs_epoch.png'))
    plt.close()

    # 2. Validation BLEU-4 vs. Epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, bleu_scores, marker='o', linestyle='-', color='g', label='BLEU-4 Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-4 Score')
    plt.title('Validation BLEU-4 Score vs. Epoch (Encoder-Only, 20,000 Images)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'bleu4_vs_epoch.png'))
    plt.close()

    # 3. Training Loss vs. Validation BLEU-4 (Scatter)
    plt.figure(figsize=(8, 6))
    plt.scatter(losses, bleu_scores, c=epochs, cmap='viridis', label='Epoch')
    plt.colorbar(label='Epoch')
    plt.xlabel('Average Training Loss')
    plt.ylabel('BLEU-4 Score')
    plt.title('Training Loss vs. BLEU-4 Score (Encoder-Only, 20,000 Images)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_vs_bleu4.png'))
    plt.close()

    # 4. Learning Rate vs. Epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, learning_rates, marker='o', linestyle='-', color='r', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs. Epoch (Encoder-Only, 20,000 Images)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'learning_rate_vs_epoch.png'))
    plt.close()

    # 5. Validation BLEU-4 Distribution (Histogram)
    plt.figure(figsize=(8, 6))
    plt.hist(bleu_distribution, bins=20, color='purple', alpha=0.7)
    plt.xlabel('BLEU-4 Score')
    plt.ylabel('Frequency')
    plt.title('Validation BLEU-4 Distribution (Latest Checkpoint, 20,000 Images)')
    plt.grid(True)
    plt.savefig(os.path.join(checkpoint_dir, 'bleu4_distribution.png'))
    plt.close()

if __name__ == "__main__":
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
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
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

    # Load checkpoint (best or latest)
    checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, 'best_pretrain.pth')
    checkpoint_dir = config.MODEL_SAVE_PATH
    epochs, losses, bleu_scores, latest_checkpoint = extract_metrics(checkpoint_dir, prefix="pretrain")
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded best checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load best checkpoint: {e}")
            checkpoint_path = None
    else:
        print(f"Best checkpoint not found: {checkpoint_path}")
        checkpoint_path = latest_checkpoint

    if checkpoint_path and latest_checkpoint:
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded latest checkpoint: {latest_checkpoint}")
        except Exception as e:
            print(f"Failed to load latest checkpoint: {latest_checkpoint}: {e}")
            exit()
    elif not latest_checkpoint:
        print(f"No epoch checkpoints found in {checkpoint_dir}")
        exit()

    # Get BLEU-4 distribution
    bleu_distribution = validate_model(model, val_dataloader, tokenizer, device, return_individual=True)
    if not bleu_distribution:
        print("No valid BLEU-4 scores obtained from validation")
        exit()

    # Reconstruct learning rate schedule
    steps_per_epoch = 20000 // 8  # 20,000 images / batch_size=8
    learning_rates = get_learning_rate_schedule(len(epochs), steps_per_epoch=steps_per_epoch)

    # Generate plots
    if epochs:
        plot_metrics(epochs, losses, bleu_scores, learning_rates, bleu_distribution, checkpoint_dir)
        print(f"Plots saved in {checkpoint_dir}:")
        print("- training_loss_vs_epoch.png")
        print("- bleu4_vs_epoch.png")
        print("- loss_vs_bleu4.png")
        print("- learning_rate_vs_epoch.png")
        print("- bleu4_distribution.png")
    else:
        print(f"No valid checkpoints found in {checkpoint_dir}")