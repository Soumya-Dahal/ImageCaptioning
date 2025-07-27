import torch
from sentencepiece import SentencePieceProcessor
from custom_vit_decoder import ImageCaptioningModel
from torchvision import transforms
from PIL import Image
import config
import heapq
import os

def create_causal_mask(seq_len, device):
    """Create a causal mask for the decoder."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

def decode_without_special_tokens(tokenizer, token_ids):
    """Decode token IDs, removing special tokens."""
    special_tokens = {tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()}
    filtered_tokens = [tid for tid in token_ids if tid not in special_tokens]
    return tokenizer.decode(filtered_tokens)

def beam_search(model, image, tokenizer, beam_width=5, max_len=config.MAX_SEQ_LEN, device=config.DEVICE):
    """Generate caption using beam search."""
    model.eval()
    with torch.no_grad():
        # Encode image
        memory = model.encoder(image)  # [1, 197, embed_dim]
        memory = model.projection(memory)  # [1, 197, dec_embed_dim]

        # Initialize beam
        sequences = [([tokenizer.bos_id()], 0.0)]  # List of (sequence, score)

        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == tokenizer.eos_id():
                    all_candidates.append((seq, score))
                    continue

                input_ids = torch.tensor([seq], dtype=torch.long, device=device)  # [1, seq_len]
                mask = create_causal_mask(input_ids.size(1), device)
                logits = model.decoder(input_ids, memory, mask)  # [1, seq_len, vocab_size]
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                topk_probs, topk_ids = torch.topk(probs, beam_width, dim=-1)

                for i in range(beam_width):
                    candidate = seq + [topk_ids[0, i].item()]
                    candidate_score = score + topk_probs[0, i].item()
                    all_candidates.append((candidate, candidate_score))

            # Select top-k candidates
            sequences = heapq.nlargest(beam_width, all_candidates, key=lambda x: x[1])

        # Return best caption
        best_seq, _ = sequences[0]
        caption = decode_without_special_tokens(tokenizer, best_seq[1:])  # Skip <BOS>
        return caption

def generate_caption(image_path, model, tokenizer, beam_width=5, max_len=config.MAX_SEQ_LEN, device=config.DEVICE):
    """Generate caption for a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    caption = beam_search(model, image, tokenizer, beam_width, max_len, device)
    return caption

if __name__ == "__main__":
    try:
        tokenizer = SentencePieceProcessor(model_file=config.TOKENIZER_MODEL_PATH)
    except Exception as e:
        print(f"Tokenizer initialization error: {e}")
        raise

    model = ImageCaptioningModel(
        vocab_size=config.VOCAB_SIZE,
        img_size=config.IMAGE_SIZE[0],
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM,
        dec_embed_dim=config.DEC_EMBED_DIM
    ).to(config.DEVICE)

    # Load checkpoint (try best_pretrain.pth or latest epoch)
    checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, "best_pretrain.pth")
    if not os.path.exists(checkpoint_path):
        # Find latest epoch checkpoint
        checkpoint_files = [f for f in os.listdir(config.MODEL_SAVE_PATH) if f.startswith("pretrain_epoch_") and f.endswith(".pth")]
        if checkpoint_files:
            latest_epoch = max(int(f.split("_epoch_")[1].split(".pth")[0]) for f in checkpoint_files)
            checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, f"pretrain_epoch_{latest_epoch}.pth")
        else:
            raise FileNotFoundError("No checkpoint found in checkpoints directory")

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")

    image_path = "data/test2017/000000000019.jpg"
    try:
        caption = generate_caption(image_path, model, tokenizer, beam_width=5)
        print("Generated Caption:", caption)
    except FileNotFoundError as e:
        print(e)
