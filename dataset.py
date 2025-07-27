import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    def __init__(self, annotations_file, images_dir, tokenizer, max_seq_len, transform=None, max_samples=None):
        """Initialize COCO dataset for image captioning."""
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)
        self.annotations = self.data['annotations']
        # Limit to max_samples if specified
        if max_samples is not None:
            self.annotations = self.annotations[:max_samples]
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Load and preprocess image and caption."""
        ann = self.annotations[idx]
        img_id = ann['image_id']
        img_file = f"{img_id:012d}.jpg"
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        caption = ann['caption']
        tokens = self.tokenizer.encode(caption)
        tokens = tokens[:self.max_seq_len - 1] + [self.tokenizer.eos_id()]  # Truncate and add <EOS>
        tokens = tokens + [self.tokenizer.pad_id()] * (self.max_seq_len - len(tokens))  # Pad
        
        if idx == 0:  # Log first item for debugging
            print(f"Image shape: {image.shape}, Tokens length: {len(tokens)}")
        
        return image, torch.tensor(tokens, dtype=torch.long)
