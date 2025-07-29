import os
import torch

# Image and model dimensions
IMAGE_SIZE = (224, 224)  # Input image size for ViT
PATCH_SIZE = 16  # Patch size for ViT encoder
EMBED_DIM = 768  # Encoder embedding dimension
DEC_EMBED_DIM = 512  # Decoder embedding dimension

# Training hyperparameters
BATCH_SIZE = 8  # Batch size for training
PRETRAIN_EPOCHS = 3  # Epochs for pre-training on COCO
FINETUNE_EPOCHS = 5  # Epochs for fine-tuning
LEARNING_RATE = 1e-4  # Learning rate for pre-training
FINETUNE_LR = 1e-5  # Learning rate for fine-tuning
WARMUP_STEPS = 1000  # Warmup steps for learning rate scheduler

# Sequence and vocabulary settings
MAX_SEQ_LEN = 50  # Maximum caption length
VOCAB_SIZE = 30000  # Vocabulary size for tokenizer

# Device and optimization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
USE_AMP = True  # Enable mixed precision training

# File paths
TOKENIZER_MODEL_PATH = "./tokenizer.model"  # Path to SentencePiece model
ANNOTATIONS_PATH = "./data/annotations/captions_train2017.json"  # COCO train annotations
IMAGES_PATH = "./data/train2017"  # COCO train images
VAL_ANNOTATIONS_PATH = "./data/annotations/captions_val2017.json"  # COCO validation annotations
VAL_IMAGES_PATH = "./data/val2017"  # COCO validation images
MODEL_SAVE_PATH = "./checkpoints"  # Directory for saving checkpoints
PRETRAINED_MODEL_PATH = "./checkpoints/best_pretrain.pth"  # Path for pre-trained model

