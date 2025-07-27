import sentencepiece as spm
import os
import config

class SentencePieceTokenizer:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model {model_path} not found")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        if self.sp.get_piece_size() != config.VOCAB_SIZE:
            print(f"Warning: Tokenizer vocab size {self.sp.get_piece_size()} != config.VOCAB_SIZE {config.VOCAB_SIZE}")

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

    def bos_id(self):
        return self.sp.bos_id()

    def eos_id(self):
        return self.sp.eos_id()

    def pad_id(self):
        return self.sp.pad_id()

    def vocab_size(self):
        return self.sp.get_piece_size()