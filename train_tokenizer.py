import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input='captions.txt',
    model_prefix='tokenizer',
    vocab_size=30000,
    character_coverage=1.0,
    model_type='bpe',
    bos_id=3,
    eos_id=4,
    pad_id=5,
    bos_piece='<BOS>',
    eos_piece='<EOS>',
    pad_piece='<PAD>'
)
