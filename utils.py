# utils.py

# Placeholder for BLEU / METEOR scoring utils etc.

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)

