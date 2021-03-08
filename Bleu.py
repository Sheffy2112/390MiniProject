# test document function

# one word different
from nltk.translate.bleu_score import sentence_bleu

import warnings

warnings.filterwarnings("ignore")

# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu

reference = [['your', 'life', 'will', 'never', 'be', "the"]]
candidate = ['your', 'life', 'will', 'never', 'be', "better"]



# cumulative BLEU scores


def bleuScore(initial, compare):
    score = sentence_bleu(reference, compare, weights=(1, 0, 0, 0))
    print('Cumulative 1-gram: %f' % sentence_bleu(reference, compare, weights=(0.25, 0.25, 0.25, 0.25)))


bleuScore(reference, candidate)
