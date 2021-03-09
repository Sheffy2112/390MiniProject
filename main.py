from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
import numpy as np
import matplotlib.pyplot as plt
import pickle
import heapq

import warnings

warnings.filterwarnings("ignore")

# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu

# import text
path = "C:/Users/drewa/OneDrive/Documents/GitHub/390MiniProject/1661-0.txt"

text = open(path, encoding='utf-8').read().lower()

# split dataset into individual words ,ouput of words is a python list

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

# unique sorted words list
unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

# feature Engineering part

WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])

# one hot encoding

X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1

model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb"))


def prepare_input(text):
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    for t, word in enumerate(text.split()):
        # print(word)
        x[0, t, unique_word_index[word]] = 1
    return x


prepare_input("It is not a lack".lower())


def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completions(text, n=3):
    if text == "":
        return ("0")
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]




def results(sentence):
    res = 0

    end = len(sentence.split()) - 3
    seq = " ".join(tokenizer.tokenize(sentence.lower())[0:end])
    reference = " ".join(tokenizer.tokenize(sentence.lower())[0:end + 1])
    reference = reference.split()
    candidate = reference.copy()

    candidate = candidate[:-1]
    reference = [reference]
    predict_completion = predict_completions(seq, 5)
    for i in predict_completion:

        candidate = candidate + [i]
        score = bleuScore(reference, candidate)
        candidate = candidate[:-1]

        if score > res: res = score
    return res

    # result should return a missing word trying to be guessed along with a array of the corrected words

q = "Your life will never be the same again"
s = "I have lived in here for ten years"


Data = [q, s]
'''
Grab the function bleuScore:  this method calculates the score comparing the inital sentence with the compare sentence
the var  compare is an list of strings var = ['your', 'life','will', 'never','be', 'yellow']
the var initial is an list of lists of string = [['your', 'life','will', 'never','be', 'the']]
the output will be the resulting score


'''
"""
will need to have the following lines in your code
import warnings

warnings.filterwarnings("ignore")

# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu
"""
def bleuScore(initial, compare):
    score = sentence_bleu(initial, compare, weights=(1, 0, 0, 0))
    return sentence_bleu(initial, compare, weights=(0.25, 0.25, 0.25, 0.25))


def resultsAll(array):
    count, sumScore = 0, 0

    for i in array:
        score = results(i)
        sumScore = sumScore + score
        count += 1

    return sumScore / count


val = resultsAll(Data)
print("the Final Evaluation of the Model Is " + str(val))
