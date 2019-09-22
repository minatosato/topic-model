
from typing import Dict
from typing import List

from functools import reduce
from tqdm import tqdm

import numpy as np
import scipy as sp
from scipy import special

from livedoor_news_corpus import LivedoorNewsCorpus

import argparse
parser = argparse.ArgumentParser(description='LDA')
parser.add_argument('--log', type=str, default="./bpr-log")
parser.add_argument('--stop_words_file', type=str, default="stop_words.txt")
parser.add_argument('--limit', type=int, default=20)

args = parser.parse_args()

corpus: LivedoorNewsCorpus = LivedoorNewsCorpus(
    stop_words_path=args.stop_words_file, limit=args.limit)

w2i = corpus.w2i
docs = corpus.docs

N_W: int = len(w2i)    # 語彙数
N_D: int = len(docs)   # ドキュメント数
N_K: int = 3           # トピック数
alpha: float = 0.1
beta: float = 0.1

Z: List[np.ndarray] = list(map(lambda x: np.zeros(len(x)), docs))

n_d_k: np.ndarray = np.zeros(shape=(N_D, N_K)) # ドキュメントdのトピックkのカウント
n_k_w: np.ndarray = np.zeros(shape=(N_K, N_W)) # トピックkの単語wのカウント
n_k: np.ndarray = np.zeros(shape=(N_K, ))      # トピックkのカウント


def perplexity(docs: List[List[int]]):
    word_distribution_for_topics: np.ndarray = (n_k_w + beta) / (n_k[:, None] + beta * N_W)
    sum_log_prob: float = 0.0
    num_of_total_words: int = 0
    for i in range(len(docs)):
        topic_ditribution_for_doc: np.ndarray = (n_d_k[i] + alpha) / (n_d_k[i].sum() + alpha * N_K) # このドキュメントのトピック分布
        for j in range(len(docs[i])):
            topic_ditribution_for_word: np.ndarray = word_distribution_for_topics[:, docs[i][j]]
            log_prob = np.log(topic_ditribution_for_word * topic_ditribution_for_doc)
            prob: float = np.dot(topic_ditribution_for_word, topic_ditribution_for_doc)
            sum_log_prob += np.log(prob)
        num_of_total_words += len(docs[i])
    return np.exp(- (1/num_of_total_words) * sum_log_prob)

# initialize
for i in range(len(docs)):    
    for j in range(len(docs[i])):
        word = docs[i][j]
        topic: int = np.random.choice(range(N_K), 1)[0]

        Z[i][j] = topic
        n_k_w[topic, word] += 1
        n_k[topic] += 1

    for k in range(N_K):
        n_d_k[i, k] = (Z[i] == k).sum()

for iteration in tqdm(range(1000)):
    for i in range(len(docs)):
        for j in range(len(docs[i])):
            word = docs[i][j]
            topic = int(Z[i][j])

            n_d_k[i, topic] -= 1
            n_k_w[topic, word] -= 1
            n_k[topic] -= 1

            prob = np.zeros(shape=(N_K, ))
            for k in range(N_K):
                prob[k] = (n_d_k[i, k] + alpha) * (n_k_w[k, word] + beta) / (n_k[k] + beta*N_W)
            
            prob /= prob.sum()
            topic = np.random.multinomial(1, prob).argmax()
            
            Z[i][j] = topic
            n_d_k[i, topic] += 1
            n_k_w[topic, word] += 1
            n_k[topic] += 1

    bunshi = sum([special.digamma(n_d_k[i, k] + alpha) for k in range(N_K) for i in range(N_D)]) - N_D * N_K * special.digamma(alpha)
    bunbo = N_K * sum([special.digamma(len(d) + alpha*N_K) for d in docs]) - N_D * N_K * special.digamma(alpha*N_K)
    alpha = alpha * bunshi / bunbo

    bunshi = sum([special.digamma(n_k_w[k, i] + beta) for i in range(N_W) for k in range(N_K)]) - N_K * N_W * special.digamma(beta)
    bunbo = N_W * sum([special.digamma(n_k[k] + beta*N_W) for k in range(N_K)]) - N_K * N_W * special.digamma(beta*N_W)
    beta = beta * bunshi / bunbo

    if iteration % 10 == 0:
        print(perplexity(docs))
