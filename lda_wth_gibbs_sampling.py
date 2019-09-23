# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict
from typing import List

from functools import reduce
from tqdm import tqdm

import numpy as np
import scipy as sp
from scipy.special import digamma

from livedoor_news_corpus import LivedoorNewsCorpus

import argparse
parser = argparse.ArgumentParser(description='LDA')
parser.add_argument('--log', type=str, default="./bpr-log")
parser.add_argument('--stop_words_file', type=str, default="stop_words.txt")
parser.add_argument('--limit', type=int, default=20)
parser.add_argument('--iter', type=int, default=100)

args = parser.parse_args()

corpus: LivedoorNewsCorpus = LivedoorNewsCorpus(
    stop_words_path=args.stop_words_file, limit=args.limit)

class LDA(object):
    def __init__(self,
                 docs: List[List[int]], 
                 i2w: Dict[int, str], 
                 num_topics: int = 3, 
                 iterations: int = 100, 
                 alpha: float = 0.1, 
                 beta: float = 0.1
                ):
        self.N_W: int = len(i2w)    # 語彙数
        self.N_D: int = len(docs)   # ドキュメント数
        self.N_K: int = num_topics  # トピック数

        self.docs: List[List[int]] = docs
        self.i2w: Dict[int, str] = i2w

        self.alpha: float = alpha
        self.beta: float = beta

        print("####### LDA #######")
        print(f"VOCABLARY SIZE: {self.N_W}")
        print(f"NUMBER OF DOCUMENTS: {self.N_D}")
        print(f"NUMBER OF TOPICS: {self.N_K}")
        print(f"NUMBER OF ITERATIONS: {iterations}")
        print(f"ALPHA: {self.alpha}")
        print(f"BETA: {self.beta}")

        self.history: Dict[str, np.ndarray] = {}

        self.initialize_counters()
        self.gibbs_sampling(iterations)

        print(f"FINAL ALPHA: {self.alpha}")
        print(f"FINAL BETA: {self.beta}")
        for i in range(self.N_K):
            self.print_topic(i)
        print("###################")

    def initialize_counters(self):
        self.Z: List[np.ndarray] = list(map(lambda x: np.zeros(len(x)), self.docs))
        self.n_d_k: np.ndarray = np.zeros(shape=(self.N_D, self.N_K)) # ドキュメントdのトピックkのカウント
        self.n_k_w: np.ndarray = np.zeros(shape=(self.N_K, self.N_W)) # トピックkの単語wのカウント
        self.n_k: np.ndarray = np.zeros(shape=(self.N_K, ))           # トピックkのカウント
        for i in range(len(self.docs)):    
            for j in range(len(self.docs[i])):
                word = self.docs[i][j]
                topic: int = np.random.choice(range(self.N_K), 1)[0]
                self.Z[i][j] = topic
                self.n_k_w[topic, word] += 1
                self.n_k[topic] += 1
            for k in range(self.N_K):
                self.n_d_k[i, k] = (self.Z[i] == k).sum()

    def perplexity(self):
        word_distribution_for_topics: np.ndarray = (self.n_k_w + self.beta) / (self.n_k[:, None] + self.beta * self.N_W)
        sum_log_prob: float = 0.0
        num_of_total_words: int = 0
        for i in range(len(self.docs)):
            topic_ditribution_for_doc: np.ndarray = (self.n_d_k[i] + self.alpha) / (self.n_d_k[i].sum() + self.alpha * self.N_K) # このドキュメントのトピック分布
            for j in range(len(self.docs[i])):
                topic_ditribution_for_word: np.ndarray = word_distribution_for_topics[:, self.docs[i][j]]
                log_prob = np.log(topic_ditribution_for_word * topic_ditribution_for_doc)
                prob: float = np.dot(topic_ditribution_for_word, topic_ditribution_for_doc)
                sum_log_prob += np.log(prob)
            num_of_total_words += len(self.docs[i])
        return np.exp(- (1/num_of_total_words) * sum_log_prob)

    def decrement_counters(self, topic: int, doc: int, word: int) -> None:
        self.n_d_k[doc, topic] -= 1
        self.n_k_w[topic, word] -= 1
        self.n_k[topic] -= 1

    def increment_counters(self, topic: int, doc: int, word: int) -> None:
        self.n_d_k[doc, topic] += 1
        self.n_k_w[topic, word] += 1
        self.n_k[topic] += 1

    def resampling_topic(self, doc: int, word: int) -> int:
        prob: np.ndarray = (self.n_d_k[doc, :] + self.alpha) * (self.n_k_w[:, word] + self.beta) / (self.n_k + self.beta*self.N_W)
        prob /= prob.sum()
        topic: int = np.random.multinomial(1, prob).argmax()
        return topic

    def gibbs_sampling(self, iterations: int = 100) -> None:
        self.n_d: np.ndarray = np.array(list(map(len, self.docs))) 

        perplexity_history: List[float] = []

        with tqdm(total=iterations, leave=True, ncols=100) as progress:
            for iteration in range(iterations):
                for i in range(len(self.docs)):
                    for j in range(len(self.docs[i])):
                        word = self.docs[i][j]
                        topic = int(self.Z[i][j])

                        self.decrement_counters(topic, i, word)
                        topic = self.resampling_topic(i, word)
                        
                        self.Z[i][j] = topic
                        self.increment_counters(topic, i, word)

                bunshi = digamma(self.n_d_k + self.alpha).sum() - self.N_D * self.N_K * digamma(self.alpha)
                bunbo = self.N_K * digamma(self.n_d + self.alpha*self.N_K).sum() - self.N_D * self.N_K * digamma(self.alpha*self.N_K)
                self.alpha = self.alpha * bunshi / bunbo

                bunshi = digamma(self.n_k_w + self.beta).sum() - self.N_K * self.N_W * digamma(self.beta)
                bunbo = self.N_W * digamma(self.n_k + self.beta*self.N_W).sum() - self.N_K * self.N_W * digamma(self.beta*self.N_W)
                self.beta = self.beta * bunshi / bunbo

                perplexity_history.append(self.perplexity())
                
                description_list: List[str] = []
                description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
                description_list.append(f"CURRENT PERPLEXITY: {np.round(perplexity_history[-1], 3)}")
                progress.set_description(', '.join(description_list))
                progress.update(1)
        


        self.history["perplexity"] = np.array(perplexity_history)

    def print_topic(self, i: int, num_words: int = 20):
        tmp = self.n_k_w[i].copy()
        tmp /= tmp.sum()

        indices: np.ndarray = self.n_k_w[i].argsort()[::-1][:num_words]
        print(f"topic #{i}: ", end="")
        for index in indices:
            print(f"{np.round(tmp[index], 3)}*{self.i2w[index]}", end="")
            if index != indices[-1]:
                print(" + ", end="")
        print("")

lda = LDA(corpus.docs, corpus.i2w, num_topics=3, iterations=args.iter)


# http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt