
from typing import Dict
from typing import List
from typing import Set
from pathlib import Path

from functools import reduce
from tqdm import tqdm

import numpy as np
import scipy as sp
from scipy import special

import MeCab
mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

w2i: Dict[str, int] = {}
i2w: Dict[int, str] = {}

target = set(["dokujo-tsushin", "it-life-hack", "sports-watch"])
dirs: List[Path] = [_dir for _dir in Path("text").glob("*") if _dir.is_dir() and _dir.name in target]

text_file: Path
raw: List[str] = []
for _dir in dirs:
    for text_file in tqdm(list(_dir.glob("*.txt"))[:20]):
        doc_tokens: List[str] = []
        with text_file.open("r") as f:
            node = mecab.parseToNode(f.read())
            while node:
                token: str
                if node.feature.split(",")[6] == '*':
                    token = node.surface
                else:
                    token = node.feature.split(",")[6]
                part = node.feature.split(",")[0]
                if part in set(["名詞", "形容詞", "動詞"]):
                    doc_tokens.append(token)
                node = node.next
            # results = tokenize_ja(f.read())
            # for sentence in results.sentences:
            #     for token in sentence.tokens:
            #         doc_tokens.append(token.words[0].text)
        raw.append(" ".join(doc_tokens))

unique_words: Set[str] = set(reduce(lambda x, y: x + y, map(str.split, raw)))

for _word in unique_words:
    index: int = len(w2i)
    w2i[_word] = index
    i2w[index] = _word



docs: List[List[int]] = list(map(lambda sentence: [w2i[word] for word in sentence], map(str.split, raw)))

N_W: int = len(w2i)    # 語彙数
N_D: int = len(docs)   # ドキュメント数
N_K: int = 3           # トピック数
alpha: float = 1
beta: float = 1

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





    



