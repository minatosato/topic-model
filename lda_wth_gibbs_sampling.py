
from typing import Dict
from typing import List
from typing import Set
from pathlib import Path

from functools import reduce
from tqdm import tqdm

import numpy as np

import stanfordnlp #stanfordnlp.download('ja')
tokenize_ja = stanfordnlp.Pipeline(lang='ja', processors="tokenize")


w2i: Dict[str, int] = {}
i2w: Dict[int, str] = {}

target = set(["dokujo-tsushin", "it-life-hack", "sports-watch"])
dirs: List[Path] = [_dir for _dir in Path("text").glob("*") if _dir.is_dir() and _dir.name in target]

text_file: Path
raw: List[str] = []
for _dir in dirs:
    for text_file in tqdm(list(_dir.glob("*.txt"))[:10]):
        doc_tokens: List[str] = []
        with text_file.open("r") as f:
            results = tokenize_ja(f.read())
            for sentence in results.sentences:
                for token in sentence.tokens:
                    doc_tokens.append(token.words[0].text)
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
W: int = sum(map(len, docs)) # total number of words
alpha = 1
beta = 0.001

Z: List[np.ndarray] = list(map(lambda x: np.zeros(len(x)), docs))

n_d_k: np.ndarray = np.zeros(shape=(N_D, N_K)) # ドキュメントdのトピックkのカウント
n_k_w: np.ndarray = np.zeros(shape=(N_K, N_W)) # トピックkの単語wのカウント
n_k: np.ndarray = np.zeros(shape=(N_K, ))      # トピックkのカウント


# initialize
for i in range(len(docs)):
    doc = docs[i]

    for j in range(len(doc)):
        word = doc[j]
        topic: int = np.random.choice(range(N_K), 1)[0]

        Z[i][j] = topic
        n_k_w[topic, word] += 1
        n_k[topic] += 1

    for k in range(N_K):
        n_d_k[i, k] = (Z[i] == k).sum()

for iteration in tqdm(range(1000)):
    for i in range(len(docs)):
        doc = docs[i]

        for j in range(len(doc)):
            word = docs[i][j]
            topic = int(Z[i][j])

            n_d_k[i, topic] -= 1
            n_k_w[topic, word] -= 1
            n_k[topic] -= 1

            prob = np.zeros(shape=(N_K, ))
            for k in range(N_K):
                prob[k] = (n_d_k[i, k] + alpha) * (n_k_w[k, word] + beta) / (n_k[k] + beta*W)
            
            prob /= prob.sum()
            topic = np.random.multinomial(1, prob).argmax()
            
            Z[i][j] = topic
            n_d_k[i, topic] += 1
            n_k_w[topic, word] += 1
            n_k[topic] += 1


