# 
# Copyright (c) 2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

from pathlib import Path
from typing import Optional
from typing import Dict
from typing import List
from typing import Set

from functools import reduce
from tqdm import tqdm

import MeCab

class LivedoorNewsCorpus(object):
    root: Path
    stop_words_path: Optional[str]
    stop_words: Set[str]

    vocab: Set[str]
    w2i: Dict[str, int]
    i2w: Dict[int, str]

    raw: List[str]
    docs: List[List[int]]

    def __init__(self, 
                 stop_words_path: Optional[str] = None,
                 limit: int = 20,
                 target: Set[str] = set(["dokujo-tsushin", "it-life-hack", "sports-watch"])
                 ):
        self.root = Path("./text")
        self.stop_words_path = stop_words_path
        self.stop_words = set()
        self.w2i = {}
        self.i2w = {}

        extract_parts: Set[str] = set(["名詞", "形容詞", "動詞"])

        if not self.root.exists():
            fname: str = "ldcc-20140209.tar.gz"
            if not Path(fname).exists():
                os.system(f"wget https://www.rondhuit.com/download/{fname}")
            os.system(f"tar -zxvf {fname}")
        
        if self.stop_words_path is not None:
            if not Path(self.stop_words_path).exists():
                raise FileNotFoundError(f"stop_words_path: {self.stop_words_path} does not exist.")
            with Path(self.stop_words_path).open("r") as f:
                for _word in f.read().split("\n"):
                    self.stop_words.add(_word)

        ignore_file_name: str = "LICENSE.txt"
        dirs: List[Path] = [_dir for _dir in self.root.glob("*") if _dir.is_dir() and _dir.name in target]

        mecab = MeCab.Tagger()

        text_file: Path
        self.raw = []
        for _dir in dirs:
            file_list: List[Path] = [i for i in _dir.glob("*.txt") if i.name != ignore_file_name]
            for text_file in file_list[:limit]:
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
                        if part in extract_parts and token not in self.stop_words:
                            doc_tokens.append(token)
                        node = node.next
                self.raw.append(" ".join(doc_tokens))
        
        self.vocab = set(reduce(lambda x, y: x + y, map(str.split, self.raw)))
        for _word in self.vocab:
            index: int = len(self.w2i)
            self.w2i[_word] = index
            self.i2w[index] = _word
        
        self.docs = list(map(lambda doc: [self.w2i[word] for word in doc], map(str.split, self.raw)))
