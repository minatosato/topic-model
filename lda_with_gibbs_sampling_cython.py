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
parser.add_argument('--stop_words_file', type=str, default=None)
parser.add_argument('--limit', type=int, default=20)
parser.add_argument('--iter', type=int, default=100)

args = parser.parse_args()

corpus: LivedoorNewsCorpus = LivedoorNewsCorpus(
    stop_words_path=args.stop_words_file, limit=args.limit)

from gibbs_sampling import gibbs_sampling

ret = gibbs_sampling(corpus.docs, len(corpus.i2w), 3, args.iter)
# print(np.round(ret/ret.sum(axis=1)[:, None]))

