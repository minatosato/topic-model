import numpy as np
from scipy.special import digamma
import cython
cimport numpy as np
from math import exp
from libc.math cimport exp as c_exp
from cython.parallel import prange
from threading import Thread
from cython.parallel import prange
from cython.parallel import threadid
from tqdm import tqdm


@cython.boundscheck(False)
@cython.wraparound(False)
def gibbs_sampling(list docs, unsigned int vocab_size, unsigned int num_topics):
    cdef unsigned int N_K = num_topics
    cdef unsigned int N_W = vocab_size
    cdef unsigned int N_D = len(docs)

    cdef double alpha = 0.01
    cdef double beta = 0.01

    cdef list Z = list(map(lambda x: np.zeros_like(x).astype(np.int32), docs))
    cdef unsigned int i, j, k, word, topic, iteration

    cdef int[:,:] n_d_k = np.zeros(shape=(N_D, N_K)).astype(np.int32)
    cdef int[:,:] n_k_w = np.zeros(shape=(N_K, N_W)).astype(np.int32)
    cdef int[:] n_k = np.zeros(shape=(N_K, )).astype(np.int32)

    cdef np.ndarray[np.double_t, ndim=1] topic_distribution = np.zeros(shape=(N_K, ))
    cdef double[:] doc
    cdef int doc_len
    cdef np.ndarray[np.int_t, ndim=1] topics = np.arange(0, N_K)

    for i in range(N_D):
        doc_len = docs[i].shape[0]
        for j in range(doc_len):
            word = docs[i][j]
            topic = np.random.choice(topics, 1)[0]
            Z[i][j] = topic

            n_k_w[topic, word] += 1
            n_k[topic] += 1

        for k in range(N_K):
            n_d_k[i, k] = (Z[i] == k).sum()
    
    for iteration in tqdm(range(100), ncols=100):
        for i in range(N_D):
            doc_len = docs[i].shape[0]
            for j in range(doc_len):
                word = docs[i][j]
                topic = Z[i][j]

                n_d_k[i, topic] -= 1
                n_k_w[topic, word] -= 1
                n_k[topic] -= 1

                for k in range(N_K):
                    topic_distribution[k] = (n_d_k[i, k] + alpha) * (n_k_w[k, word] + beta) / (n_k[k] + beta * N_W)
                topic_distribution /= topic_distribution.sum()

                topic = np.random.multinomial(1, topic_distribution).argmax()
                Z[i][j] = topic

                n_d_k[i, topic] += 1
                n_k_w[topic, word] += 1
                n_k[topic] += 1

        

            

    