import numpy as np
from scipy.special import digamma
import cython
cimport numpy as np
from math import exp
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
from cython.parallel import prange
from threading import Thread
from cython.parallel import prange
from cython.parallel import threadid
from tqdm import tqdm

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

# cdef extern from "time.h":
#     long int time(int)

srand48(1234)

@cython.boundscheck(False)
@cython.wraparound(False)
def gibbs_sampling(list docs, unsigned int vocab_size, unsigned int num_topics, unsigned int num_iterations):
    cdef unsigned int N_K = num_topics
    cdef unsigned int N_W = vocab_size
    cdef unsigned int N_D = len(docs)
    cdef unsigned int iterations = num_iterations

    cdef double alpha = 0.1
    cdef double beta = 0.1

    cdef list Z = list(map(lambda x: np.zeros_like(x).astype(np.int32), docs))
    cdef unsigned int i, j, k, word, topic, iteration

    cdef int[:,:] n_d_k = np.zeros(shape=(N_D, N_K)).astype(np.int32)
    cdef int[:,:] n_k_w = np.zeros(shape=(N_K, N_W)).astype(np.int32)
    cdef int[:] n_k = np.zeros(shape=(N_K, )).astype(np.int32)
    cdef int[:] n_d = np.zeros(shape=(N_D, )).astype(np.int32)

    cdef int num_of_total_words = 0

    cdef double[:] topic_distribution = np.zeros(shape=(N_K, ))
    cdef double[:] doc
    cdef double total, roulette
    cdef int doc_len
    cdef np.ndarray[np.int_t, ndim=1] topics = np.arange(0, N_K)



    # PPL用
    cdef double[:,:] word_distribution_for_topics
    cdef double[:] topic_ditribution_for_word
    cdef double[:] topic_ditribution_for_doc = np.zeros(shape=(N_K))
    cdef double sum_log_prob
    cdef double prob
    cdef double tmp
    cdef double perplexity

    cdef list description_list

    print("####### LDA #######")
    print(f"VOCABLARY SIZE: {N_W}")
    print(f"NUMBER OF DOCUMENTS: {N_D}")
    print(f"NUMBER OF TOPICS: {N_K}")
    print(f"NUMBER OF ITERATIONS: {100}")
    print(f"ALPHA: {alpha}")
    print(f"BETA: {beta}")

    for i in range(N_D):
        doc_len = docs[i].shape[0]
        for j in range(doc_len):
            word = docs[i][j]
            topic = np.random.choice(topics, 1)[0]
            Z[i][j] = topic

            n_k_w[topic, word] += 1
            n_k[topic] += 1
            n_d_k[i, topic] += 1
        n_d[i] = doc_len
        num_of_total_words += doc_len

    with tqdm(total=iterations, leave=True, ncols=100) as progress:
        for iteration in range(iterations):
            for i in range(N_D):
                doc_len = docs[i].shape[0]
                for j in range(doc_len):
                    word = docs[i][j]
                    topic = Z[i][j]

                    n_d_k[i, topic] -= 1
                    n_k_w[topic, word] -= 1
                    n_k[topic] -= 1

                    total = 0.0
                    for k in range(N_K):
                        topic_distribution[k] = (n_d_k[i, k] + alpha) * (n_k_w[k, word] + beta) / (n_k[k] + beta * N_W)
                        total += topic_distribution[k]
                    roulette = drand48() * total
                    total = 0.0
                    for k in range(N_K):
                        total += topic_distribution[k]
                        if total >= roulette:
                            break
                                
                    topic = k
                    Z[i][j] = topic

                    n_d_k[i, topic] += 1
                    n_k_w[topic, word] += 1
                    n_k[topic] += 1


        
            bunshi = digamma(np.array(n_d_k) + alpha).sum() - N_D * N_K * digamma(alpha)
            bunbo = N_K * digamma(np.array(n_d) + alpha*N_K).sum() - N_D * N_K * digamma(alpha*N_K)
            alpha = alpha * bunshi / bunbo

            bunshi = digamma(np.array(n_k_w) + beta).sum() - N_K * N_W * digamma(beta)
            bunbo = N_W * digamma(np.array(n_k) + beta*N_W).sum() - N_K * N_W * digamma(beta*N_W)
            beta = beta * bunshi / bunbo



            sum_log_prob = 0.0
            word_distribution_for_topics = (np.array(n_k_w) + beta) / (np.array(n_k)[:, None] + beta * N_W)
            for i in range(N_D):

                tmp = 0.0
                for k in range(N_K):
                    tmp += n_d_k[i, k]

                    topic_ditribution_for_doc[k] = n_d_k[i, k] + alpha
                for k in range(N_K):
                    topic_ditribution_for_doc[k] /= (tmp + alpha * N_K)
                # topic_ditribution_for_doc = (np.array(n_d_k[i]) + alpha) / (tmp + alpha * N_K) # このドキュメントのトピック分布
                for j in range(docs[i].shape[0]):
                    topic_ditribution_for_word = word_distribution_for_topics[:, docs[i][j]]
                    # prob = np.dot(topic_ditribution_for_word, topic_ditribution_for_doc)
                    prob = 0.0
                    for k in range(N_K):
                        prob += topic_ditribution_for_doc[k] * topic_ditribution_for_word[k]
                    sum_log_prob += c_log(prob)
            perplexity = c_exp(- (1.0 /num_of_total_words) * sum_log_prob)
        

            description_list = []
            description_list.append(f"ITER={iteration+1:{len(str(iterations))}}")
            description_list.append(f"CURRENT PERPLEXITY: {np.round(perplexity, 3):.3f}")
            progress.set_description(', '.join(description_list))
            progress.update(1)

    print(alpha)
    print(beta)
    return np.array(n_d_k)

