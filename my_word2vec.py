import jieba
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


# 加载停用词
def load_stop_words(file="stopwords.txt"):
    with open(file, "r", encoding='utf-8') as f:
        stopwords_list = f.read().split('\n')
        return stopwords_list


# 分词
def cut_words(file="数学原始数据.csv"):
    all_data = pd.read_csv(file, encoding='gbk', names=['data'])['data']
    stopwords_list = load_stop_words()
    result = []
    for sentence in all_data:  # 遍历每行句子
        c_words = jieba.lcut(sentence)  # 给每行句子分词
        result.append([word for word in c_words if word not in stopwords_list])  # 判断是否属于停用词
    return result  # 获得分词后的数据列表


# 三大参数 word2id,id2word,word2onehot
def get_dict():
    result = cut_words()  # 获得分词后的数据列表
    id2word = []  # 列表的下标就是id
    for sentence in result:
        for word in sentence:
            if word not in id2word:  # 给result去掉重复词
                id2word.append(word)
    word2id = {word: id for id, word in enumerate(id2word)}
    word2onehot = {}
    for word, id in word2id.items():
        onehot = np.zeros((1, len(id2word)))
        onehot[0][id] = 1
        word2onehot[word] = onehot
    return word2id, id2word, word2onehot


if __name__ == "__main__":
    result = cut_words()
    word2id, id2word, word2onehot = get_dict()

    word_size = len(word2id)  # 总共的词数
    embedding_size = 107
    lr = 0.01
    epoch = 10
    n_gram = 3  # 其他词的范围

    # 初始化权重参数w1,w2
    w1 = np.random.normal(-1, 1, size=(word_size, embedding_size))
    w2 = np.random.normal(-1, 1, size=(embedding_size, word_size))

    for e in range(epoch):
        for sentence in tqdm(result):
            for n_index, now_word in enumerate(sentence):
                now_word_onehot = word2onehot[now_word] # 当前词
                other_words = sentence[max(0, n_index - n_gram):n_index] + sentence[n_index + 1:n_index + 1 + n_gram] # 其他词
                for other_word in other_words:
                    other_word_onehot = word2onehot[other_word]
                    # 模型的forword
                    hidden = now_word_onehot @ w1
                    p = hidden @ w2
                    pre = softmax(p)
                    # 反向
                    G2 = pre - other_word_onehot
                    delta_w2 = hidden.T @ G2
                    G1 = G2 @ w2.T
                    delta_w1 = now_word_onehot.T @ G1
                    # 更新梯度
                    w1 = w1 - lr * delta_w1
                    w2 = w2 - lr * delta_w2

    # 把训练好的模型进行存储
    with open("word2vec.pkl", 'wb') as f:
        pickle.dump([w1, word2id, id2word], f)
