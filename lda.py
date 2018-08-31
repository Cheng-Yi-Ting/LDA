# coding:utf-8
__author__ = "liuxuejiang"
import jieba
import jieba.posseg as pseg
import os
import sys
import math
import json
from collections import OrderedDict

from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class Data():
    def __init__(self):
        self.docs = {}
        self.seg_docs = self.get_seg_docs()
        self.stopword = []

    def read_file(self, path, type):
        # file.read([size])从文件读取指定的字节数，如果未给定或为负则读取所有。
        if type == 'json':
            with open(path, 'r', encoding='utf-8') as file:
                data = json.loads(file.read())
        elif type == 'txt':
            with open(path, 'r', encoding='utf-8') as file:
                data = file.read()
        return data

    def get_seg_docs(self):
        _seg_docs = []
        FOLDER_NAME = 'data'
        # DOCUMENT = 'test.json'
        DOCUMENT = 'ettoday.news.json'
        STOPWORD = 'stopword.txt'
        # 其中__file__虽然是所在.py文件的完整路径，但是这个变量有时候返回相对路径，有时候返回绝对路径，因此还要用os.path.realpath()函数来处理一下。
        # 获取当前文件__file__的路径，    __file__是当前执行的文件
        FILE_DIR = os.path.join(os.path.split(
            os.path.realpath(__file__))[0], FOLDER_NAME)

        self.docs = self.read_file(FILE_DIR + '/' + DOCUMENT, 'json')
        self.stopword = self.read_file(FILE_DIR + '/' + STOPWORD, 'txt')
        self.ca = []
        # jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)，或者用
        # jieba.lcut 以及 jieba.lcut_for_search 直接返回 list
        # isalpha()去除不是字母組成的字，中文字也算，e.g:\r\n，不然會斷出\r\n
        for i in range(len(self.docs)):
            # 計算幾個類別
            # self.ca.append(self.docs[i]['category'])
            content_str = ''
            # content_seg = []
            for w in jieba.lcut(self.docs[i]['content']):
                if len(w) > 1 and w not in self.stopword and w.isalpha():
                    content_str = content_str+' '+w
                    # content_seg.append(w)
            _seg_docs.append(content_str)
        # print(self.ca)
        # category = list(set(self.ca))
        # print(category)
        return _seg_docs


if __name__ == '__main__':
    data = Data()
    # print(data.seg_docs)
    # corpus = ["我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
    #           "他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
    #           "小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
    #           "我 爱 北京 天安门"]  # 第四类文本的切词结果
    corpus = data.seg_docs
    # 將文本中的詞語，轉換成詞頻矩陣
    vectorizer = CountVectorizer()
    # print(vectorizer)
    # 計算詞語出現的頻率
    tf = vectorizer.fit_transform(corpus)
    # print("###################################")
    # 獲取詞袋中所有文本關鍵字，詞袋中所有的字詞
    words = vectorizer.get_feature_names()
    # print(words)
    # print("###################################")
    # print(tf)

    lda = LatentDirichletAllocation(
        n_topics=100, learning_offset=50., random_state=0)
    docres = lda.fit_transform(tf)
    # print("###################################")
    # 文檔-主題分佈矩陣
    print(docres)
    # print("###################################")
    # 主題-詞語分佈矩陣
    print(lda.components_)
    # 印出每個主題下權重教中的字詞
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-20 - 1:-1]]))
