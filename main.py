from __future__ import unicode_literals
from hazm import *
import pandas as pd



class Preprocessor():

    def __init__(self):
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer() 
        self.stopwords_set = set(stopwords_list())
        
    def filter_stopwords(self,input_tokens):
        return [token for token in input_tokens if token not in self.stopwords_set]
    
    def tokenizer(self,content):
        return word_tokenize(content)
    
    def fnormilizer(self,content):
        return self.normalizer.normalize(content)
    
    def lemmetizer_and_stemmer(self,token):
        res = self.lemmatizer.lemmatize(token)
        res = self.stemmer.stem(res)
        return res

class Positional_index():
    def __init__(self,dataframe):
        self.indexes = {}
        self.titles = {}
        for i in range(len(dataframe)):
            self.titles[i] = dataframe.loc[i].title
    
    def add_and_merge_content(self,tokens,id):
        pass


if __name__ == '__main__':
    dataset = pd.read_excel("C:\\university\\Information Retrival\\Project\\information-retrieval\\IR1_7k_news.xlsx")
    preprocessor = Preprocessor()
    positional_index = Positional_index(dataset)
    for i in range(len(dataset)):
        normalized_content = preprocessor.fnormilizer(dataset.loc[i].content)
        tokens = preprocessor.tokenizer(normalized_content)
        tokens = preprocessor.filter_stopwords(tokens)
        for token in tokens:
