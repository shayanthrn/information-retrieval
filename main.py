from __future__ import unicode_literals
from hazm import *
import pandas as pd
import pickle
import os



class Preprocessor():

    def __init__(self):
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer() 
        self.stopwords_set = set(stopwords_list())
        self.stopwords_set.update(['.',':','?','!','/','//','*','**','***','[',']','{','}',';','\'','\"','(',')',''])
        
    def filter_stopwords(self,input_tokens):
        return [token for token in input_tokens if token not in self.stopwords_set]
    
    def tokenizer(self,content):
        return word_tokenize(content)
    
    def fnormilizer(self,content):
        return self.normalizer.normalize(content)
    
    def lemmetizer_and_stemmer(self,token):
        res = self.lemmatizer.lemmatize(token)
        # res = self.stemmer.stem(res)
        return res



class Positional_index():
    def __init__(self,dataframe):
        self.indexes = {}
        self.titles = {}
        for i in range(len(dataframe)):
            self.titles[i] = dataframe.loc[i].title
    
    def add_and_merge_content(self,tokens,id):
        for token in set(tokens):
            token_indexes = [i for i,val in enumerate(tokens) if val==token]
            if(token in self.indexes.keys()):
                self.indexes[token]['count'] += len(token_indexes)
                self.indexes[token]['postings'][id] = {'count':len(token_indexes),'positions':token_indexes}
            else:
                self.indexes[token] = {'count':len(token_indexes), 'postings':{id:{'count':len(token_indexes),'positions':token_indexes}}}
    
    def find(self,token):
        if(token in self.indexes.keys()):
            return list(self.indexes[token]['postings'].keys())
        return None
    

if __name__ == '__main__':
    preprocessor = Preprocessor()

    if(os.path.isfile('positional_index.pickle')):
        file_to_read = open("positional_index.pickle", "rb")
        positional_index=pickle.load(file_to_read)
    else:
        dataset = pd.read_excel("C:\\university\\Information Retrival\\Project\\information-retrieval\\IR1_7k_news.xlsx")
        positional_index = Positional_index(dataset)
        for i in range(len(dataset)): #deploy phase
        # for i in range(2):  #test phase
            #preprocess
            normalized_content = preprocessor.fnormilizer(dataset.loc[i].content)
            tokens = preprocessor.tokenizer(normalized_content)
            tokens = preprocessor.filter_stopwords(tokens)
            for j in range(len(tokens)):
                tokens[j] = preprocessor.lemmetizer_and_stemmer(tokens[j])
            #indexing
            positional_index.add_and_merge_content(tokens,i)
        file_to_write = open("positional_index.pickle", "wb")
        pickle.dump(positional_index, file_to_write)

    while(True):
        query = input("Enter your Query: (enter exit to exit from the program)\n")
        if(query.lower()=="exit"):
            os._exit(0)
        normlized = preprocessor.fnormilizer(query)
        tokens = preprocessor.tokenizer(normlized)
        tokens = preprocessor.filter_stopwords(tokens)
        for j in range(len(tokens)):
            tokens[j] = preprocessor.lemmetizer_and_stemmer(tokens[j])
        if(len(tokens)==1):
            print(positional_index.find(tokens[0]))
        else:
            print(tokens[0])