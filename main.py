from __future__ import unicode_literals
from hazm import *
import pandas as pd
import pickle
import os
import math



class Preprocessor():

    def __init__(self):
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer() 
        self.stopwords_set = set(stopwords_list())
        self.stopwords_set.update(['.',':','?','!','/','//','*','**','***','[',']','{','}',';','\'','\"','(',')','','،',''])
        
        
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
        self.squaroftfidfsize = [0 for a in range(len(dataframe))]
        for i in range(len(dataframe)):
            self.titles[i] = dataframe.loc[i].title
    
    def add_and_merge_content(self,tokens,id):
        for token in set(tokens):
            token_indexes = [i for i,val in enumerate(tokens) if val==token]
            if(token in self.indexes.keys()):
                self.indexes[token]['count'] += len(token_indexes)
                self.indexes[token]['postings'][id] = {'tf':len(token_indexes),'positions':token_indexes}
            else:
                self.indexes[token] = {'count':len(token_indexes), 'postings':{id:{'tf':len(token_indexes),'positions':token_indexes}}}
    
    def find(self,token):
        result = {}
        if(token in self.indexes.keys()):
            for key in self.indexes[token]['postings'].keys():
                result[key] = self.indexes[token]['postings'][key]['positions']
            return result
        return None

def make_championlist(positional_index):
    for token in positional_index.indexes.keys():
        championlist = []
        postings = positional_index.indexes[token]['postings']
        for docid in postings.keys():
            championlist.append((docid,postings[docid]['tfidf']))
        championlist= sorted(championlist, key=lambda x: x[1],reverse=True)
        positional_index.indexes[token]['championlist'] = championlist[:40]   #40 best documents are in champion list
        positional_index.indexes[token]['lowlist'] = championlist[40:]


def calculate_idf(positional_index,N):
    for key in positional_index.indexes.keys():
        positional_index.indexes[key]['idf'] = math.log10(N/(len(positional_index.indexes[key]['postings'].keys()))) 
        for docid in positional_index.indexes[key]['postings'].keys():
            tfidf = positional_index.indexes[key]['idf']* (1+math.log10(positional_index.indexes[key]['postings'][docid]['tf']))
            positional_index.indexes[key]['postings'][docid]['tfidf'] = tfidf
            positional_index.squaroftfidfsize[docid] += tfidf**2

def make_positional_index(dataset):
    preprocessor = Preprocessor()
    positional_index = Positional_index(dataset)
    for i in range(len(dataset)): #deploy phase
        #preprocess
        normalized_content = preprocessor.fnormilizer(dataset.loc[i].content)
        tokens = preprocessor.tokenizer(normalized_content)
        tokens = preprocessor.filter_stopwords(tokens)
        for j in range(len(tokens)):
            tokens[j] = preprocessor.lemmetizer_and_stemmer(tokens[j])
        #indexing
        positional_index.add_and_merge_content(tokens,i)
    calculate_idf(positional_index,len(dataset))
    make_championlist(positional_index)
    file_to_write = open("positional_index.pickle", "wb")
    pickle.dump(positional_index, file_to_write)
    return positional_index


    
def main():
    dataset = pd.read_excel("C:\\university\\Information Retrival\\Project\\information-retrieval\\IR1_7k_news.xlsx")
    if(os.path.isfile('positional_index.pickle')):
        file_to_read = open("positional_index.pickle", "rb")
        positional_index=pickle.load(file_to_read)
    else:
        positional_index= make_positional_index(dataset)

    preprocessor = Preprocessor()
    while(True):
        query = input("Enter your Query: (enter exit to exit from the program)\n")
        if(query.lower()=="exit"):
            os._exit(0)
        normlized = preprocessor.fnormilizer(query)
        qtokens = preprocessor.tokenizer(normlized)
        qtokens = preprocessor.filter_stopwords(qtokens)
        for j in range(len(qtokens)):
            qtokens[j] = preprocessor.lemmetizer_and_stemmer(qtokens[j]) 
        terms = {}
        related_docs = {} #docid: dot product
        for term in qtokens:
            if(term in terms.keys()):
                terms[term]+= 1
            else:
                terms[term] = 1
        vectorsize_query = 0
        
        for term in terms.keys():
            if(term in positional_index.indexes.keys()):
                idf= positional_index.indexes[term]['idf']
                query_term_tfidf = (1+math.log10(terms[term]))*idf
                vectorsize_query += query_term_tfidf**2
                for doc in positional_index.indexes[term]['championlist']:
                    docid = doc[0]
                    doc_tfidf = doc[1]
                    if(docid in related_docs.keys()):
                        related_docs[docid] += query_term_tfidf*doc_tfidf
                    else:
                        related_docs[docid] = query_term_tfidf*doc_tfidf

        if(len(related_docs.keys())<5): #if didnt find k best in championlist k=5
            for term in terms.keys():
                if(term in positional_index.indexes.keys()):
                    idf= positional_index.indexes[term]['idf']
                    query_term_tfidf = (1+math.log10(terms[term]))*idf
                    vectorsize_query += query_term_tfidf**2
                    for doc in positional_index.indexes[term]['lowlist']:
                        docid = doc[0]
                        doc_tfidf = doc[1]
                        if(docid in related_docs.keys()):
                            related_docs[docid] += query_term_tfidf*doc_tfidf
                        else:
                            related_docs[docid] = query_term_tfidf*doc_tfidf


        if(len(related_docs.keys())==0):
            print("no result found")
        else:
            vectorsize_query = vectorsize_query**0.5
            for docid in related_docs.keys():
                related_docs[docid] /= positional_index.squaroftfidfsize[docid] **0.5
                related_docs[docid] /= vectorsize_query
            
            related_docs_tuple = [(k, v) for k, v in related_docs.items()]
            related_docs_tuple=sorted(related_docs_tuple, key=lambda x: x[1],reverse=True)
            print("5 Best document ID with their similarity score:")
            print(related_docs_tuple[:5])
            print("title of most related doc:")
            print(positional_index.titles[related_docs_tuple[0][0]])
            sentences = sent_tokenize(dataset.loc[related_docs_tuple[0][0]].content)
            print("related sentences:\n")
            for sen in sentences:
                if(qtokens[0] in sen):
                    print(sen)

def check_zipfs_law():
    preprocessor = Preprocessor()
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
    frequency = []
    for key in positional_index.indexes.keys():
        frequency.append((key,positional_index.indexes[key]['count']))
    print(sorted(frequency, key=lambda x: x[1],reverse=True)[:20])

def check_heaps_law():
    preprocessor = Preprocessor()
    dataset = pd.read_excel("C:\\university\\Information Retrival\\Project\\information-retrieval\\IR1_7k_news.xlsx")
    positional_index = Positional_index(dataset)
    count = 1000
    for i in range(count): #deploy phase
    # for i in range(2):  #test phase
        #preprocess
        normalized_content = preprocessor.fnormilizer(dataset.loc[i].content)
        tokens = preprocessor.tokenizer(normalized_content)
        tokens = preprocessor.filter_stopwords(tokens)
        for j in range(len(tokens)):
            tokens[j] = preprocessor.lemmetizer_and_stemmer(tokens[j])
        #indexing
        positional_index.add_and_merge_content(tokens,i)
    sum =0
    for key in positional_index.indexes.keys():
        sum+= positional_index.indexes[key]['count']
    print("sum for tokens for ",count," document is: ",sum," and length of vocabulary(M) is : ", len(list(positional_index.indexes.keys())))

if __name__ == '__main__':
    main()        

