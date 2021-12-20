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
        self.stopwords_set.update(['.',':','?','!','/','//','*','**','***','[',']','{','}',';','\'','\"','(',')','','،'])
        
        
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
        result = {}
        if(token in self.indexes.keys()):
            for key in self.indexes[token]['postings'].keys():
                result[key] = self.indexes[token]['postings'][key]['positions']
            return result
        return None
    
def main():
    preprocessor = Preprocessor()
    dataset = pd.read_excel("C:\\university\\Information Retrival\\Project\\information-retrieval\\IR1_7k_news.xlsx")
    if(os.path.isfile('positional_index.pickle')):
        file_to_read = open("positional_index.pickle", "rb")
        positional_index=pickle.load(file_to_read)
    else:
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
            ndict= positional_index.find(tokens[0])
            if(ndict == None):
                print("No result found")
                continue
            most_related_id = list(ndict.keys())[0]
            print("Id of most related document is ",most_related_id," and its title is :\n")
            print(positional_index.titles[most_related_id])
            sentences = sent_tokenize(dataset.loc[most_related_id].content)
            print("related sentences:\n")
            for sen in sentences:
                if(tokens[0] in sen):
                    print(sen)
        else:
            docIDs = []
            docIDs_l = []
            priorities = []
            for token in tokens:
                docIDs.append(positional_index.find(token))
            if None in docIDs:
                print("No result found")
                continue
            for dic in docIDs:
                docIDs_l.append(list(dic.keys()))
            while(len(docIDs_l)>0):
                priorities.append(set.intersection(*map(set,docIDs_l)))
                docIDs_l.pop()
            for i in range(1,len(priorities)):
                priorities[i] = priorities[i].difference(priorities[i-1])
            final_list = []
            
            for myset in priorities:
                temp_list=[]
                for id in list(myset):
                    score = 0
                    for j in range(len(docIDs)-1):
                        if(id in docIDs[j].keys() and id in docIDs[j+1].keys()):
                            for position in docIDs[j][id]:
                                if(position+1 in docIDs[j+1][id]):
                                    score += 1
                    temp_list.append((id,score))
                temp_list=sorted(temp_list, key=lambda x: x[1],reverse=True)
                final_list += temp_list
            
            most_related_id = final_list[0][0]
            print("Id of most related document is ",most_related_id," and its title is :\n")
            print(positional_index.titles[most_related_id])
            sentences = sent_tokenize(dataset.loc[most_related_id].content)
            print("related sentences:\n")
            for sen in sentences:
                if(tokens[0] in sen):
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
    # main()
    # check_zipfs_law()
    check_heaps_law()            

