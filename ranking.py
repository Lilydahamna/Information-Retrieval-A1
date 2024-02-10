from functools import total_ordering
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import re
import time
import json
import math

# test data
# vocabulary = ['apples', 'bananas', 'pineapples']
# document = ['apples', 'bananas', 'oranges', 'apples', 'apples', 'bananas']
# invertedIndex = {'apples': {'ID': 3}, 'bananas': {'ID': 2}}
# doc2 = ['bananas', 'pineapples', 'pineapples']

# vocabulary is global and built once
# invertedIndex is also global and keeps being updated
def buildAndUpdateIndex (documents):
    '''
    vocabulary: global list of stemmed words built once
    invertedIndex: global index that is updated by repeated calls to this function
    @param documents: a list of tuples representing all the documents in one file, with the first element of the
            tuple being the document id, and the 2nd is a list of stemmed words [('ID',['another', 'word']), ()]
    The fcn. goes over each word and updating the index with the no. of times it appears in each doc
    '''
    for docid, value in documents.items():
        documentid = docid
        for word in value[1]:
            if word in vocabulary:
                if word in invertedIndex:
                    currWordTfMap = invertedIndex.get(word)
                    if documentid in currWordTfMap:
                        currWordTfMap[documentid] +=1
                        invertedIndex[word] = currWordTfMap
                    else:
                        currWordTfMap[documentid] = 1
                        invertedIndex[word] = currWordTfMap
                else:
                    invertedIndex[word] = {documentid: 1}
    

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
# Initialize Porter Stemmer
porter = PorterStemmer()

# Initialize set of stopwords by combining provided stopwords with the ones in the NLTK resource
file_path = 'StopWords.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    provided_stop_words = file.read().split()
    
stop_words = set(provided_stop_words)| set(stopwords.words('english'))

def process_document(doc_text):
    '''
    @param doc_text: represents all the text contained between <DOC></DOC> tags

    process_document takes an individual doc, gets the doc_id and does the preprocessing on the rest of the document to return a tuple: (doc_id, [list of stemmed words in the doc])
    '''
    # Define regular expressions to match document ID and the rest of the information as text without tags
    doc_id_pattern = re.compile(r'<DOCNO>(.*?)</DOCNO>')
    markup_pattern = re.compile(r'<[^>]+>')

    # Extract document ID
    doc_id = re.search(doc_id_pattern, doc_text).group(1).strip()

    # Remove words enclosed within angle brackets 
    document = re.sub(markup_pattern, '', doc_text)
    
    processed_words = []
    # Tokenization: Split the document into words
    tokens = word_tokenize(document.lower())  
    
    # Preprocess tokens
    for token in tokens:
        if token.isalpha():
            # Remove stopwords and stem the word
            if token not in stop_words:
                stemmed_token = porter.stem(token)
                processed_words.append(stemmed_token)


    # Return document ID and the rest of the information without tags
    return doc_id, processed_words

def splitter(file):
    '''
    @param file- variable with the multiple tag delimited documents stored in a file
    Splits the documents in s file by the <DOC> tag and calls the process_document fcn. on each doc
    '''
    # first split all the documents in the file
    documents = re.findall(r'<DOC>(.*?)</DOC>', file, re.DOTALL)
    processed_documents = {}

    # process each document - a document being everything contained in the <DOC> tags
    for document in documents:
        doc_id, processed_words = process_document(document)
        processed_documents[doc_id] = (len(processed_words), processed_words)

    return processed_documents

# Mostly taken from preprocess.py with some changes
def preprocess_and_build_index(folder_path):
    '''
    @param folder_path: path to the folder with all files

    This basically just reads each file and calls the splitter and buildAndUpdate fcns on each file.
    '''
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through each file
    file_count = 0
    for file_name in files:
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file_name)
        
        # Read the contents of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            documents = file.read()
        # Preprocess the documents in this file
        curr_file_processed_docs= splitter(documents)
        # build index for processed file
        buildAndUpdateIndex(curr_file_processed_docs)
        all_docs_processed.update(curr_file_processed_docs)
        
        file_count +=1
        print('Processed file no. ', file_count, '... of ', len(files))

def process_query(doc_text):
    '''
    @param doc_text: represents all the text contained between <DOC></DOC> tags

    process_document takes an individual doc, gets the doc_id and does the preprocessing on the rest of the document to return a tuple: (doc_id, [list of stemmed words in the doc])
    '''
    # Define regular expressions to match document ID and the rest of the information as text without tags
    #doc_id_pattern = re.compile(r'<top>(.*?)</top>')
    markup_pattern = re.compile(r'<[^>]+>')

    # Extract document ID
    #doc_id = re.search(doc_id_pattern, doc_text).group(1).strip()

    # Remove words enclosed within angle brackets 
    document = re.sub(markup_pattern, '', doc_text)
    
    processed_words = []
    # Tokenization: Split the document into words
    tokens = word_tokenize(document.lower())  
    
    # Preprocess tokens
    for token in tokens:
        if token.isalpha():
            # Remove stopwords and stem the word
            if token not in stop_words:
                stemmed_token = porter.stem(token)
                processed_words.append(stemmed_token)


    # Return document ID and the rest of the information without tags
    return processed_words

def query_splitter(file):
    '''
    @param file- variable with the multiple tag delimited documents stored in a file
    Splits the documents in s file by the <DOC> tag and calls the process_document fcn. on each doc
    '''
    # first split all the documents in the file
    documents = re.findall(r'<top>(.*?)<desc>', file, re.DOTALL)
    processed_documents = {}

    # process each document - a document being everything contained in the <DOC> tags
    counter = 1
    for document in documents:
        processed_words = process_query(document)
        processed_documents[counter] = (len(processed_words), processed_words)
        counter +=1
       
    return processed_documents

def compute_idf(documents, t):
    df = 0
    if t in data:
        df = len(data[t])
    
    idf = math.log(1 + (len(documents) - df + 0.5) / (df+ 0.5))
    return idf

def score (document,query):
    bm25 = 0.0
    total_length = 0 
    for doc in document_corpus:
        total_length += data1[doc][0]
    average_length = total_length/len(document_corpus)

    for term in query:
        tf = 0 
        if term in data:
            if document in data[term]:
                tf = data[term][document]
     
        numerator = compute_idf(document_corpus, term) * tf * (1.5+1)
        denominator= tf+ 1.5 * (1-0.75+0.75 * (data1[document][0]/average_length))
        bm25 += numerator/denominator
    return bm25
    

# Global vars
# set folder path and initialize invertedIndex
folder_path = 'coll'
invertedIndex = {}

all_docs_processed = {}
'''
A dictionary mapping document id's to a tuple containing document length and the stemmed words in a list
{"DOC_id" : (5,['a', 'doc', 'with', '5', 'words']), ...}
'''

# Opening Vocabulary.txt and making a set out of the words
with open('Vocabulary.txt', 'r') as file:
    # Create an empty set to store the words
    vocabulary = set()
    
    # Iterate through each line in the file
    for line in file:
        # Strip leading and trailing whitespace, and add the word to the set
        vocabulary.add(line.strip())

print("Preprocessing and index building starting...Starting timer...")
start_time = time.time()

with open('query.txt', 'r', encoding='utf-8') as file:
            document = file.read()
queries = query_splitter(document)
query = query_splitter(document)[1][1]

# run preprocessing and build the index
#preprocess_and_build_index(folder_path)
with open('index.json') as json_file:
    data = json.load(json_file)

with open('documents.json') as json_file:
    data1 = json.load(json_file)

print(queries)

print(query)



#limited_documents = []
#for q in query:
#    limited_documents.append(data[q])


#result = {}
#for dict in limited_documents:
#    result.update(dict)

#dictionaryKeys = result.keys()
#document_corpus = [key for key in dictionaryKeys]

#testing = []
#for elem in document_corpus:
#    common = any(check in data1[elem][1] for check in query)
#    if common:
#        testing.append((elem,common))
#    else:
#        testing.append((elem,"False"))
    
    

#print(data["associ"]["AP880709-0030"])
#print(len(document_corpus))

#final = []
#counter = 0
#for doc in document_corpus:
#    final.append((doc,score(doc,query)))
    

#sorted_by_second = sorted(final, key=lambda tup: tup[1], reverse=True)

#if len(sorted_by_second) > 1000:
#    final_sorted = sorted_by_second[:1000]

#
output=[]



for index, query_dict in queries.items():
    limited_documents = []
    query = query_dict[1]
    for term in query:
        if term in data:
            limited_documents.append(data[term])
    result = {}
    for dict in limited_documents:
        result.update(dict)
    dictionaryKeys = result.keys()
    document_corpus = [key for key in dictionaryKeys]
    final = []
    print(index)
    print(query)
    #print(len(document_corpus))
    for doc in document_corpus:
        final.append((doc,score(doc,query)))
    
    sorted_by_second = sorted(final, key=lambda tup: tup[1], reverse=True)
    if len(sorted_by_second) > 1000:
        sorted_by_second = sorted_by_second[:1000]
    
    output.append((index,sorted_by_second))

    
output_file = 'results.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for query in output:
        index = 1
        #print(query[0])
        for doc in query[1]:
            
            line = str(query[0]) +' '+'Q0'+ ' '+ doc[0] + ' ' + str(index) + ' ' +str(doc[1]) + ' ' +'run_1'
            file.write(line + '\n')
            index+=1 
        


   


print("Inverted Index built")
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

print ("Length of Inverted index is ", len(invertedIndex))
print ("Length of all documents processed is ", len(all_docs_processed))
# Specify the file path
#index_path = 'index.json'
# Write the dictionary to the file
#with open(index_path, 'w') as file:
#    json.dump(invertedIndex, file, indent=4)


#print("Dictionary has been written to", index_path)

index_path = 'final2.json'
#Write the dictionary to the file
with open(index_path, 'w') as file:
    json.dump(output, file, indent=4)
    

print("Dictionary has been written to", index_path)
