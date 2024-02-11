import re
import os
import time
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from rank_bm25 import BM25Okapi
from collections import Counter
import itertools

# GLOBAL VARS
Pseudo_Rel_Enabled = False

stemmer = SnowballStemmer("english") # Porter2 is an improved version of Porter Stemmer
markup_pattern = re.compile(r'<[^>]+>')

nltk.download('stopwords')
with open('StopWords.txt', 'r', encoding='utf-8') as file:
    provided_stop_words = file.read().split()
stop_words = set(provided_stop_words) | set(stopwords.words('english'))

'''STEP 1: PREPROCESSING'''
documents = []
vocabulary_processed_tokens = set()
vectorizer = CountVectorizer(stop_words= list(stop_words), token_pattern=r'\b[a-z]+\b')

def preprocess():
    
    X = vectorizer.fit_transform(documents)

    # Get the vocabulary 
    vocabulary = vectorizer.get_feature_names_out()
    
    # stem words
    for word in vocabulary:
        vocabulary_processed_tokens.add(stemmer.stem(word))


def extract_documents(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through each file
    for file_name in files:
        
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            document = file.read()
            # Remove words enclosed within angle brackets
            document = re.sub(markup_pattern, '', document)
            documents.append(document)

def preprocess_runner():
    print("STEP 1: Starting pre-processing and building vocabulary...Timer Started...")
    start_time = time.time()

    # prepare docs and preprocess
    extract_documents("coll")
    preprocess()

    pre_processing_end_time = time.time()
    pre_processing_elapsed = pre_processing_end_time - start_time
    print("Pre-processing complete, time taken: ", pre_processing_elapsed, 'seconds' )

    # add vocabulary to a new file
    with open('Vocabulary.txt', 'w', encoding='utf-8') as output_file:
        for token in vocabulary_processed_tokens:
            output_file.write(token + '\n')
    
    print('Vocabulary:',len(vocabulary_processed_tokens), 'words, written to Vocabulary.txt' )

'''STEP 2: BUILDING INVERTED INDEX'''
# Global vars for inverted index
folder_path = 'coll'
invertedIndex = {}
all_docs_processed = {}
'''
A dictionary mapping document id's to a tuple containing document length and the stemmed words in a list
{"DOC_id" : (5,['a', 'doc', 'with', '5', 'words']), ...}
'''

def buildAndUpdateIndex (documents):
    '''
    vocabulary_processed_tokens: global list of stemmed words built once
    invertedIndex: global index that is updated by repeated calls to this function
    @param documents: a dictionary mapping docIDS to tuples representing all the documents in one file, with the first element of the tuple being the document length, and the 2nd is a list of stemmed words {'docID': (2, ['two', 'words']), ...}
    The fcn. goes over each word and updating the index with the no. of times it appears in each doc
    '''
    for docid, value in documents.items():
        documentid = docid
        for word in value[1]:
            if word in vocabulary_processed_tokens:
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
    tokens = wordpunct_tokenize(document.lower())  
    
    # Preprocess tokens
    for token in tokens:
        if token.isalpha():
            # Remove stopwords and stem the word
            if token not in stop_words:
                processed_words.append(stemmer.stem(token))

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

def preprocess_docs_and_build_index(folder_path):
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
        print('Processed file no. ', file_count, 'of ', len(files))


def invertedIndexRunner():
    print("STEP 2: Preprocessing and index building starting...Starting timer...")
    start_time = time.time()
    # run preprocessing and build the index
    preprocess_docs_and_build_index(folder_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print ("Inverted Index built, time taken: ", elapsed_time, "seconds")
    print ("Length of Inverted index is ", len(invertedIndex))
    print ("No. of documents processed is ", len(all_docs_processed))
    # Specify the file path
    index_path = 'invertedIndex.json'
    # Write the dictionary to the file
    with open(index_path, 'w') as file:
        json.dump(invertedIndex, file, indent=4)

    print("InvertedIndex has been written to", index_path)


'''STEP 3: RETRIEVAL AND RANKING'''

# Global vars for this section
output = []
pr_output = []
def process_query(doc_text):

    # Define regular expressions to match document ID and the rest of the information as text without tags
    markup_pattern = re.compile(r'<[^>]+>')

    # Remove words enclosed within angle brackets 
    document = re.sub(markup_pattern, '', doc_text)
    
    processed_words = []
    # Tokenization: Split the document into words
    tokens = wordpunct_tokenize(document.lower())  
    
    unnecessary = {'document', 'will', 'relevant', 'mention', 'discuss', 'note'}
    # Preprocess tokens
    for token in tokens:
        if token.isalpha():
            # Remove stopwords and stem the word
            if (token not in stop_words) and (token not in unnecessary):
                processed_words.append(stemmer.stem(token))

    # Return document ID and the rest of the information without tags
    return processed_words

def query_splitter(file):
    '''
    @param file- variable with the multiple tag delimited queries stored in a file
    Splits the queries in a file including the title, desc and narr fields in the query and calls the process_document fcn. on each 
    '''
    # first split all the documents in the file
    documents = re.findall(r'<top>(.*?)</top>', file, re.DOTALL)
    
    processed_documents = {}

    # process each document - a document being everything contained in the <DOC> tags
    counter = 1
    for document in documents:
        processed_words = process_query(document)
        processed_documents[counter] = (len(processed_words), processed_words)
        counter +=1
       
    return processed_documents

def getRelevantDocs(query):
    relevant_docs = set()
    query_unique = set(query)
    for term in query_unique:
        if invertedIndex.get(term) != None:
            currtermdocs = set(invertedIndex.get(term).keys())
            for doc in currtermdocs:
                relevant_docs.add(doc)
        else:
            continue
    
    return list(relevant_docs)

# docIDS_list is the list of relevant DocidS
def getTokenizedCorpus(docIDs_list):
    tokenized_corpus = []

    for docID in docIDs_list:
        # fetch the stemmed words for the particular document
        tokenized_corpus.append(all_docs_processed.get(docID)[1])

    return tokenized_corpus

# gets the scores and returns top 1000 or all if <1000
def get_scores(tokenized_corpus, docIDs_list,query):
    bm25 = BM25Okapi(tokenized_corpus)
    doc_scores = bm25.get_scores(query)

    results = []
    for i in range(len(doc_scores)):
        results.append((docIDs_list[i],doc_scores[i]))
    
    sorted_list = sorted(results, key=lambda x: x[1], reverse=True)

    if len(sorted_list) > 1000:
        sorted_list = sorted_list[:1000]

    return sorted_list

# Only run if the pseudo relevance flag is set to true
def getDocWordsPseudoRelevance(query_scores):
    # query scores is a list of tuples [(DOC ID, score), ...]
    words = []

    for docID,score in query_scores:
        # fetch the stemmed words for the particular document
        words.append(all_docs_processed.get(docID)[1])

    merged_list = list(itertools.chain.from_iterable(words))
    return merged_list
    

# Only run if the pseudo relevance flag is set to true
def pseudoRelevance(query, query_index, curr_query_scores, num_terms=7):
    word_list = getDocWordsPseudoRelevance(curr_query_scores[0:3])
    #print('wordlist: ', word_list)

    # Count the frequency of each term
    term_freq = Counter(word_list)
    
    # Select the top 'num_terms' terms based on frequency
    top_terms = [term for term, _ in term_freq.most_common(num_terms)]
    
    query.extend(top_terms)

    relevant_docspr = getRelevantDocs(query)
    print('PR Relevant docs', len(relevant_docspr))
    tokenized_corpuspr = getTokenizedCorpus(relevant_docspr)
    print('PR Size of tokenized corpus', len(tokenized_corpuspr))

    curr_query_scorespr = get_scores(tokenized_corpuspr,relevant_docspr,query)
    print('PR Query score length',len(curr_query_scorespr))
    pr_output.append((query_index, curr_query_scorespr))
    print('PR Query, ',query_index, ' processed: ', query)

def retrieve_and_rank(queries):
    for index,querytuple in queries.items():
        
        relevant_docs = getRelevantDocs(querytuple[1])
        print('Relevant docs retrieved', len(relevant_docs))
        tokenized_corpus = getTokenizedCorpus(relevant_docs)

        curr_query_scores = get_scores(tokenized_corpus,relevant_docs,querytuple[1])
        print('Length of scored queries: ',len(curr_query_scores))
        output.append((index, curr_query_scores))
        print('Query',index, ' processed: ', querytuple[1])

        if Pseudo_Rel_Enabled:
            pseudoRelevance(querytuple[1], index, curr_query_scores)

    return

def run_retrieval_and_ranking():
    print ('STEP 3: Starting query pre-processing, retrieval and ranking...Timer started...')
    start_time = time.time()

    # read the queries
    with open('queries.txt', 'r', encoding='utf-8') as file:
                document = file.read()

    queries = query_splitter(document)

    print("Running queries...")
    retrieve_and_rank(queries)

    print('Queries processed, writing to output file')
    output_file = 'results.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        for query in output:
            index = 1
            #print(query[0])
            for doc in query[1]:
                
                line = str(query[0]) +' '+'Q0'+ ' '+ doc[0] + ' ' + str(index) + ' ' +str(doc[1]) + ' ' +'run_1'
                file.write(line + '\n')
                index+=1 
    # Write out results of using pseudo-relevance
    if Pseudo_Rel_Enabled:
        print('Queries processed with pseudo-relevance enabled, writing to output file')
        output_pr_file = 'pr_results.txt'
        with open(output_pr_file, 'w', encoding='utf-8') as file:
            for query in pr_output:
                index = 1
                #print(query[0])
                for doc in query[1]:
                    
                    line = str(query[0]) +' '+'Q0'+ ' '+ doc[0] + ' ' + str(index) + ' ' +str(doc[1]) + ' ' +'run_2'
                    file.write(line + '\n')
                    index+=1 

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Retrieval and ranking complete, time taken: ", elapsed_time, "seconds")


def main():
    start_time = time.time()

    preprocess_runner() # pre-processing
    invertedIndexRunner() # make inverted index
    run_retrieval_and_ranking() # retrieval and ranking

    end_time = time.time()
    total_time = end_time - start_time
    print("Done! Time taken: ", total_time)


main()