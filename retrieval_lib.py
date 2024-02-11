from rank_bm25 import BM25Okapi
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import time
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import SnowballStemmer


# corpus = [
#     "Hello there good man!",
#     "It is quite windy in London",
#     "How is the weather today?"
# ]

# tokenized_corpus = [doc.split(" ") for doc in corpus]

# print(tokenized_corpus)

# bm25 = BM25Okapi(tokenized_corpus)

# query = "windy London"
# tokenized_query = query.split(" ")

# doc_scores = bm25.get_scores(tokenized_query)

# print(doc_scores)

'''
Approach using library:
    Stem query words and tokenize
    Find the relevant documents for each word using the inverted index
    Get the list of words for these docs from the all_docs_processed (this is the tokenized_corpus)
    Initialize the bm25 model using this
    get doc scores
'''

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
# Initialize Porter Stemmer
stemmer = SnowballStemmer("english")

# Initialize set of stopwords by combining provided stopwords with the ones in the NLTK resource
file_path = 'StopWords.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    provided_stop_words = file.read().split()
    
stop_words = set(provided_stop_words)| set(stopwords.words('english'))

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
    @param file- variable with the multiple tag delimited documents stored in a file
    Splits the documents in s file by the <DOC> tag and calls the process_document fcn. on each doc
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

# print(len(relevant_docs))

# docIDs_list = list(relevant_docs)

# docIDS_list is the list of relevant DocidS
def getTokenizedCorpus(docIDs_list):
    tokenized_corpus = []

    for docID in docIDs_list:
        # fetch the stemmed words for the particular document
        tokenized_corpus.append(all_docs_processed.get(docID)[1])

    return tokenized_corpus

# print (len(tokenized_corpus))
# print(docIDs_list[0])
# print(tokenized_corpus[0])

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

def runner(queries):
    for index,querytuple in queries.items():
        
        relevant_docs = getRelevantDocs(querytuple[1])
        print('Relevant docs', len(relevant_docs))
        tokenized_corpus = getTokenizedCorpus(relevant_docs)
        print('Size of tokenized corpus', len(tokenized_corpus))

        curr_query_scores = get_scores(tokenized_corpus,relevant_docs,querytuple[1])
        print('Query score length',len(curr_query_scores))
        output.append((index, curr_query_scores))
        print('Query, ',index, ' processed: ', querytuple[1])

    return

print ('Starting query pre-processing...Timer started...')
start_time = time.time()

with open('queries.txt', 'r', encoding='utf-8') as file:
            document = file.read()
queries = query_splitter(document)
first_query = query_splitter(document)[1][1]

# run preprocessing and build the index
#preprocess_and_build_index(folder_path)

print("Loading json files...")
with open('index.json') as json_file:
    invertedIndex = json.load(json_file)

with open('all_docs_processed.json') as json_file:
    all_docs_processed = json.load(json_file)

# print(queries)

#print(first_query)

output = []

print("Running queries...")

# print(queries)
runner(queries)


print('Queries proceesed, writing to output file')

output_file = 'results.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for query in output:
        index = 1
        #print(query[0])
        for doc in query[1]:
            
            line = str(query[0]) +' '+'Q0'+ ' '+ doc[0] + ' ' + str(index) + ' ' +str(doc[1]) + ' ' +'run_2'
            file.write(line + '\n')
            index+=1 

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")
     


     
     


