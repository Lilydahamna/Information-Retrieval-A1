import nltk
from nltk.corpus import stopwords
import string
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize


# Porter2 is an improved version of Porter Stemmer
stemmer = SnowballStemmer("english")

markup_pattern = re.compile(r'<[^>]+>')

# Initialize set of stopwords by combining provided stopwords with the ones in the NLTK resource
nltk.download('stopwords')
with open('StopWords.txt', 'r', encoding='utf-8') as file:
    provided_stop_words = file.read().split()
stop_words = set(provided_stop_words) | set(stopwords.words('english'))

documents = []

def stemming_tokenizer(text):
    result = []
    # tokenizing
    tokens = wordpunct_tokenize(text.lower())
    for token in tokens:
        if token.isalpha():
            # Remove stopwords and stem the word
            if token not in stop_words:
                stemmed_token = stemmer.stem(token)
                result.append(stemmed_token)
    return result


vectorizer = CountVectorizer(tokenizer=stemming_tokenizer)


def preprocess():
    
    global processed_tokens
    
    X = vectorizer.fit_transform(documents)

    # Get the vocabulary 
    vocabulary = vectorizer.get_feature_names_out()

    print(X.shape)
    print(len(vocabulary))

    processed_tokens = set(vocabulary)
    



def extract_documents(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through each file
    size = 0
    for file_name in files:
        
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            # Split file into a collection of documents 
            collection = re.findall(r'<DOC>(.*?)</DOC>', file.read(), re.DOTALL)
            
            for document in collection:
                # Remove words enclosed within angle brackets
                document = re.sub(markup_pattern, '', document)
                documents.append(document)

       

# prepare documents and preprocess 
extract_documents("coll")
preprocess()

# add vocabulary to a new file
with open('Vocabulary.txt', 'w', encoding='utf-8') as output_file:
    for token in processed_tokens:
        output_file.write(token + '\n')

print(len(processed_tokens))

