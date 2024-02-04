import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
import os

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
porter = PorterStemmer()

markup_pattern = re.compile(r'<[^>]+>')

# Initialize set of stopwords by combining provided stopwords with the ones in the NLTK resource
file_path = 'StopWords.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    provided_stop_words = file.read().split()
    
stop_words = set(provided_stop_words)| set(stopwords.words('english'))

processed_tokens = set()

def preprocess(document):
    
    # Remove words enclosed within angle brackets 
    document = re.sub(markup_pattern, '', document)
    
    # Tokenization: Split the document into words
    tokens = word_tokenize(document.lower())  
    
    # Preprocess tokens
    for token in tokens:
        if token.isalpha():
            # Remove stopwords and stem the word
            if token not in stop_words:
                stemmed_token = porter.stem(token)
                processed_tokens.add(stemmed_token)
                

def preprocess_documents(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through each file
    for file_name in files:
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file_name)
        
        # Read the contents of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            document = file.read()
        # Preprocess the document
        preprocess(document)

# set folder path and preprocess documents
folder_path = 'coll'
preprocess_documents(folder_path)

# Write tokens to new file
output_file = 'Vocabulary.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for token in processed_tokens:
        file.write(token + '\n')
