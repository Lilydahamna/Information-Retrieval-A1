import nltk
from nltk.corpus import stopwords
import string
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer


# Porter2 is an improved version of Porter Stemmer
stemmer = SnowballStemmer("english")

markup_pattern = re.compile(r'<[^>]+>')

# Initialize set of stopwords by combining provided stopwords with the ones in the NLTK resource
nltk.download('stopwords')
with open('StopWords.txt', 'r', encoding='utf-8') as file:
    provided_stop_words = file.read().split()
stop = set(provided_stop_words) | set(stopwords.words('english'))

documents = []
processed_tokens = set()
vectorizer = CountVectorizer(stop_words= list(stop), token_pattern=r'\b[a-z]+\b')

def preprocess():
    
    X = vectorizer.fit_transform(documents)

    # Get the vocabulary 
    vocabulary = vectorizer.get_feature_names_out()
    
    # stem words
    for word in vocabulary:
     processed_tokens.add(stemmer.stem(word))


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
            
       

# prepare documents and preprocess 
extract_documents("coll")
preprocess()

# add vocabulary to a new file
with open('Vocabulary.txt', 'w', encoding='utf-8') as output_file:
    for token in processed_tokens:
        output_file.write(token + '\n')

print(len(processed_tokens))