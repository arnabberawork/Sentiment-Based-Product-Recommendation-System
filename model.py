# Importing Libraries
import numpy as np
import pandas as pd
import re, nltk, spacy
import pickle as pk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Force loading of wordnet corpus to avoid lazy loading issues
wordnet.ensure_loaded()

# Initialize lemmatizer AFTER corpus is loaded
lemmatizer = WordNetLemmatizer()

# Load spaCy model disabling unnecessary components
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Load pre-trained models and vectorizers
count_vector = pk.load(open('pickle_files/count_vector.pkl', 'rb'))
tfidf_transformer = pk.load(open('pickle_files/tfidf_transformer.pkl', 'rb'))
model = pk.load(open('pickle_files/model.pkl', 'rb'))
recommend_matrix = pk.load(open('pickle_files/user_final_rating.pkl', 'rb'))

# Read product data
product_df = pd.read_csv('data/sample30.csv', sep=",")

# Preprocessing functions
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    return re.sub(pattern, '', text)

def remove_punctuation_and_splchars(words):
    cleaned = []
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        if word:
            word = remove_special_characters(word, True)
            cleaned.append(word)
    return cleaned

def stem_words(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize(words):
    # Safe lemmatization with ensured corpus loaded
    return [lemmatizer.lemmatize(word, pos='v') for word in words]

def normalize(words):
    # Use cached stopwords list to avoid repeated calls
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words]
    words = remove_punctuation_and_splchars(words)
    words = [word for word in words if word not in stop_words]
    return words

def normalize_and_lemmaize(input_text):
    # Clean and lemmatize input text
    input_text = remove_special_characters(input_text)
    words = word_tokenize(input_text)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)

def model_predict(text_series):
    """
    Predict sentiment for a pandas Series or list of texts.
    Convert input to list if needed.
    """
    if not isinstance(text_series, (list, pd.Series)):
        text_series = [text_series]
    word_vector = count_vector.transform(text_series)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    return model.predict(tfidf_vector)

def recommend_products(user_name):
    # Reload recommend matrix fresh (optional)
    recommend_matrix = pk.load(open('pickle_files/user_final_rating.pkl', 'rb'))
    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[:20])
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())]
    output_df = product_frame[['name', 'reviews_text']].copy()
    output_df['lemmatized_text'] = output_df['reviews_text'].map(normalize_and_lemmaize)
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    return output_df

def top5_products(df):
    total_product = df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name', 'predicted_sentiment']).agg('count').reset_index()
    merge_df = pd.merge(rec_df, total_product['reviews_text'], on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x'] / merge_df['reviews_text_y']) * 100
    merge_df = merge_df.sort_values(ascending=False, by='%percentage')
    return pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] == 1][:5])
