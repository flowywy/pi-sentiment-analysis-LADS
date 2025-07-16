# --- Load stopwords 
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
import pandas as pd
import re
import string
import wordninja
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import os


stop_words_en = set(stopwords.words('english'))
custom_stopwords = {
    'im', 'ive', 'youve', 'theyve', 'hes', 'shes', 'its', 'theres', 'whats', 'thats', 'were', 'youre', 'you re', 'they re',
    'make', 'made', 'makes', 'argghhh', 'arghhh', 'ugh', 'omg', 'omgg', 'yayyy', 'loo b tupi',
    'hmmm', 'ummm', 'uhhh', 'emmm', 'yaaa', 'ehhh', 'awww', 'huh',
    'grrr', 'yaaaay', 'booo', 'aaah', 'meh', 'eh', 'owww', 'whoa', 'woah',
    'dawg', 'just', 'really', 'actually', 'literally', 'basically',
    'kinda', 'sorta'
}
stop_words_en.update(custom_stopwords)

# --- Load slang dictionary ---
def load_slang_dict():
    slang_dict = {}
    slang_path = os.path.join(os.path.dirname(__file__), "slang.txt")
    with open(slang_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.replace("'", "").split(":")
                if len(parts) == 2:
                    slang_dict[parts[0].strip()] = parts[1].strip()
    return slang_dict

# --- Clean emojis & special chars ---
def clean_data_ulasan(text):
    text = re.sub(r'(<3|:\)|:-\)|:\(|:-\(|:D|XD|xD)', '', text, flags=re.IGNORECASE)  # hapus emoji teks
    text = re.sub(r'[^\w\s]', '', text)        # hapus tanda baca
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # hapus karakter non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()   # rapikan spasi
    return text

# --- Case folding ---
def case_folding(text):
    return text.lower()

# --- Remove slang ---
def remove_slang(text, slang_dict):
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

# --- Normalisasi (huruf berulang & split) ---
keywords_with_numbers = {'f2p', 'p2w', '5star', '4star', '10pull', '10x', '2d', '3d'}

def reduce_elongated_words(word):
    return re.sub(r'(.)\1{2,}', r'\1\1', word)

def prep_text(text):
    if pd.isnull(text):
        return ""

    raw_words = text.split()
    cleaned_words = []

    for word in raw_words:
        if word in keywords_with_numbers:
            cleaned_words.append(word)
        else:
            word_clean = ''.join(char for char in word if char not in string.punctuation and not char.isdigit())
            word_clean = reduce_elongated_words(word_clean)
            cleaned_words.append(word_clean)

    final_words = []
    for word in cleaned_words:
        if word in keywords_with_numbers:
            final_words.append(word)
        else:
            split_words = wordninja.split(word)
            final_words.extend(split_words)

    return ' '.join(final_words).strip()

# --- Remove stopwords ---
def remove_stopwords(text):
    if pd.isnull(text):
        return ""
    return ' '.join([word for word in text.split() if word not in stop_words_en])

# --- Tokenisasi ---
def tokenize_text(text):
    return word_tokenize(text)

# --- Lemmatization ---
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(tokens):
    lemmatized_text = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    return ' '.join(lemmatized_text)

# --- Pipeline lengkap untuk DataFrame ---
def preprocess_dataframe(df, slang_dict):
    df = df.copy()
    df['clean_content'] = df['content'].apply(clean_data_ulasan)
    df['case_folding'] = df['clean_content'].apply(case_folding)
    df['slang_removed'] = df['case_folding'].apply(lambda x: remove_slang(x, slang_dict))
    df['normalized'] = df['slang_removed'].apply(prep_text)
    df['stopword'] = df['normalized'].apply(remove_stopwords)
    df['tokenized'] = df['stopword'].apply(tokenize_text)
    df['preprocess'] = df['tokenized'].apply(lemmatize)
    return df