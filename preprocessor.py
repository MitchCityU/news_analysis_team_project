import pandas as pd
import re
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df = pd.read_csv("True.csv")

df['processed_title'] = df['title'].astype(str).apply(preprocess_text)
df['processed_text'] = df['text'].astype(str).apply(preprocess_text)

df.to_csv("preprocessed_True.csv", index=False)

print(df[['processed_title', 'processed_text']].head())
