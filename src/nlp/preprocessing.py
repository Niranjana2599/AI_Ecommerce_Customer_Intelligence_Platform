import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================================================
# DOWNLOAD NLTK DATA
# =========================================================

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# =========================================================
# OBJECTS
# =========================================================

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'[^a-zA-Z ]', '', text)

    text = text.split()

    text = [
        lemmatizer.lemmatize(word)
        for word in text
        if word not in stop_words
    ]

    text = ' '.join(text)

    return text