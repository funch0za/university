'''
Скопируйте из исходников файл reviews.csv в свой проект.

Подготовьте данные для обучения, как на занятии: 
оставьте только буквы и пробелы (без лишних пробелов), 
проведите лемматизацию слов, удалите стоп-слова.

Сохраните обработанные отзывы в файл reviews_preprocessed.csv со столбцами: 
review и label. Примечание: в столбце label должен быть 1, 
если отзыв позитивный и 0, 
если он негативный.
'''
import re
import nltk
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = None
LEMMATIZER = None


def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d", "", text)

    text = [LEMMATIZER.lemmatize(t) for t in word_tokenize(text) if t not in STOPWORDS]
    text = [t for t in text if t not in STOPWORDS]
    return " ".join(text)


def work_with_data(filename):
    data = pd.read_csv(filename)
    data["label"] = data["sentiment"].apply(lambda label: 1 if label == 'positive' else 0)
    tqdm.pandas()

    global STOPWORDS
    global LEMMATIZER

    STOPWORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()

    nltk.download('wordnet')
    nltk.download("stopwords")

    data["processed"] = data["review"].progress_apply(preprocess_text)
    data[["processed", "label"]]. to_csv(f"{filename}_preprocessed.csv", index=False, header=True)

    print(data.head(50))
    print(data.tail(50))
    print("Saved in ", f"{filename}_preprocessed.csv")


work_with_data("reviews.csv")

