import codecs
import csv
import json
import os
import spacy_udpipe
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, SyllableTokenizer
from nltk.stem import PorterStemmer
from langdetect import detect_langs
import ssl
import nltk
from nltk.tokenize import word_tokenize

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download('punkt')
# spacy_udpipe.download("sv")
#
nlp = spacy_udpipe.load("sv")
STOPWORDS_EN = set(stopwords.words("english"))
STOPWORDS_SE = set(stopwords.words("swedish"))
STOPWORDS_CACHE = {}

def stem_words(text):
    stemmer = PorterStemmer()
    text = text.replace(".", "")
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words

def get_stopwords(lang):
    if lang not in STOPWORDS_CACHE:
        if lang == "en":
            STOPWORDS_CACHE[lang] = STOPWORDS_EN
        elif lang == "sv":
            STOPWORDS_CACHE[lang] = STOPWORDS_SE
        else:
            STOPWORDS_CACHE[lang] = set()
    return STOPWORDS_CACHE[lang]

def tokenize_string(text):
    text = text.replace(".", "")
    return word_tokenize(text)

def tokenize_lemetize(text):
    text = text.replace(".", "")
    return [token.lemma_ for token in nlp(text)]

def remove_stop_words(words, lang):
    stopwords = get_stopwords(lang)
    return [word for word in words if word.lower() not in stopwords]

def process_text(text, algo):
    words
    if algo === "lem":
        words = tokenize_lemetize(text)
    else:
        words = tokenize_lemetize(text)

    language = detect_langs(text)[0].lang

    words = stem_words(text)
    syllable_tokenizer = SyllableTokenizer(sonority_hierarchy=["aouåeiyäö","lmnvrwj","szfh","xbcdqgkpt"])
    syllable_list = syllable_tokenizer.tokenize_sents(words)
    result = [syllable.lower() for s in syllable_list for syllable in s]
    return remove_stop_words(result, language)



def process_data_file(filename):
    names= []
    sni_full_list = []
    with open(filename, "r", encoding="utf-8-sig") as csvfile:
        try:
            reader = csv.DictReader(csvfile, delimiter=";")
            for row in reader:
                try:
                    company_name = row["company_name"]
                    sni_full = row["sni_full"]
                    sni_full_list.append(sni_full)
                    names.append(company_name)
                except KeyError as e:
                    print(f"Missing key {e} in row: {row}")
        except csv.Error as e:
            print(f"CSV error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    processed_data = [process_text(name) for name in names]

    if os.path.isfile('stemmed.json'):
        with codecs.open('stemmed.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data.extend(processed_data)
            f.seek(0)
            json.dump(file_data, f, ensure_ascii=False)
    else:
        with codecs.open('stemmed.json', 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False)

# process_data_file("company_names.csv")