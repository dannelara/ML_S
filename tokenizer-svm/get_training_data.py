import json

def get_training_data(kind):
    if kind == "sni_full":
        with open('sni_full.json', 'r', encoding="utf-8",errors="replace") as f:
            return json.load(f)
    elif kind == "names_lemmatized":
        with open('lemmatized.json', 'r', encoding="utf-8", errors="replace") as f:
            return json.load(f)
    elif kind == "names_stemmed":
        with open('stemmed.json', 'r', encoding="utf-8", errors="replace") as f:
            return json.load(f)
    elif kind == "names_tokens":
        with open('tokens_norm.json', 'r', encoding="utf-8", errors="replace") as f:
            return json.load(f)        
    else: return False