import pandas as pd

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]"," ",text)
    text = re.sub(r"\s+"," ",text)
    return text.strip()

def get_dataset():
    df = pd.read_csv('SMSSpamCollection',
                sep = '\t',
                header = None,
                names = ['label','text'])
    df['label'] = df['label'].map({'ham':0,"spam":1})
    df['clean_text']   =  df['text'].apply(clean_text) 
    new_df = df[['label','clean_text']]
    return new_df




