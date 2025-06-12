import pandas as pd
import re


df = pd.read_csv('pubg_reviews_all.csv')

allowed_pattern = re.compile(r"^[a-zA-Z0-9\s.,!?'\-]+$")

def is_clean_english(text):
    return bool(allowed_pattern.fullmatch(str(text).strip()))

df = df[df['review'].apply(is_clean_english)]

df = df[df['review'].str.strip().str.len() > 3]

df.to_csv('processing_pubg_reviews.csv', index=False)