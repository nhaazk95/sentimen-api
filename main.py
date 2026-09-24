import json, re, joblib, nltk
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

model = joblib.load('svm_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
with open('slang_dict.json', encoding='utf-8') as f:
    slang_dict = json.load(f)

stop_words = set(nltk_stopwords.words('indonesian'))
kata_penting = {'tidak','belum','sangat','kurang','terlalu','sudah','paling','lama','sekali'}
stop_words -= kata_penting
stemmer = StemmerFactory().create_stemmer()

ACCURACY_TRAINING = 0.9778
PRECISION_TRAINING = 0.9780
RECALL_TRAINING = 0.9778
F1_MACRO_TRAINING = 0.9779
CV_F1_MACRO = 0.8314

def full_preprocess(text):
    text = "" if not text else str(text)
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(slang_dict.get(w, w) for w in text.split())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

app = FastAPI()

class ReviewInput(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentimen: str
    accuracy_training: float
    precision_training: float
    recall_training: float
    f1_macro_training: float
    cv_f1_macro: float

@app.post("/predict", response_model=PredictResponse)
def predict(data: ReviewInput):
    clean = full_preprocess(data.text)
    X = tfidf.transform([clean])
    pred_int = model.predict(X)[0]
    label = label_encoder.inverse_transform([pred_int])[0]
    return {
        "sentimen": label,
        "accuracy_training": ACCURACY_TRAINING,
        "precision_training": PRECISION_TRAINING,
        "recall_training": RECALL_TRAINING,
        "f1_macro_training": F1_MACRO_TRAINING,
        "cv_f1_macro": CV_F1_MACRO
    }
