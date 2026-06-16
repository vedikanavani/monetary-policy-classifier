import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------
# 1. Create FastAPI app
# -----------------------
app = FastAPI()

# -----------------------
# 2. Enable CORS (VERY IMPORTANT for frontend)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# 3. Load dataset
# -----------------------
df = pd.read_csv("bank_statement_data.csv")

texts = df["TEXT"]
labels = df["LABEL"]

# -----------------------
# 4. Train model ONCE at startup
# -----------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

# -----------------------
# 5. Define request format
# -----------------------
class Request(BaseModel):
    text: str

# -----------------------
# 6. Prediction endpoint
# -----------------------
@app.post("/predict")
def predict(req: Request):
    X_new = vectorizer.transform([req.text])
    pred = model.predict(X_new)
    return {"prediction": pred[0]}