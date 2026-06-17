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
df = pd.read_csv("train.csv")

label_map = {
    0: "dovish",
    1: "hawkish",
    2: "neutral"
}

df["LABEL"] = df["label"].map(label_map)

texts = df["sentence"]
labels = df["LABEL"]
# -----------------------
# 4. Train model ONCE at startup
# -----------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

print("CLASSES:")
print(model.classes_)

print("LABEL COUNTS:")
print(df["LABEL"].value_counts())

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

@app.get("/")
def home():
    return {"status": "ok"}
