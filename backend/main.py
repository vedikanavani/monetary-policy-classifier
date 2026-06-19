import pandas as pd
from fastapi import FastAPI ##FastAPI is a class in the fasapi library. 
from fastapi.middleware.cors import CORSMiddleware #this goes into fastapi, then one layer deeper into middleware, then one layer 
#deeper into cors and then imports the class CORSMiddleware. 
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


df = pd.read_csv("train.csv")

label_map = {
    0: "dovish",
    1: "hawkish",
    2: "neutral"
}

df["LABEL"] = df["label"].map(label_map)

texts = df["sentence"]
labels = df["LABEL"]

vectorizer = TfidfVectorizer()
#TfidfVectorizer 
#a classic vectoriser takes sentences from the dataset, creates a vocabulary, and then for each word in the vocabulary assigns an index. 
#it then creates, for each sentence, a vector that is as long as the total no of words in the vocabulary. if a word is in the sentence,
#that spot in the vector recieves a 1, otherwise 0. 
#TfidfVectorizer doesn't put 1 or 0 in the vector, it puts values ranging between 1 and 0 on the basis of how common those words are, 
#because the less common a word the more indicative it is. (eg. in this case committee would not be very telling and would get low weight, 
#but the word "tightening would be informative for hawkish sentences, so would get a higher weight.")
X = vectorizer.fit_transform(texts)
#vectoriser is the vectoriser we've created. the fit_transform function belongs to the TfidfVectorizer class. fit learns the 
#vocabulary of the texts and transform turns it into numbers. X is the final set of all sentence vectors. 

model = LogisticRegression()
#creates an empty logistic regression model. 
model.fit(X, labels)
#model here is the empty logisticregression model we've created. model.fit means going into this empty model and accessing fit. so 
#accessing the fit function that exists in this logistic regresssion object, because that's what the class has from scikit-learn. 
#"fit" is the function that takes care of the entire learning process (adjusting weights).


class Request(BaseModel):
    text: str
#new class called "request" is defined. Basemodel is from the libaray Pydantic. This is basically saying that the class request should
#inherit the properties of BaseModel. Basemodel is a class itself, and is used for checking data types, validating inputs, etc. 
#text:str means that a valid type of "reqeust" here should be of type string. Basemodel is what verifies that. 


@app.post("/predict")
#@ is a decorator, which means it tells FastAPI something about the function below it. 
#accesses "post" which belongs to app, which is of the FastAPI class, so post exists in FastAPI. So in some sense post "connects" FastAPI to 
# to the function predict, which is defined below.  
def predict(req: Request): # the input of the predict function must be of class request. 
    X_new = vectorizer.transform([req.text]) # req.text is the text from the input of class Request that this function takes. the [] turns this
    #into a list. this is because the vectoriser receieved lots of sentences during training. so here, transform takes this new sentence and 
    #turns it into numbers based on the original vocabulary. the vector is stored in the variable X_new. 
    pred = model.predict(X_new) #here, the trained model that was trained after model.fit calls the predict function that exists in the logistic 
    #regression class and runs it on X_new. 
    return {"prediction": pred[0]} #here, the final prediction is the first element of pred because a logistic regression is designed to handle 
    #several different "sentences" or blocks of text at once and return different output for all of them. since we are only running the funciton 
    #on one block of text, accessing the first element in the pred list returns the output for this whole chunk of text. 



@app.get("/") #get belongs to app=FastAPI object. "/" as a string refers to a route or path. so this says when someone sends a get request to /, the
#function below must be run. 
def home():
    return {"status": "ok"}
#funtion just returns status:ok. this is just to check when the server is running if FastAPI is wowrking and requests are reaching. 

#done!! :D

