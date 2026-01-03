import joblib
import numpy as np
import json
from fastapi import FastAPI
from pydantic import BaseModel
from nlpziad import lemmatisation
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("comubot.pkl")


with open("reponse.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def reponse(text):
    intent = model.predict([text])
    des = model.decision_function([text])
    prob = des[0][np.argmax(des)]
  
    if prob > 0.05:
        return data[intent[0]]
    elif prob > -0.3:
        return "Pouvez-vous reformuler votre question ?"
    else:
        return "Votre question est hors contexte"
print(reponse("bonjour"))
# ---- API ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # <-- ici "*" pour tester en local, sinon ton domaine
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, OPTIONS...
    allow_headers=["*"],
)
class Question(BaseModel):
    text: str

@app.post("/chat")
def chat_api(question: Question):
    answer = reponse(question.text)
    return {
        "question": question.text,
        "response": answer
    }
