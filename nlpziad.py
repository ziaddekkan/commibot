import string
from nltk.corpus import stopwords
import spacy
from nltk import word_tokenize
import unicodedata
nlp = spacy.load("fr_core_news_md")
frenchstopword = set(stopwords.words("french"))
def lemmatisation(Q):
  nfkd_form = unicodedata.normalize('NFKD', Q)
  Q ="".join([c for c in nfkd_form if not unicodedata.combining(c)])
  mylist = word_tokenize(Q,language="french")
  mylist = [x.lower() for x in mylist if x not in string.punctuation and x not in frenchstopword]
  text = " ".join(mylist)
  lem = nlp(text)

  lemmatized = [x.lemma_ for x in lem]
  return lemmatized