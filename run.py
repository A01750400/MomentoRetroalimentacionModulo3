from SentimentClassifier import SentimentClassifier
from NER import NER
from Translation import TranslatorComparer

#Exercise 1
print("------Excercise 1------")
ex1 = SentimentClassifier()
ex1.classify_file("tiny_movie_reviews_dataset.txt")

#Exercise 2
print("------Excercise 2------")
ex2 = NER(useAllData=False)

#Exercise 3
print("------Excercise 3------")
ex3 = TranslatorComparer('europarl-v7.es-en.es', 'europarl-v7.es-en.en')
