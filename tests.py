from Translation import GoogleTranslation, MyMemoryTranslation
from SentimentClassifier import SentimentClassifier

def TestSentiment():
    print("Testing sentiment classification\n")

    s = SentimentClassifier()
    positive_sentence = "estoy muy feliz de haberte conocido"
    negative_sentence = "eres pésimo"
    print(positive_sentence)
    print(s.analyze_sentence(positive_sentence))
    print(negative_sentence)
    print(s.analyze_sentence(negative_sentence))

#The NER part of the assigment is hard to test due to flair's nature

def TestTranslation():
    print("\n\nTesting translation\n")
    print("testing translation with sentence \'hola, compa\'")
    spanish_sentence = "hola, compa"

    a = MyMemoryTranslation()
    print("Translation with MyMemoryTranslation")
    print(a.translate(spanish_sentence))

    b = GoogleTranslation()
    print("Translation with Google Translate")
    print(b.translate(spanish_sentence))

if __name__ == '__main__':
    TestSentiment()
    TestTranslation()
