from pysentimiento import create_analyzer


# shouldn't be double underscores for private methods: https://towardsdatascience.com/whats-the-meaning-of-single-and-double-underscores-in-python-3d27d57d6bd1
class SentimentClassifier:
    '''
    A sentiment classifier implemented using the Pysentimiento
    sentiment classifier model.
    '''

    def __init__(self, filename: str):
        '''
        Creates an instance of a sentiment classifier,
        receives a filename from which it will source the
        sentences to be classified.
        '''
        self.__analyzer = create_analyzer(task="sentiment", lang="en")
        self.__lines = []
        self.__analyses = []
        self.__read_txt(filename)
        self.__analyze_sentences()
        self.__output_analyses()

    def __read_txt(self, filename: str):
        '''
        Reads a text file and separates the content by lines.
        '''
        with open(filename) as file:
            self.__lines = [line.rstrip() for line in file]

    def __analyze_sentence(self, sentence: str):
        '''
        Calls the analyzer's predict function with the line of text received.
        '''
        return self.__analyzer.predict(sentence)

    def __analyze_sentences(self):
        '''
        Classifies each sentence from the text file.
        '''
        for sentence in self.__lines:
            analisis = self.__analyze_sentence(sentence)
            result = analisis.output
            if result == 'POS':
                res = 'POSITIVE'
            elif result == 'NEG':
                res = 'NEGATIVE'
            else:
                if analisis.probas['POS'] > analisis.probas['NEG']:
                    res = 'POSITIVE'
                else:
                    res = 'NEGATIVE'
            self.__analyses.append(res)

    def __output_analyses(self):
        '''
        Outputs the results.
        '''
        for analisis in self.__analyses:
            print(analisis)

