from pysentimiento import create_analyzer

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
        self._analyzer = create_analyzer(task="sentiment", lang="en")
        self._lines = []
        self._analyses = []
        self._read_txt(filename)
        self._analyze_sentences()
        self._output_analyses()

    def _read_txt(self, filename: str):
        '''
        Reads a text file and separates the content by lines.
        '''
        with open(filename) as file:
            self.__lines = [line.rstrip() for line in file]

    def _analyze_sentence(self, sentence: str):
        '''
        Calls the analyzer's predict function with the line of text received.
        '''
        return self._analyzer.predict(sentence)

    def _analyze_sentences(self):
        '''
        Classifies each sentence from the text file.
        '''
        for sentence in self.__lines:
            analisis = self._analyze_sentence(sentence)
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
            self._analyses.append(res)

    def _output_analyses(self):
        '''
        Outputs the results.
        '''
        for analisis in self._analyses:
            print(analisis)

