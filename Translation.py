from translate import Translator
from googletrans import Translator as GoogleTrans
from nltk.translate.bleu_score import sentence_bleu

class GoogleTranslation:
    '''
    A wrapper for the Google Translate library object
    '''

    def __init__(self) -> None:
        '''
        Initializes the Google Translate object
        '''
        self._translator = GoogleTrans()

    def translate(self, spanish_text: str):
        '''
        Wrapper to call the translation API
        '''
        return self._translator.translate(spanish_text, src='es', dest='en').text


class MyMemoryTranslation:
    '''
    Wrapper for the MyMemory Translate library object
    '''

    def __init__(self) -> None:
        '''
        Initializes the MyMemory Translate object
        '''
        self._translator = Translator(to_lang='en', from_lang='es')

    def translate(self, spanish_text: str):
        '''
        Wrapper to call the translation API
        '''
        return self._translator.translate(spanish_text)


class TranslatorComparer:
    '''
    Helper object to handle the translation comparisons
    '''

    def __init__(self, es_filename: str, en_filename: str) -> None:
        '''
        Initializes the object's attributes and starts the comparions using the files received
        as parameters
        '''
        self._google = GoogleTranslation()
        self._mymemory = MyMemoryTranslation()
        self._es = self._read_file(es_filename)
        self._en = self._read_file(en_filename)
        self._scores = {'Google': [], 'MyMemory': []}
        self._run_comparisons()


    def _read_file(self, filename: str):
        '''
        Read the text file and store the first 100 lines in a list
        '''
        with open(filename) as file:
            return [line.rstrip() for line in file][:100]

    def _line_valid(self, line_num: int):
        '''
        Validates if the line is no longer than 500 characters
        (required by the MyMemory API)
        '''
        return len(self._es[line_num]) < 500

    def _compare(self, line_num: int):
        '''
        Translates a line of text with both translators and compares the results
        '''
        if not self._line_valid(line_num):
            return False
        ref = self._en[line_num].split()
        input_txt = self._es[line_num]
        translation = self._google.translate(input_txt).split()
        self._scores['Google'].append(sentence_bleu(ref, translation))
        #print(translation, sentence_bleu(ref, translation))
        translation = self._mymemory.translate(input_txt)
        self._scores['MyMemory'].append(sentence_bleu(ref, translation))
        #print(translation, sentence_bleu(ref, translation))
        return True

    def _run_comparisons(self):
        '''
        Iterates through the lines of text and compares the translations, keeping track of the results
        to give a final assessment of each translatot's performance
        '''
        google_sum = 0
        mymemory_sum = 0
        lines = len(self._es)
        for i in range(lines):
            if self._compare(i):
                google_sum += self._scores['Google'][-1]
                mymemory_sum += self._scores['MyMemory'][-1]

        print("Google Translator: ",'{0:.3g}'.format((google_sum / len(self._scores['Google']))))
        print("MyMemory Translator: ",'{0:.3g}'.format((mymemory_sum / len(self._scores['MyMemory']))))

