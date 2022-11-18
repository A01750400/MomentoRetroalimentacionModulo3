from translate import Translator
from googletrans import Translator as GoogleTrans
from nltk.translate.bleu_score import sentence_bleu

class GoogleTranslation:

    def __init__(self) -> None:
        self._translator = GoogleTrans()

    def translate(self, spanish_text: str):
        return self._translator.translate(spanish_text, src='es', dest='en').text


class MyMemoryTranslation:

    def __init__(self) -> None:
        self._translator = Translator(to_lang='en', from_lang='es')

    def translate(self, spanish_text: str):
        return self._translator.translate(spanish_text)


class TranslatorComparer:

    def __init__(self, es_filename: str, en_filename: str) -> None:
        self._google = GoogleTranslation()
        self._mymemory = MyMemoryTranslation()
        self._es = self.__read_file(es_filename)
        self._en = self.__read_file(en_filename)
        self._scores = {'Google': [], 'MyMemory': []}
        self.run_comparisons()

    def __read_file(self, filename: str):
        with open(filename) as file:
            return [line.rstrip() for line in file][:100]

    def line_valid(self, line_num: int):
        return len(self._es[line_num]) < 500

    def __compare(self, line_num: int):
        if not self.line_valid(line_num):
            return False
        ref = self._en[line_num].split()
        input_txt = self._es[line_num]
        translation = self._google.translate(input_txt).split()
        self._scores['Google'].append(sentence_bleu(ref, translation))
        print(translation, sentence_bleu(ref, translation))
        translation = self._mymemory.translate(input_txt)
        self._scores['MyMemory'].append(sentence_bleu(ref, translation))
        print(translation, sentence_bleu(ref, translation))
        return True

    def run_comparisons(self):
        google_sum = 0
        mymemory_sum = 0
        lines = len(self._es)
        for i in range(lines):
            if self.__compare(i):
                google_sum += self._scores['Google'][-1]
                mymemory_sum += self._scores['MyMemory'][-1]

        print("Google Translator: ",'{0:.3g}'.format((google_sum / len(self._scores['Google']))))
        print("MyMemory Translator: ",'{0:.3g}'.format((mymemory_sum / len(self._scores['MyMemory']))))

