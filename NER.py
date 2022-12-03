from flair.data import Corpus
from flair.datasets.biomedical import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

class NER:

    def __init__(self, *, useAllData: bool) -> None:
        self._useAllData = useAllData
        self._data_folder = './NERData'

        self._columns = {0: 'text', 1: 'ner'}
        self._make_corpus()
        self._tag_type = 'ner'
        self._tag_dictionary = self._corpus.make_label_dictionary(label_type='ner')
        self._initialize_embeddings()
        self._initialize_sequence_tagger()
        self._initialize_trainer()
        self._run_training()
        self._plot_results()


    def _make_corpus(self):
        if self._useAllData:
            train_file = 'train'
        else:
            train_file = 'train_trunc'

        self._corpus: Corpus = ColumnCorpus(data_folder = self._data_folder, column_format = self._columns,
                                      train_file=train_file,
                                      test_file='test',
                                      dev_file='dev')


    def _initialize_embeddings(self):
        embedding_types = [

            WordEmbeddings('glove'),

            FlairEmbeddings('news-forward-fast'),

            FlairEmbeddings('news-backward-fast'),
        ]

        self._embeddings = StackedEmbeddings(embeddings=embedding_types)


    def _initialize_sequence_tagger(self):
        self._tagger = SequenceTagger(hidden_size=256,
                                embeddings=self._embeddings,
                                tag_dictionary=self._tag_dictionary,
                                tag_type=self._tag_type)


    def _initialize_trainer(self):
        self._trainer = ModelTrainer(self._tagger, self._corpus)


    def _run_training(self):
        self._trainer.train('resources/taggers/ner-english',
                      train_with_dev=True,
                      write_weights=True,
                      max_epochs=1)


    def _plot_results(self):
        plotter = Plotter()
        plotter.plot_training_curves('resources/taggers/ner-english/loss.tsv')
        plotter.plot_weights('resources/taggers/ner-english/weights.txt')
