from flair.data import Corpus
from flair.datasets.biomedical import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings

class NER:

    def __init__(self, *, useAllData: bool) -> None:
        data_folder = './NERData'

        columns = {0: 'text', 1: 'ner'}
        # 1. get the corpus
        if useAllData:
            train_file = 'train'
        else:
            train_file = 'train_trunc'

        corpus: Corpus = ColumnCorpus(data_folder = data_folder, column_format = columns,
                                      train_file=train_file,
                                      test_file='test',
                                      dev_file='dev')

        # 2. what tag do we want to predict?
        tag_type = 'ner'

        # 3. make the tag dictionary from the corpus
        tag_dictionary = corpus.make_label_dictionary(label_type='ner')

        # 4. initialize each embedding we use
        embedding_types = [

            # GloVe embeddings
            WordEmbeddings('glove'),

            # contextual string embeddings, forward
            FlairEmbeddings('news-forward-fast'),

            # contextual string embeddings, backward
            FlairEmbeddings('news-backward-fast'),
        ]

        # embedding stack consists of Flair and GloVe embeddings
        embeddings = StackedEmbeddings(embeddings=embedding_types)

        # 5. initialize sequence tagger
        from flair.models import SequenceTagger

        tagger = SequenceTagger(hidden_size=256,
                                embeddings=embeddings,
                                tag_dictionary=tag_dictionary,
                                tag_type=tag_type)

        # 6. initialize trainer
        from flair.trainers import ModelTrainer

        trainer = ModelTrainer(tagger, corpus)

        # 7. run training
        trainer.train('resources/taggers/ner-english',
                      train_with_dev=True,
                      write_weights=True,
                      max_epochs=1)

        from flair.visual.training_curves import Plotter

        plotter = Plotter()
        plotter.plot_training_curves('resources/taggers/ner-english/loss.tsv')
        plotter.plot_weights('resources/taggers/ner-english/weights.txt')
