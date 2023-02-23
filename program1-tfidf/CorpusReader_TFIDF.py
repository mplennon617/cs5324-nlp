from nltk.corpus import CorpusReader
from math import log

# DOCUMENTATION:
#   1. PlainTextCorpusReader:           https://www.nltk.org/_modules/nltk/corpus/reader/plaintext.html
#   2. Corpus Reader (Parent of (1)):   https://www.nltk.org/api/nltk.corpus.reader.html?highlight=corpusreader#nltk.corpus.reader.CorpusReader
#   3. nltk.corpus.reader (if you care):https://www.nltk.org/_modules/nltk/corpus/reader


class CorpusReader_TFIDF:

    # ********************* Constructor *********************
    def __init__(self, corpus, tf="raw", idf="base", stopWord="none", toStem=False, ignoreCase=True):
        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        self.stopWord = stopWord
        self.toStem = toStem
        self.ignoreCase = ignoreCase

    # ******************** Shared Methods ********************
    # TODO: fields() only exists in ToolboxCorpusReader,
    # not PlainTextCorpusReader. Do we need to extend functionality for it?

    # def fields(self, fileids=None):
    #     return self.corpus.fields(fileids=fileids)

    def raw(self, fileids=None):
        """! Returns raw data based on given fileids 
        (default is all docs in the corpus).

        @param fields   List of fileids to read from
        @return  Raw corpus data.
        """
        return self.corpus.raw(fileids=fileids)

    def words(self, fileids=None):
        """! Returns set of words from given fileids 
        (default is all docs in the corpus).

        @param fields   List of fileids to read from
        @return  Set of words.
        """
        return self.corpus.words(fileids=fileids)

    # *************** TF-IDF Specific Methods ***************
    def tfidf(self, fileid, returnZero=False):
        return 0

    def tfidfAll(self, fileid, returnZero=False):
        return 0

    def tfidfNew(self, words):
        return 0

    def idf(self):
        """! Return IDF of each term in the corpus as a dictionary. 
        Key is the words, value is the IDF

        @return  Dictionary of IDF values.
        """

        idfs = {}
        print('We\'re in!')

        # Find N
        N = len(self.corpus.fileids())

        # Calculate Document Frequencies (n_i)
        for doc in self.corpus.fileids():
            for word in self.words('doc'):
                idfs[word] = idfs.get(word, default=0) + 1

        # Calculate IDF_i for each n_i
        for word in idfs.keys():
            idfs[word] = log(N / idfs[word])

        return idfs

    def cosine_sim(self, fileid1, fileid2):
        return 0

    def cosine_sim_new(self, words, fileid):
        return 0

    def query(self, words):
        return 0
