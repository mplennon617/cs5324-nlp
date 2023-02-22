class CorpusReader_TFIDF:

    def __init__(self, corpus, tf="raw", idf="base", stopWord="none", toStem=false, ignoreCase=true):
        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        self.stopWord = stopWord
        self.toStem = toStem
        self.ignoreCase = ignoreCase

    # Shared Methods
    def fields():
        return 0

    def raw():
        return 0

    def raw(fileids=[]):
        return 0

    def words():
        return 0

    def words(fileids=[]):
        return 0

    # TF-IDF Specific Methods
    def tfidf(filieid, returnZero=False):
        return 0

    def tfidfAll(filieid, returnZero=False):
        return 0

    def tfidfNew(words):
        return 0

    def idf():
        return 0

    def cosine_sim(fileid1, fileid2):
        return 0

    def cosine_sim_new(words, fileid):
        return 0

    def query(words):
        return 0
