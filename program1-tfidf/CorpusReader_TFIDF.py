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

        @param fileids   List of fileids to read from
        @return  Set of words.
        """
        return self.corpus.words(fileids=fileids)

    # *************** TF-IDF Specific Methods ***************
    def tfidf(self, fileid, returnZero=False):
        """! Gets the TF/IDF vector for a given fileid. 
        (default is all docs in the corpus).

        @param fileid   The fileid to read from
        @param returnZero If true, return a dictionary of words
        containing all words with TF/IDF value of zero.
        @return  TF/IDF vector dictionary.
        """

        tfidf = {}
        zeros = {}

        rawDoc = self.raw(fileid)
        docWords = self.words(fileid)

        # Get IDF
        idfs = self.get_idf()

        # Get TF-IDF
        for word in docWords:

            # Calculate TF
            tf = rawDoc.count(word)

            if self.tf == 'log' and tf != 0:
                tf = 1 + log(tf)

            # Fill either zeros or tfidf depending on setting
            tfidf[word] = tf * idfs[word]
            if returnZero == True and tfidf[word] == 0:
                zeros[word] = 0

        if returnZero:
            return zeros
        return tfidf

    def tfidfAll(self, returnZero=False):
        """! Gets the TF/IDF vector for all docs in the corpus. 
        (default is all docs in the corpus).

        @param fileid   The fileid to read from
        @param returnZero If true, tfidf value will be based on
        all words with tfidf values of zero.
        @return  TF/IDF vector dictionary. Key is the fileid
        and value is the TFIDF calculation for the document.
        """
        tfidfall = {}

        for doc in self.corpus.fileids():

            print('ONE')
            # Get tfidf for this document and merge it into tfidfall
 
            doctfidf = self.tfidf(doc, returnZero=returnZero)

            print('TWO')
            tfidfall[doc] = doctfidf
            print('THREE')

        return tfidfall

    def tfidfNew(self, words):

        tfidf = {}

        # Get IDF
        idfs = self.get_idf()

        # Get TF-IDF
        for word in words:

            # Calculate TF
            tf = words.count(word)

            if self.tf == 'log' and tf != 0:
                tf = 1 + log(tf)

            # Fill either zeros or tfidf depending on setting
            tfidf[word] = tf * idfs[word]

        return tfidf

    def get_idf(self):
        """! Return IDF of each term in the corpus as a dictionary. 
        Key is the words, value is the IDF

        @return  Dictionary of IDF values.
        """

        idfs = {}

        # Find N
        N = len(self.corpus.fileids())

        # Calculate Document Frequencies (n_i)
        for doc in self.corpus.fileids():
            appeared = {}
            # print('ONE')
            # For each word, if it did not yet appear in the doc, add one to doc frequency
            for word in self.words(doc):
                if word not in appeared:
                    # print('TWO',word)
                    idfs[word] = idfs.get(word, 0) + 1
                    appeared[word] = True

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
