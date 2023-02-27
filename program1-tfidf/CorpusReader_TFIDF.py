"""
Michael Lennon 
CS 5324 Program 1
TF/IDF Corpus Reader
"""

from nltk.corpus import CorpusReader, stopwords
from nltk.stem.snowball import SnowballStemmer

from math import log2
import itertools

import numpy as np
from numpy.linalg import norm

# DOCUMENTATION:
#   1. PlainTextCorpusReader:           https://www.nltk.org/_modules/nltk/corpus/reader/plaintext.html
#   2. Corpus Reader (Parent of (1)):   https://www.nltk.org/api/nltk.corpus.reader.html?highlight=corpusreader#nltk.corpus.reader.CorpusReader
#   3. nltk.corpus.reader (if you care):https://www.nltk.org/_modules/nltk/corpus/reader


class CorpusReader_TFIDF:

    # ********************* Constructor *********************
    def __init__(self, corpus, tf="raw", idf="base", stopWord="none", toStem=False, ignoreCase=True):
        """ 
        @param corpus NLTK corpus object.
        @param tf term frequency method to use. Use "raw" or "log"
        @param idf idf method to use. Use "base" or "smooth"
        @param stopWord: list of stopwords to remove. Use "none, "standard", or give a filename
        @param toStem: whether to use the snowball stemmer on the words
        @param ignoreCase: whether to force lowercase on the words
        @return List of all fileids.
        """
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

    def fileids(self):
        """! Returns all fileids in the corpus.

        @return List of all fileids.
        """
        return self.corpus.fileids()

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

        snow_stemmer = SnowballStemmer(language='english')

        tfidf = {}
        zeros = {}

        rawDoc = self.raw(fileid)
        docWords = self.filterWords(self.words(fileid))

        # Get IDF
        idfs = self.get_idf()

        # Get TF-IDF
        for word in docWords:
            if self.toStem:
                word = snow_stemmer.stem(word)
            # Calculate TF
            tf = rawDoc.count(word)

            if self.tf == 'log' and tf != 0:
                tf = 1 + log2(tf)

            # Fill either zeros or tfidf depending on setting
            # Pop all zeros from tfidf if we aren't returning zeros
            tfidf[word] = tf * idfs[word]
            if returnZero == True and tfidf[word] == 0:
                zeros[word] = 0
            if returnZero == False and tfidf[word] == 0:
                tfidf.pop(word)

        # Add zeros for TF-IDF for all other words in the corpus (TF = 0)
        for word in self.words():
            if word not in tfidf:
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
        snow_stemmer = SnowballStemmer(language='english')

        tfidfall = {}

        for doc in self.fileids():

            # Get tfidf for this document and merge it into tfidfall
            doctfidf = self.tfidf(doc, returnZero=returnZero)
            tfidfall[doc] = doctfidf

        return tfidfall

    def tfidfNew(self, words):
        """! Gets the TF/IDF vector for a "new" document.

        @param words The list of words to get TF/IDF values for
        @return  TF/IDF vector dictionary.
        """

        snow_stemmer = SnowballStemmer(language='english')

        tfidf = {}

        # Get IDF
        idfs = self.get_idf()

        # Get TF-IDF
        for word in words:
            if self.toStem:
                word = snow_stemmer.stem(word)

            # Calculate TF
            tf = words.count(word)

            if self.tf == 'log' and tf != 0:
                tf = 1 + log2(tf)

            # Fill tfidf. If there's no IDF, the IDF is 1.
            # (either 1, or 1 + log2(1))
            if word not in idfs:
                # TODO: What should the IDF be if the word wasn't in the original corpus?
                idfs[word] = 1
            tfidf[word] = tf * idfs[word]

        return tfidf

    def get_idf(self):
        """! Return IDF of each term in the corpus as a dictionary. 
        Key is the words, value is the IDF

        @return  Dictionary of IDF values.
        """

        snow_stemmer = SnowballStemmer(language='english')

        idfs = {}

        # Find N
        N = len(self.fileids())

        # Calculate Document Frequencies (n_i)
        for doc in self.fileids():
            appeared = {}

            docWords = self.filterWords(self.words(doc))

            # For each word, if it did not yet appear in the doc, add one to doc frequency
            for word in docWords:
                if self.toStem:
                    word = snow_stemmer.stem(word)

                if word not in appeared:
                    idfs[word] = idfs.get(word, 0) + 1
                    appeared[word] = True

        # Calculate IDF_i for each n_i
        for word in idfs.keys():

            if self.idf == 'smooth':
                idfs[word] = log2(1 + (N / idfs[word]))
            else:
                idfs[word] = log2(N / idfs[word])

        return idfs

    def cosine_sim(self, fileid1, fileid2):
        """! Return Ithe cosine similarity between two documents
        in the corpus.

        @param fileid1  The name of the first file to compare
        @param fileid2  The name of the second file to compare
        @return  Cosine similarity between the two files.
        """

        # Get TF/IDF dictionaries and force them to be the same size and order
        tfidf1 = self.tfidf(fileid1)
        tfidf2 = self.tfidf(fileid2)

        for word in tfidf1:
            if word not in tfidf2:
                tfidf2[word] = 0
        for word in tfidf2:
            if word not in tfidf1:
                tfidf1[word] = 0

        tfidf1 = dict(sorted(tfidf1.items(), key=lambda x: x[0]))
        tfidf2 = dict(sorted(tfidf2.items(), key=lambda x: x[0]))

        # Calculate cosine similarity
        # Used numpy approach as shown in this example: https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
        tfidf_vals1 = np.array(list(tfidf1.values()))
        tfidf_vals2 = np.array(list(tfidf2.values()))

        return np.dot(tfidf_vals1, tfidf_vals2)/(norm(tfidf_vals1)*norm(tfidf_vals2))

    def cosine_sim_new(self, words, fileid):
        """! Return the cosine similarity between a new document
        and an existing document in the corpus.

        @param word  The "new" document.
        @param fileid2  The name of the existing file to compare
        @return  Cosine similarity between the new and existing file.
        """

        # Get TF/IDF dictionaries and force them to be the same size and order
        tfidf1 = self.tfidf(fileid)
        tfidf2 = self.tfidfNew(words)

        for word in tfidf1:
            if word not in tfidf2:
                tfidf2[word] = 0
        for word in tfidf2:
            if word not in tfidf1:
                tfidf1[word] = 0

        tfidf1 = dict(sorted(tfidf1.items(), key=lambda x: x[0]))
        tfidf2 = dict(sorted(tfidf2.items(), key=lambda x: x[0]))

        # Calculate cosine similarity
        tfidf_vals1 = np.array(list(tfidf1.values()))
        tfidf_vals2 = np.array(list(tfidf2.values()))

        return np.dot(tfidf_vals1, tfidf_vals2)/(norm(tfidf_vals1)*norm(tfidf_vals2))

    def query(self, words):
        """! Return the cosine similarity between a new document
        and an existing document in the corpus.

        @param word  The query.
        @return  A list of documents sorted by cosine similarity to the query.
        """
        res = []

        for doc in self.fileids():
            res.append((doc, self.cosine_sim_new(words, doc)))

        res = sorted(res, key=lambda x: x[1], reverse=True)

        return res

    # ******************* Helper Methods *******************
    def filterWords(self, words):
        """! Filters a list of words by removing stopwords and setting
        the words to lowercase depending on corpus properties.

        @param word  The list of words to filter.
        @return  The filtered words.
        """
        wordsFiltered = []

        # No filtering needs to be done
        if self.stopWord == 'none' and not self.ignoreCase:
            return words
        # Case ignored; no stopwords need to be removed
        elif self.stopWord == 'none' and self.ignoreCase:
            wordsFiltered = [w.lower() for w in words]
            return wordsFiltered
        # Stopwords need to be removed
        else:
            toRemove = []
            # Read from NLTK stopwords, or read from custom file
            # Build list of stopwords to remove
            if self.stopWord == 'standard':
                toRemove = stopwords.words('english')
            else:
                with open(self.stopWord, 'r') as data_file:
                    for line in data_file:
                        lineSplit = line.split()
                        toRemove.append(lineSplit)
                    toRemove = list(itertools.chain.from_iterable(toRemove))

            # Filter words in toRemove
            for w in words:
                if w not in toRemove:
                    if self.ignoreCase:
                        w = w.lower()
                    wordsFiltered.append(w)

        return wordsFiltered
