from nltk.corpus import inaugural, PlaintextCorpusReader
from CorpusReader_TFIDF import *

# Investigating the inaugural corpus
print('\n> Investigating the inaugural corpus <\n')
print('*************** 1 ***************')
print(len(inaugural.words()))
print('*************** 2 ***************')
print(inaugural.sents())
print('*************** 3 ***************')
print(len(inaugural.sents()))
print('*************** 4 ***************')
print(inaugural.fileids())
print('*************** 5 ***************')
print(inaugural.sents(['1789-Washington.txt']))
print('*************** 6 ***************')
print(len(inaugural.sents(['1789-Washington.txt'])))

# Testing CorpusReader_TFIDF functions
myCorpus = CorpusReader_TFIDF(inaugural)

print('\n> Testing CorpusReader_TFIDF functions <\n')
print('*************** 1 ***************')
print(myCorpus.words())
print(len(myCorpus.words()))
print('*************** 2 ***************')
dict = myCorpus.tfidf('1789-Washington.txt')
print(dict)
print('*************** 3 ***************')
dict = myCorpus.tfidf('1789-Washington.txt', returnZero = True)
print(dict)
print('*************** 4 ***************')
dict = myCorpus.tfidfAll()
print(dict)
print('*************** 5 ***************')
dict = myCorpus.tfidfAll(returnZero = True)
print(dict)

# print(myCorpus.tfidf('1789-Washington.txt'))
# print("-----\n")
# q = myCorpus.tfidfAll()
# for x in q:
#    print(x, q[x])
# print("-----\n")
# print(myCorpus.cosine_sim('1789-Washington.txt', '2021-Biden.txt')
# print("-----\n")
# print(myCorpus.cosine_sim_new(['citizens', 'economic', 'growth', 'economic'],
# '2021-Biden.txt')

#  This is for testing your own corpus
#
#  create a set of text files, store them in a directory specified from 'rootDir' variable
#
#
'''
rootDir = '/myhomedirectory'   # change that to the directory where the files are
newCorpus = PlaintextCorpusReader(rootDir, '*')
tfidfCorpus = CorpusReader_TFIDF(newCorpus)
q = tfidfCorpus.tfidfAll()
for x in q:
   print(x, q[x])
print("-----\n")
'''
