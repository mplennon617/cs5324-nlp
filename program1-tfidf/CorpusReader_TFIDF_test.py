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
myCorpus = CorpusReader_TFIDF(
    inaugural, toStem=True, ignoreCase=True, stopWord='toRemove.txt')

print('\n> Testing CorpusReader_TFIDF functions <\n')
print('*************** 1 ***************')
print(myCorpus.words())
print(len(myCorpus.words()))
print('*************** 2 ***************')
dict = myCorpus.tfidf('1789-Washington.txt')
print(dict)
print('*************** 3 ***************')
dict = myCorpus.tfidf('1789-Washington.txt', returnZero=True)
print(dict)
print('*************** 4 ***************')
print('This will take some time...')
dict = myCorpus.tfidfAll()
print(dict)
print('*************** 5 ***************')
print(len(myCorpus.words('1789-Washington.txt')))
print(len(myCorpus.words('1793-Washington.txt')))
print(myCorpus.cosine_sim('1789-Washington.txt', '1793-Washington.txt'))
print(myCorpus.cosine_sim('1789-Washington.txt', '2017-Trump.txt'))
content = []

with open('1793-washington-local.txt', 'r') as data_file:
    for line in data_file:
        lineSplit = line.split()
        content.append(lineSplit)
    content = list(itertools.chain.from_iterable(content))
    print(content)

print('*************** 6 ***************')
dict = myCorpus.tfidfNew(content)
print(dict)
print('*************** 7 ***************')
print(myCorpus.cosine_sim_new(content, '1789-Washington.txt'))
print('*************** 8 ***************')
print('This will take some time...')
print(myCorpus.query(content))
