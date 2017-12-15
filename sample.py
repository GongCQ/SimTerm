import gensim as gs
import pymongo as pm
import datetime as dt
import Public
import os

class Bow:
    def __init__(self, docList, termQuant, userDictSet = None):
        self.userDictSet = userDictSet if userDictSet is not None else set()
        self.termQuant = termQuant
        self.docList = docList
        self.parseList = []
        for doc in docList:
            self.parseList.append(doc.parse)
        self.wordId = gs.corpora.Dictionary(self.parseList)
        for word, id in self.wordId.token2id.items():
            self.wordId.id2token[id] = word
        corpus = [self.wordId.doc2bow(parse) for parse in self.parseList]

        tfIdf = gs.models.TfidfModel(corpus)
        for c in range(len(corpus)):
            cor = corpus[c]
            ti = tfIdf[cor]
            tiDict = {} # use for sort
            for i in range(len(ti)):
                ti[i] = [self.wordId.id2token[ti[i][0]], ti[i][1]]
                tiDict[ti[i][0]] = ti[i][1]
            tiSort = sorted(tiDict.items(), key=lambda d:d[1], reverse=True)
            self.docList[c].tiSort = tiSort
            self.docList[c].tiDict = tiDict

        self.termToId = {}
        self.idToTerm = {}
        maxTermId = 0
        for d in range(len(self.docList)):
            doc = self.docList[d]
            termSet = set()
            for ts in range(len(doc.tiSort)):
                word = doc.tiSort[ts][0]
                if (ts <= int(len(doc.tiSort) * self.termQuant) and Public.ValidWord(word)) or word in self.userDictSet:
                    termSet.add(word)
            termList = []
            for word in doc.parse:
                if word in termSet:
                    termList.append(word)
                    if word not in self.termToId.keys():
                        self.termToId[word] = maxTermId
                        self.idToTerm[maxTermId] = word
                        maxTermId += 1
            doc.termList = termList

    def GetId(self, word):
        return self.termToId[word]

    def GetWord(self, id):
        return self.idToTerm[id]

    def GetVocabSize(self):
        return len(self.termToId.keys())

    def SaveIdTerm(self, path):
        fileIdToTerm = open(os.path.join(path, 'idToTerm.txt'), 'w')
        for id, term in self.idToTerm.items():
            fileIdToTerm.write(str(id) + ',' + str(term) + os.linesep)
        fileIdToTerm.flush()
        fileIdToTerm.close()

        fileTermToId = open(os.path.join(path, 'termToId.txt'), 'w')
        for term, id in self.termToId.items():
            fileTermToId.write(str(term) + ',' + str(id) + os.linesep)
        fileTermToId.flush()
        fileTermToId.close()



class Document:
    def __init__(self, doc):
        self.content = doc['content']
        self.parse = doc['parse']
        self.time = doc['time']
        self.title = doc['title']
        self.key = doc['_id']

userDictPath = 'dict'
mongoConnStr = 'mongodb://gongcq:gcq@localhost:27017/text'
termQuant = 0.3
days = 30

mc = pm.MongoClient(mongoConnStr)
db = mc['text']
docs = db['section'].find({'time': {'$gte': dt.datetime.now() - dt.timedelta(days=days)}})
userDictSet = Public.FileToSet(userDictPath)

docList = []
for doc in docs:
    if doc['masterId'] != '':
        continue
    document = Document(doc)
    docList.append(document)
bow = Bow(docList, termQuant, userDictSet)
bow.SaveIdTerm('.')

debug = 0