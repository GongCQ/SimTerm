import sample
import gensim as gs

termListList = []
for doc in sample.bow.docList:
    termListList.append(doc.termList)

wvModel = gs.models.Word2Vec(termListList, size=100, window=10, min_count=5)

def Sim(word, model):
    print('=== query word: ' + word)
    if word not in model.wv.vocab.keys():
        print('None')
        return
    vec = model[word]
    simList = model.wv.most_similar_cosmul(positive=[word])
    for i in range(min(20, len(simList))):
        print(simList[i])


debug = 0