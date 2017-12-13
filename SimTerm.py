import tensorflow as tf
import os

class SimTerm:
    def __init__(self, vocabSize, contextLen, batchSize, wordEmbLen, hiddenLen, learnRate = 0.1):
        self.vocabSize = vocabSize
        self.contextLen = contextLen
        self.batchSize = batchSize
        self.wordEmbLen = wordEmbLen
        self.learnRate = learnRate
        self.hiddenLen = hiddenLen

        self.graph = tf.Graph()
        with self.graph.as_default():
            # input layer and object value
            self.input = tf.placeholder(dtype=tf.int32, shape=[self.batchSize, self.contextLen], name='input')
            self.objVal = tf.placeholder(dtype=tf.int32, shape=[self.batchSize], name='objVal')

            # word embedding, and concatenate
            self.embTable = tf.get_variable(name='embTable', initializer=tf.zeros([self.vocabSize, self.wordEmbLen]), dtype=tf.float32)
            self.wordEmb = tf.nn.embedding_lookup(self.embTable, self.input, name='wordEmb')
            self.wordEmbCon = tf.reshape(self.wordEmb,
                                         shape=[self.batchSize if self.batchSize is not None else -1,
                                                self.contextLen * self.wordEmbLen],
                                         name='wordEmbCon')

            # tanh hidden
            self.wHid = tf.get_variable(name='wHid',
                                        initializer=tf.truncated_normal([self.contextLen * self.wordEmbLen, self.hiddenLen], stddev=0.1),
                                        dtype=tf.float32)
            self.bHid = tf.get_variable(name='bHid', initializer=tf.zeros([self.hiddenLen]), dtype=tf.float32)
            self.hidden = tf.tanh(tf.matmul(self.wordEmbCon, self.wHid) + self.bHid, name='hidden')

            # output layer
            self.wOut = tf.get_variable(name='wOut',
                                        initializer=tf.truncated_normal([self.hiddenLen, self.vocabSize], stddev=0.1),
                                        dtype=tf.float32)
            self.bOut = tf.get_variable(name='bOut',
                                        initializer=tf.zeros([self.vocabSize]),
                                        dtype=tf.float32)
            self.output = tf.nn.softmax(tf.matmul(self.hidden, self.wOut) + self.bOut,
                                        name='output')

            # loss
            self.oneHotObjVal = tf.one_hot(self.objVal, self.vocabSize)
            loss = tf.losses.softmax_cross_entropy(self.oneHotObjVal, self.output,
                                                   weights=1, reduction=tf.losses.Reduction.NONE)
            self.cost = tf.reduce_mean(loss, axis=0)

            # train
            trainVar = tf.trainable_variables()
            grad = tf.gradients(self.cost, trainVar) # grad, norm = tf.clip_by_global_norm(tf.gradients(self.cost, trainVar), 5)
            optimizer = tf.train.GradientDescentOptimizer(learnRate)
            self.trainOper = optimizer.apply_gradients(zip(grad, trainVar))

            # initialize
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())

    def Train(self, input, objVal):
        wv = self.output.eval(session=self.sess, feed_dict={self.input:input})
        cost, oneHotObjVal, _ = self.sess.run([self.cost, self.oneHotObjVal, self.trainOper], feed_dict={self.input: input, self.objVal: objVal})
        output = self.Eval(input)
        return cost, output, oneHotObjVal

    def Test(self, input, objVal):
        cost = self.sess.run(self.cost, feed_dict={self.input: input, self.objVal: objVal})
        return cost

    def Eval(self, input):
        output = self.sess.run(self.output, feed_dict={self.input: input})
        return output

    def GetWordEmb(self, wordId):
        emb = tf.nn.embedding_lookup(self.embTable, wordId)
        embVal = emb.eval(session=self.sess)
        return embVal

    def SaveEmbTable(self, path):
        embList = self.GetWordEmb(list(range(self.vocabSize)))
        np.savetxt(path, embList, fmt='%s,', newline='\n')

import random
import datetime as dt
import numpy as np
from sample import bow
batchSize = 50
contextLen = 7
interval = 3
samples = []
for doc in bow.docList:
    for i in range(0, len(doc.termList) - contextLen, 5):
        window = doc.termList[i : i + contextLen]
        objSeq = contextLen - 1
        objWord = bow.GetId(window[objSeq])
        context = []
        for j in range(contextLen):
            if j != objSeq:
                context.append(bow.GetId(window[j]))
        samples.append((context, objWord))
        ddd = 0

shufSeq = list(range(len(samples)))
random.shuffle(shufSeq)
batchs = []
batch = ([], [])
for s in shufSeq:
    if len(batch[0]) == batchSize:
        batchs.append(batch)
        batch = ([], [])
    batch[0].append(samples[s][0])
    batch[1].append(samples[s][1])

model = SimTerm(bow.GetVocabSize(), contextLen - 1, batchSize=None, wordEmbLen=100, hiddenLen=500, learnRate=1)
for b in range(len(batchs)):
    batch = batchs[b]
    cost, output, objOneHot = model.Train(batch[0], batch[1])
    print(str(dt.datetime.now()) +
          ', batch=' + str(b) + '/' + str(len(batchs)) +
          ', cost=' + str(np.exp(cost)))
    if b % 500 == 0:
        path = os.path.join('.', 'modelFile', 'model' + str(b) + '.csv')
        model.SaveEmbTable(path)
        print('=== success to save model in ' + path)
    for i in range(batchSize):
        outputSortIndex = np.argsort(-output[i])
        objOneHotSortIndex = np.argsort(-objOneHot[i])
        # print('output: ' + str(output[i][outputSortIndex[0 : 10]]))
        # print('objVal: ' + str(objOneHot[i][objOneHotSortIndex[0 : 10]]))
        objId = objOneHotSortIndex[0]
        objIndexInOutputSort = None
        for n in range(len(outputSortIndex)):
            if objId == outputSortIndex[n]:
                objIndexInOutputSort = n
                break
        # print(str(objId) + ': ' + str(objIndexInOutputSort) + '/' + str(bow.GetVocabSize()))
        ddd = 0
    ddd = 0
