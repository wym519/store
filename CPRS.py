import csv
import random
import numpy as np
from nltk.tokenize import word_tokenize
MAX_TITLE_LENGTH=30
MAX_NEWS=50
MAX_BODY_LENGTH=200
npratio=4
with open('train.tsv') as f:
    trainuser=f.readlines()
with open('test.tsv') as f:
    testuser=f.readlines()
with open('docs.tsv') as f:
    data=f.readlines()

news = {}
for i in data:
    line = i.split('\t')
    body_tokens = word_tokenize(line[4].lower())
    news[line[0]] = [line[1], line[2], word_tokenize(line[3].lower()), body_tokens, max(len(body_tokens), 1)]


def newsample(negatives, ratio):
    if ratio > len(negatives):
        return random.sample(negatives * (ratio // len(negatives) + 1), ratio)
    else:
        return random.sample(negatives, ratio)

train_pn = []
train_p = []
train_label = []
train_label_speed = []
train_user_his = []
train_user_his_satis = []
for i in trainuser:

    line = i.replace('\n', '').split('\t')

    clickids = [newsindex[x] for x in line[3].split()][-MAX_NEWS:]
    clicknewslen = [news[x][4] for x in line[3].split()][-MAX_NEWS:]
    dwelltime = [int(x) for x in line[4].split()][-MAX_NEWS:]
    readspeed = [clicknewslen[_] / (dwelltime[_] + 1) for _ in range(len(dwelltime))]
    avgreadspeed = np.mean(readspeed)
    readspeednorm = [min(max(np.log2(avgreadspeed / x) + 6, 0), 13) for x in readspeed]

    pdoc = [newsindex[x] for x in line[7].split()]
    ndoc = [newsindex[x] for x in line[8].split()]
    dwelltimewordvec = [int(x) for x in line[11].split()]
    wordvecnewslen = [news[x][4] for x in line[7].split()]
    readspeedwordvec = [wordvecnewslen[_] / (dwelltimewordvec[_] + 1) for _ in range(len(dwelltimewordvec))]
    readspeedwordvecnorm = [min(max(np.log2(avgreadspeed / x) + 6, 0), 13) / 13. for x in readspeedwordvec]

    for pdocindex in range(len(pdoc)):
        pdocid = pdoc[pdocindex]
        npdocindex = newsample(ndoc, npratio)
        npdocindex.append(pdocid)
        label = [0] * npratio + [1]
        train_pn.append(npdocindex)
        train_p.append(pdocid)
        train_label.append(label)
        train_user_his.append(clickids + [0] * (MAX_NEWS - len(clickids)))
        train_user_his_satis.append(readspeednorm + [0] * (MAX_NEWS - len(clickids)))
        train_label_speed.append(readspeedwordvecnorm[pdocindex])

test_pn = []
test_label = []
test_user_his_satis = []
test_impression_index = []
for i in testuser:
    line = i.replace('\n', '').split('\t')

    clickids = [newsindex[x] for x in line[3].split()][-MAX_NEWS:]
    dwelltime = [int(x) for x in line[4].split()][-MAX_NEWS:]
    clicknewslen = [news[x][4] for x in line[3].split()][-MAX_NEWS:]
    readspeed = [clicknewslen[_] / (dwelltime[_] + 1) for _ in range(len(dwelltime))]
    avgreadspeed = np.mean(readspeed)
    readspeednorm = [min(max(np.log2(avgreadspeed / x) + 6, 0), 13) for x in readspeed]

    pdoc = [newsindex[x] for x in line[7].split()][:MAX_NEWS]
    ndoc = [newsindex[x] for x in line[8].split()][:300]
    impression_start_end = []
    impression_start_end.append(len(test_pn))

    for mp in pdoc:
        test_pn.append(mp)
        test_label.append(1)
        test_user_his.append(clickids + [0] * (MAX_NEWS - len(clickids)))
        test_user_his_satis.append(readspeednorm + [0] * (MAX_NEWS - len(clickids)))

    for mp in ndoc:
        test_pn.append(mp)
        test_label.append(0)
        test_user_his.append(clickids + [0] * (MAX_NEWS - len(clickids)))
        test_user_his_satis.append(readspeednorm + [0] * (MAX_NEWS - len(clickids)))
    impression_start_end.append(len(test_pn))
    test_impression_index.append(impression_start_end)

word_dict_raw = {'PADDING': [0, 999999]}

for i in news:
    for j in news[i][2]:
        if j in word_dict_raw:
            word_dict_raw[j][1] += 1
        else:
            word_dict_raw[j] = [len(word_dict_raw), 1]

for i in news:
    for j in news[i][3]:
        if j in word_dict_raw:
            word_dict_raw[j][1] += 1
        else:
            word_dict_raw[j] = [len(word_dict_raw), 1]

word_dict = {}
for i in word_dict_raw:
    if word_dict_raw[i][1] >= 3:
        word_dict[i] = [len(word_dict), word_dict_raw[i][1]]
print(len(word_dict), len(word_dict_raw))

embdict = {}
import pickle

with open('/data/wuch/glove.840B.300d.txt', 'rb')as f:
    while True:
        line = f.readline()
        if len(j) == 0:
            break
        line = line.split()
        word = line[0].decode()
        if len(word) != 0:
            vec = [float(x) for x in line[1:]]
            if word in word_dict:
                embdict[word] = vec

from numpy.linalg import cholesky

emb_table = [0] * len(word_dict)
wordvec = []
for i in embdict.keys():
    emb_table[word_dict[i][0]] = np.array(embdict[i], dtype='float32')
    wordvec.append(emb_table[word_dict[i][0]])
wordvec = np.array(wordvec, dtype='float32')

mu = np.mean(wordvec, axis=0)
Sigma = np.cov(wordvec.T)

norm = np.random.multivariate_normal(mu, Sigma, 1)

for i in range(len(emb_table)):
    if type(emb_table[i]) == int:
        emb_table[i] = np.reshape(norm, 300)

emb_table[0] = np.zeros(300, dtype='float32')
emb_table = np.array(emb_table, dtype='float32')

news_words = [[0] * MAX_TITLE_LENGTH]

for i in news:
    line = []
    for word in news[i][2]:
        if word in word_dict:
            line.append(word_dict[j][0])
    line = line[:MAX_TITLE_LENGTH]
    news_words.append(line + [0] * (MAX_TITLE_LENGTH - len(line)))

news_body = [[0] * MAX_BODY_LENGTH]

for i in news:
    line = []
    for word in news[i][3]:

        if word in word_dict:
            line.append(word_dict[j][0])
    line = line[:MAX_BODY_LENGTH]

    news_body.append(line + [0] * (MAX_BODY_LENGTH - len(line)))

news_words = np.array(news_words, dtype='int32')
news_body = np.array(news_body, dtype='int32')

train_p =np.array(train_p ,dtype='int32')
train_pn=np.array(train_pn,dtype='int32')
train_label=np.array(train_label,dtype='int32')
train_user_his=np.array(train_user_his,dtype='int32')
train_user_his_satis=np.array(train_user_his_satis,dtype='int32')
train_label_speed=np.array(train_label_speed,dtype='float32')

test_pn=np.array(test_pn,dtype='int32')
test_label=np.array(test_label,dtype='int32')
test_user_his=np.array(test_user_his, dtype='int32')
test_user_his_satis=np.array(test_user_his_satis,dtype='int32')

def generate_batch_data(batch_size):
    idx = np.arange(len(label))
    np.random.shuffle(idx)
    y=label
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            item_title = news_words[train_pn[i]]
            item_title_split=[item_title[:,k,:] for k in range(item_title.shape[1])]

            user_title=news_words[user_his[i]]
            user_satis=train_user_his_satis[i]
            item_body = news_body[train_pn[i]]
            item_body_split=[item_body[:,k,:] for k in range(item_body.shape[1])]

            user_body=news_body[user_his[i]]
            item_pos_title = news_words[train_p[i]]
            item_pos_body = news_body[train_p[i]]
            yield (item_title_split+[user_title]+item_body_split+[user_body,user_satis,item_pos_title,item_pos_body], [train_label[i],train_label_speed[i]])

def generate_batch_data_test(batch_size):
    idx = np.arange(len(test_label))
    y=test_label
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            item_title = news_words[test_pn[i]]
            user_title=news_words[test_user_his[i]]
            user_satis=test_user_his_satis[i]
            item_body = news_body[test_pn[i]]
            user_body=news_body[test_user_his[i]]
            yield ([item_title,user_title,item_body,user_body,user_satis], [test_label[i]])

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score
from keras.optimizers import *
import keras
from keras.layers import *


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

results=[]
for repeat in range(5):
    keras.backend.clear_session()


    title_input = Input(shape=(MAX_TITLE_LENGTH,), dtype='int32')

    body_input= Input(shape=(MAX_BODY_LENGTH,), dtype='int32')

    embedding_layer = Embedding(len(word_dict), 300, weights=[emb_table],trainable=True)

    embedded_sequences = embedding_layer(title_input)
    drop_emb=Dropout(0.2)(embedded_sequences)

    selfatt = Attention(16,16)([drop_emb,drop_emb,drop_emb])
    drop_selfatt=Dropout(0.2)(selfatt)

    attention = Dense(200,activation='tanh')(drop_selfatt)
    attention = Flatten()(Dense(1)(attention))
    attention_weight = Activation('softmax')(attention)
    title_rep=keras.layers.Dot((1, 1))([drop_selfatt, attention_weight])

    title_encoder = Model([title_input], title_rep)

    embedded_sequences2 = embedding_layer(body_input)
    drop_emb2=Dropout(0.2)(embedded_sequences2)

    selfatt2 = Attention(16,16)([drop_emb2,drop_emb2,drop_emb2])
    drop_selfatt2=Dropout(0.2)(selfatt2)

    attention2 = Dense(200,activation='tanh')(drop_selfatt2)
    attention2 = Flatten()(Dense(1)(attention2))
    attention_weight2 = Activation('softmax')(attention2)
    body_rep=keras.layers.Dot((1, 1))([drop_selfatt2, attention_weight2])

    bodyEncodert = Model([body_input], body_rep)

    his_title_input =  Input((MAX_NEWS, MAX_SENT_LENGTH,), dtype='int32')
    his_body_input = Input((MAX_NEWS,200,), dtype='int32')

    titlebeh=TimeDistributed(title_encoder)(his_title_input)
    bodybeh=TimeDistributed(bodyEncodert)(his_body_input)

    attention_title = Dense(200,activation='tanh')(titlebeh)
    attention_title = Flatten()(Dense(1)(attention_title))
    attention_weight_title = Activation('softmax')(attention_title)
    userrep_title=keras.layers.Dot((1, 1))([titlebeh, attention_weight_title])

    attention_body = Dense(200,activation='tanh')(bodybeh)
    attention_body = Flatten()(Dense(1)(attention_body))
    attention_weight_body = Activation('softmax')(attention_body)
    userrep_body=keras.layers.Dot((1, 1))([bodybeh, attention_weight_body])

    time_input = Input(shape=(MAX_NEWS,), dtype='int32')
    timeembedding_layer = Embedding(100, 50,  trainable=True)(time_input)
    attention_satis = Lambda(lambda x:K.sum(x,axis=-1))(multiply([bodybeh, Dense(256)(timeembedding_layer) ]))
    attention_weight_satis = Activation('softmax')(attention_satis)
    userrep_satis=keras.layers.Dot((1, 1))([bodybeh, attention_weight_satis])
    userrep_read=add([userrep_body,userrep_satis])

    uservecs =concatenate([Lambda(lambda x: K.expand_dims(x,axis=1))(vec) for vec in [userrep_title,userrep_read]],axis=1)
    attentionvecs= Dense(200,activation='tanh')(uservecs)
    attentionvecs = Flatten()(Dense(1)(attentionvecs))
    attention_weightvecs = Activation('softmax')(attentionvecs)
    userrep=keras.layers.Dot((1, 1))([uservecs, attention_weightvecs])

    cand_title =[Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(1+npratio)]
    cand_body =[Input(( MAX_BODY_LENGTH,), dtype='int32')  for _ in range(1+npratio)]
    cand_titlerep=[ title_encoder(cand_title[_]) for _ in range(1+npratio)]
    #cand_bodyrep=[ bodyEncodert(cand_body[_]) for _ in range(1+npratio)]

    cand_pos_title = keras.Input((MAX_SENT_LENGTH,))
    cand_pos_body = keras.Input((MAX_BODY_LENGTH,))
    cand_one_bodyrep = bodyEncodert([cand_pos_body])

    dense1=Dense(100)
    dense2=Dense(100)
    predspeed =  keras.layers.dot([dense1(userrep), dense2(cand_one_bodyrep)], axes=-1)
    logits = concatenate([keras.layers.dot([userrep, cvec], axes=-1) for cvec in cand_titlerep])
    logits = keras.layers.Activation(keras.activations.softmax)(logits)

    model = Model(cand_title+[his_title_input]+cand_body+[his_body_input,time_input,cand_pos_title,cand_pos_body], [logits,predspeed ])
    model.compile(loss=['categorical_crossentropy','mae'], optimizer=Adam(lr=0.001), metrics=['acc','mae'],loss_weights=[1.,0.4])

    cand_one_title = keras.Input((MAX_SENT_LENGTH,))
    cand_one_body = keras.Input((MAX_BODY_LENGTH,))
    cand_one_vec = title_encoder([cand_one_title])

    score = keras.layers.Activation(keras.activations.sigmoid)(dot([userrep, cand_one_vec], axes=-1))
    model_test = keras.Model([cand_one_title,his_title_input,cand_one_body,his_body_input,time_input], score)

    for ep in range(2):
        traingen=generate_batch_data(30)
        model.fit_generator(traingen, epochs=1,steps_per_epoch=len(label)//30)
    testgen=generate_batch_data_test(30)
    auc=[]
    mrr=[]
    ndcg5=[]
    ndcg10=[]
    pred = model_test.predict_generator(testgen, steps=len(test_pn)//30,verbose=1)

    auc=[]
    mrr=[]
    ndcg5=[]
    ndcg10=[]
    for m in test_impression_index:
        if np.sum(test_label[m[0]:m[1]])!=0 and m[1]<len(pred):
            auc.append(roc_auc_score(test_label[m[0]:m[1]],pred[m[0]:m[1],0]))
            mrr.append(mrr_score(test_label[m[0]:m[1]],pred[m[0]:m[1],0]))
            ndcg5.append(ndcg_score(test_label[m[0]:m[1]],pred[m[0]:m[1],0],k=5))
            ndcg10.append(ndcg_score(test_label[m[0]:m[1]],pred[m[0]:m[1],0],k=10))
    results.append([np.mean(auc),np.mean(mrr),np.mean(ndcg5),np.mean(ndcg10)])
    print(np.mean(auc),np.mean(mrr),np.mean(ndcg5),np.mean(ndcg10))