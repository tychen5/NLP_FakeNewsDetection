# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:29:46 2019

@author: tina
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk 
from nltk.tokenize import word_tokenize
import re
import os
os.chdir(r"C:\Users\tina\Desktop\台大\碩二\碩二上\資訊檢索與文字探勘導論\期末報告")
#os.getcwd() #印出目前工作目錄

# data combine-----------------------------------------------------------------

# function combineData: combine all training data =============================
def combineData():
    # 0: reliable, 1: unreliable
    name = r'C:\\Users\\tina\\Desktop\\train.csv'
    kaggleData1 = pd.read_csv(name, encoding='utf-8', header=0)
    kaggleData1 = kaggleData1.fillna(' ')
    
    # bias, bs, conspiracy, fake, hate, junksci, satire, state
    name = r'C:\\Users\\tina\\Desktop\\fake.csv'
    kaggleData2 = pd.read_csv(name, encoding='utf-8', header=0)
    kaggleData2 = kaggleData2.fillna(' ')
    
    # true, mostly true, half true, barely-true, false, pants-fire
    name = r'C:\\Users\\tina\\Desktop\\train.tsv'
    ucsb1 = pd.read_csv(name, sep='\t', header=None)
    ucsb1 = ucsb1.fillna(' ')
    name = r'C:\\Users\\tina\\Desktop\\test.tsv'
    ucsb2 = pd.read_csv(name, sep='\t', header=None)
    ucsb2 = ucsb2.fillna(' ')
    name = r'C:\\Users\\tina\\Desktop\\valid.tsv'
    ucsb3 = pd.read_csv(name, sep='\t', header=None)
    ucsb3 = ucsb3.fillna(' ')
    
    a1 = kaggleData1[["text","label"]]
    a2 = kaggleData2[["text","type"]]
    a3 = ucsb1[[2,1]]
    a4 = ucsb2[[2,1]]
    a5 = ucsb3[[2,1]]
    a2.columns = ["text","label"]
    a3.columns = ["text","label"]
    a4.columns = ["text","label"]
    a5.columns = ["text","label"]
    
    combine_data = pd.concat([a1,a2,a3,a4,a5],axis=0)
    combine_data = combine_data.reset_index(drop=True)
    combine_data['text'] = combine_data['text'].str.lower() # lowercase all characters
    #combine_data.to_csv('combine_data.csv', index=False) # write to a csv
    return combine_data
#==============================================================================
combine_data = combineData()


# POS analysis-----------------------------------------------------------------

# function PosTag =============================================================
def PosTag(combine_data):
    postags_0 = [] # pos-tag of type 1 news: unreliable, fake, false, pants-fire
    postags_1 = [] # pos-tag of type 0 news: reliable, true
    postags_bias = []
    postags_conspiracy = []
    postags_hate = []
    postags_junksci = []
    postags_satire = []
    postags_state = []
    num = [0] * 8  # num[0]: number of type 0 news, num[1]: number of type 1 news,...
    
    for x in range(combine_data.shape[0]):  
        text = word_tokenize(re.sub(r'[^a-z ]', '', combine_data.loc[x, 'text']))
        #text = word_tokenize(combine_data.loc[x, 'text'])
        pos = list(set(nltk.pos_tag(text, tagset='universal'))) # list(set(x)): remove repeated tuple in x
    
        if combine_data.loc[x, 'label'] in [1,'fake','false','pants-fire']:
            num[0] = num[0] + 1
            postags_0.extend(pos) 
        elif combine_data.loc[x, 'label'] in [0,'true']:
            num[1] = num[1] + 1
            postags_1.extend(pos)   
        elif combine_data.loc[x, 'label'] == 'bias':
            num[2] = num[2] + 1
            postags_bias.extend(pos)
        elif combine_data.loc[x, 'label'] == 'conspiracy':
            num[3] = num[3] + 1
            postags_conspiracy.extend(pos)
        elif combine_data.loc[x, 'label'] == 'hate':
            num[4] = num[4] + 1
            postags_hate.extend(pos)  
        elif combine_data.loc[x, 'label'] == 'junksci':
            num[5] = num[5] + 1
            postags_junksci.extend(pos)
        elif combine_data.loc[x, 'label'] == 'satire':
            num[6] = num[6] + 1
            postags_satire.extend(pos)   
        elif combine_data.loc[x, 'label'] == 'state':
            num[7] = num[7] + 1
            postags_state.extend(pos)
    postags = [postags_0,postags_1,postags_bias,postags_conspiracy,postags_hate,postags_junksci,postags_satire,postags_state]
    return [num,postags]
#==============================================================================
postags = PosTag(combine_data)
N = sum(postags[0])
num = postags[0]
postags = postags[1]


# compute document frequency for each term-------------------------------------
POS = ['NOUN','VERB','ADJ','ADV']
TAG = ['postags_0','postags_1','postags_bias','postags_conspiracy','postags_hate','postags_junksci','postags_satire','postags_state','postagsAll']

# function frequency ==========================================================
def frequency(POS,TAG,postags):
    postags.append([item for sublist in postags for item in sublist]) 
    
    posNum = []
    for i in POS:   
        a = []   
        for j in range(len(TAG)):
            fd = nltk.FreqDist(postags[j]) # 計算每個(詞,詞性)出現的頻率 (即出現在幾篇新聞中)
            a.append(pd.DataFrame([(wt[0], _) for (wt, _) in fd.most_common() if wt[1] == i])) # 按照各詞性出現的頻率由高至低排列
        posNum.append(dict(zip(TAG, a)))
    posNum = dict(zip(POS, posNum))
    
    termDF = [] # for x in POS, find term's document frequency
    for i in POS:
        df = pd.DataFrame(np.zeros((len(posNum[i]['postagsAll']), (len(TAG)-1))),index=posNum[i]['postagsAll'][0]) # zero matrix
        for j in range(len(TAG)-1):
            df.loc[list(posNum[i][TAG[j]][0]),j] = list(posNum[i][TAG[j]][1])
        termDF.append(df)
    termDF = dict(zip(POS, termDF))
    return [posNum,termDF]
#==============================================================================
termDF = frequency(POS,TAG,postags)
posNum = termDF[0]
termDF = termDF[1]


# compute Xsq, log likelihood ratio, expected mutual information for each term-

# function score: xsq,llr,emi =================================================
def Score(POS,TAG,posNum,termDF,num):
    import math
    SCORE = []
    for i in POS:
        n = len(posNum[i]['postagsAll'])
        xsq = np.zeros((n, (len(TAG)-1))) # zero matrix
        llr = np.zeros((n, (len(TAG)-1)))
        emi = np.zeros((n, (len(TAG)-1)))
        
        for j in range(n):
            for k in range(len(TAG)-1):
                n11 = termDF[i].iloc[j,k]
                n12 = num[k] - n11
                n21 = sum(list(termDF[i].iloc[j,])) - n11
                n22 = N - n11 - n12 - n21
                c1 = n11 + n21
                c2 = n12 + n22
                r1 = n11 + n12
                r2 = n21 + n22
                xsq[j][k] = (n11-c1*r1/N)**2/(c1*r1/N) + (n12-c2*r1/N)**2/(c2*r1/N) + (n21-c1*r2/N)**2/(c1*r2/N) + (n22-c2*r2/N)**2/(c2*r2/N)
                pt = c1/N
                p1 = n11/r1
                p2 = n21/r2
                if pt == 0:
                    llr[j][k] = -10000000000000000
                else:
                    llr[j][k] = (n11 + n21)*math.log(pt)
                if (1-pt) == 0:
                    llr[j][k] = llr[j][k] - 10000000000000000
                else:
                    llr[j][k] = llr[j][k] + (n12 + n22)*math.log(1-pt)       
                if p1 == 0:
                    llr[j][k] = llr[j][k] + 10000000000000000
                else:
                    llr[j][k] = llr[j][k] - n11*math.log(p1)    
                if (1-p1) == 0:
                    llr[j][k] = llr[j][k] + 10000000000000000
                else:
                    llr[j][k] = llr[j][k] - n12*math.log(1-p1)
                if p2 == 0:
                    llr[j][k] = llr[j][k] + 10000000000000000
                else:
                    llr[j][k] = llr[j][k] - n21*math.log(p2)
                if (1-p2) == 0:
                    llr[j][k] = llr[j][k] + 10000000000000000
                else:
                    llr[j][k] = llr[j][k] - n22*math.log(1-p2)
                llr[j][k] = -2 * llr[j][k]
                
                if (n11*N)/(c1*r1) == 0:
                    emi[j][k] = n11/N * (-10000000000000000)
                else:
                    emi[j][k] = n11/N * math.log((n11*N)/(c1*r1),2)
                if (n12*N)/(c2*r1) == 0:
                    emi[j][k] = emi[j][k] + n12/N * (-10000000000000000)
                else:
                    emi[j][k] = emi[j][k] + n12/N * math.log((n12*N)/(c2*r1),2)
                if (n21*N)/(c1*r2) == 0:
                    emi[j][k] = emi[j][k] + n21/N * (-10000000000000000)
                else:
                    emi[j][k] = emi[j][k] + n21/N * math.log((n21*N)/(c1*r2),2)
                if (n22*N)/(c2*r2) == 0:
                    emi[j][k] = emi[j][k] + n22/N * (-10000000000000000)
                else:
                    emi[j][k] = emi[j][k] + n22/N * math.log((n22*N)/(c2*r2),2) 
        score = np.array([[sum(xsq[i]) for i in range(n)], [sum(llr[i]) for i in range(n)], [sum(emi[i]) for i in range(n)], [0]*n, [0]*n])
        SCORE.append(pd.DataFrame(score.T, index = posNum[i]['postagsAll'][0], columns = ['xsq','llr','emi','tfidf','vote'])) 
    SCORE = dict(zip(POS, SCORE))
    return SCORE
#==============================================================================

# function: constructDictionary ===============================================
def constructDictionary(trainingData,frequency):
    import re
    token = []
    if type(trainingData) is list:
        if frequency == "term":
            for i in range(len(trainingData)): 
                #token.extend(word_tokenize(trainingData[i]))
                token.extend(word_tokenize(re.sub(r'[^a-z ]', '', trainingData[i])))
        elif frequency == "doc":
            for i in range(len(trainingData)):
                #token.extend(set(word_tokenize(trainingData[i])))
                token.extend(set(word_tokenize(re.sub(r'[^a-z ]', '', trainingData[i])))) # here we count the document frequenc
    else:
         if frequency == "term":
             #token.extend(word_tokenize(trainingData))
             token.extend(word_tokenize(re.sub(r'[^a-z ]', '', trainingData)))
         elif frequency == "doc":
             #token.extend(set(word_tokenize(trainingData)))
             token.extend(set(word_tokenize(re.sub(r'[^a-z ]', '', trainingData)))) # here we count the document frequenc

    from collections import Counter
    dictionary = pd.DataFrame.from_dict(Counter(token), orient='index') # Transform a Counter object into a Pandas DataFrame
    dictionary.columns = ['frequency']
    dictionary.sort_index(inplace=True) # sort dataframe by index
    dictionary['index'] = list(range(dictionary.shape[0]))   
    return dictionary
#==============================================================================

# function: isEnglish =========================================================
def isEnglish(string):
    try:
        string.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
#==============================================================================

# function: containChar =======================================================
def containChar(string):
    letter_flag = False
    for i in string:
        if i.isalpha():
            letter_flag = True
    return letter_flag
#==============================================================================

# function tfidf ==============================================================
def tfidf(combine_data):
    import math
    label = [1,'fake','false','pants-fire',0,'true','bias','conspiracy','hate','junksci','satire','state']
    idx = [t for t, j in enumerate(list(combine_data['label'])) if j in label]
    allnews = list(combine_data.iloc[idx,0])
    term_df = constructDictionary(allnews,'doc')
    term_tfidf = pd.DataFrame([0] * term_df.shape[0], index = term_df.index.tolist())
    term_nd = pd.DataFrame([0] * term_df.shape[0], index = term_df.index.tolist())
    for news in allnews:
        if not news.isspace() and isEnglish(news) and containChar(news):
            tf = constructDictionary(news,'term')
            tfidf = np.array(list(tf.loc[:,'frequency'])) * np.array([math.log(y) for y in list(term_df.loc[tf.index.tolist(),'frequency'])])
            term_tfidf.loc[tf.index.tolist(),0] = term_tfidf.loc[tf.index.tolist(),0] + [float(i)/sum(tfidf) for i in tfidf] # normalize
            term_nd.loc[tf.index.tolist(),0] = term_nd.loc[tf.index.tolist(),0] + 1
    term_tfidf = np.array(list(term_tfidf.loc[:,0])) / np.array(list(term_nd.loc[:,0]))
    term_tfidf = pd.DataFrame(term_tfidf, index = term_df.index.tolist())
    return term_tfidf
#==============================================================================
score = Score(POS,TAG,posNum,termDF,num)
term_tfidf = tfidf(combine_data)
for i in POS:
    score[i].loc[:,'tfidf'] = term_tfidf.loc[score[i].index.tolist(),0]


# vote each term if its Xsq, LLR, EMI value larger than average----------------

# function voting =============================================================
def voting(score):
    for i in POS:     
        threshold_xsq = np.min(score[i].loc[:,'xsq']) #1=>350
        threshold_llr = np.min(score[i].loc[:,'llr']) #1.75=>543
        threshold_emi = np.min(score[i].loc[:,'emi']) 
        threshold_tfidf = np.min(score[i].loc[:,'tfidf']) #1.45=>502 數字大嚴格
        
        score[i] = score[i][score[i].loc[:,'xsq']>threshold_xsq]
        score[i] = score[i][score[i].loc[:,'llr']>threshold_llr]
        score[i] = score[i][score[i].loc[:,'emi']>threshold_emi]
        score[i] = score[i][score[i].loc[:,'tfidf']>threshold_tfidf]
        
        threshold_xsq = np.mean(score[i].loc[:,'xsq'])+0*np.std(score[i].loc[:,'xsq']) #1=>350
        threshold_llr = np.mean(score[i].loc[:,'llr'])+0*np.std(score[i].loc[:,'llr']) #1.75=>543
        threshold_emi = np.mean(score[i].loc[:,'emi'])+0*np.std(score[i].loc[:,'emi']) #1.75=>543
        threshold_tfidf = np.mean(score[i].loc[:,'tfidf'])+0*np.std(score[i].loc[:,'tfidf']) #1.45=>502 數字大嚴格
        
        df1 = score[i][score[i].loc[:,'xsq']>threshold_xsq]
        df2 = score[i][score[i].loc[:,'llr']>threshold_llr]
        df3 = score[i][score[i].loc[:,'emi']>threshold_emi]
        df4 = score[i][score[i].loc[:,'tfidf']>threshold_tfidf]
        
        score[i].loc[df1.index.tolist(),'vote'] += 1
        score[i].loc[df2.index.tolist(),'vote'] += 1
        score[i].loc[df3.index.tolist(),'vote'] += 1
        score[i].loc[df4.index.tolist(),'vote'] += 1
        score[i].to_csv(i+".csv")
    return score
#==============================================================================
score1 = score
score = voting(score)


# naive bayes classification---------------------------------------------------

# function: trainMultinomialNB ================================================
def trainMultinomialNB(trainingData):
    flat_trainingData = [item for sublist in trainingData for item in sublist]
    V = constructDictionary(flat_trainingData,"term") # terms in training set
    N = len(flat_trainingData) # number of documents
    prior = [0] * len(trainingData)
    condprob = np.zeros((V.shape[0],len(trainingData))) # zero matrix
    for i in range(len(trainingData)):
        prior[i] = len(trainingData[i])/N      
        Vc = constructDictionary(trainingData[i],"term")
        condprob[list(V.loc[Vc.index.tolist(),'index']),i] = list(Vc.loc[:,'frequency'])
    condprob = condprob + 1
    rowsum = np.array([sum(condprob[i]) for i in range(V.shape[0])])
    condprob = condprob/rowsum[:,np.newaxis] # divide each column by a vector element
    condprob = pd.DataFrame(condprob, index = V.index.tolist())
    return [V,prior,condprob]
#==============================================================================

# function: testMultinomialNB =================================================
def testMultinomialNB(trainNB,featureTerms,news):
    W = constructDictionary(news,"term")
    c = trainNB[1]
    idx = [i for i in W.index.tolist() if i in featureTerms]
    for i in range(len(c)):
        for j in idx:
                c[i] = c[i] + trainNB[2].loc[j,i]*W.loc[j,'frequency']
    return c.index(max(c))
#==============================================================================

#-----------------------------------------
# prepare training data news
trainingData = []
label = [[1,'fake','false','pants-fire'],[0,'true'],['bias'],['conspiracy'],['hate'],['junksci'],['satire'],['state']]
for i in range(len(label)):
    idx = [t for t, j in enumerate(list(combine_data['label'])) if j in label[i]]
    trainingData.append(list(combine_data.iloc[idx,0]))
    
# training
trainNB = trainMultinomialNB(trainingData)

# feature selection
featureTerms = []
for i in POS: 
    featureTerms.append(list(np.array(score[i].index.tolist())[[i for i, e in enumerate(list(score[i].loc[:,'vote'])) if e == 4]]))
featureTerms = dict(zip(POS, featureTerms))

# testing
name = r'C:\\Users\\tina\\Desktop\\test.csv'
testingData = pd.read_csv(name, encoding='utf-8', header=0)
testingData = testingData.fillna(' ')

POS = ['NOUN','VERB','ADJ','ADV']
a = []
for i in POS:
    newsType = [0] * testingData.shape[0]
    for j in range(testingData.shape[0]):
        if testingData.loc[j,'text'].isspace():
            newsType[j] = 0
            #newsType[j] = testMultinomialNB(trainNB,featureTerms[i],testingData.loc[j,'title'])
        elif isEnglish(testingData.loc[j,'text']):
            newsType[j] = testMultinomialNB(trainNB,featureTerms[i],testingData.loc[j,'text'])
        else:
            newsType[j] = 0
    testingData["label"] = newsType
    a.append(newsType)
    testingData.loc[:,["label",'id']].to_csv('test_'+i+'.csv', index=False)

#-----------------------------------------------------------------------------
newsType = pd.DataFrame(np.zeros((testingData.shape[0],4)), columns = POS)
for i in range(testingData.shape[0]):
    if testingData.loc[i,'text'].isspace():
        for j in POS:
            newsType.loc[i,j] = 0
    elif isEnglish(testingData.loc[i,'text']):
        text = word_tokenize(re.sub(r'[^a-z ]', '', testingData.loc[i,'text']))
        pos = nltk.pos_tag(text, tagset='universal') 
        for j in POS:
            a = [t for (t, p) in pos if p == j]
            a = ' '.join(a)
            if a != "":
                newsType.loc[i,j] = testMultinomialNB(trainNB,featureTerms[j],a)
    else:
        for j in POS:
            newsType.loc[i,j] = 0

for i in POS:
    testingData["label"] = list(newsType.loc[:,i])
    testingData.loc[:,["label",'id']].to_csv('test_'+i+'.csv', index=False)
        
