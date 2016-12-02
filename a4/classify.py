"""
classify.py
"""
import requests
import pickle
import sys
import numpy as np
from TwitterAPI import TwitterAPI
from scipy.sparse import lil_matrix

import time
import re

consumer_key = 'JbNmHiCvktwSRFeWwvYbEgBIu'
consumer_secret = '6WpENU31DWOZuKgZ61jmRguRezynhQubKMDhYVLeWXliiwksNS'
access_token = '1376939035-dC2nd33K47GiCw3y8KtHZWAATrM2qgnS6fmUTEW'
access_token_secret = 'xCW918qTRl2vnWxaecQbSdVeuPho1MoD9xsJqU1EBb6ai'
call=0
def get_twitter():

  
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def tokenize(string, lowercase, keep_punctuation, prefix,
             collapse_urls, collapse_mentions):
   
    if not string:
        return []
    if lowercase:
        string = string.lower()
    tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
    if keep_punctuation:
        tokens = string.split()
    else:
        tokens = re.sub('\W+', ' ', string).split()
    if prefix:
        tokens = ['%s%s' % (prefix, t) for t in tokens]
    return tokens


def tweet2tokens(tweet, use_descr=True, lowercase=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True):
   
    tokens = tokenize(tweet['text'], lowercase, keep_punctuation, None,
                       collapse_urls, collapse_mentions)
    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'], lowercase,
                               keep_punctuation, descr_prefix,
                               collapse_urls, collapse_mentions))
    return tokens
    
    


def get_gender(tweet, male_names, female_names):
    name = get_first_name(tweet)
    if name in female_names:
        return 1
    elif name in male_names:
        return 0
    else:
        return -1
    
   

def get_census_names():
   
    males = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')
    males_pct = dict([(m.split()[0].lower(), float(m.split()[1]))
                  for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                    for f in females if f])
    male_names = set([m for m in males_pct if m not in females_pct or
                  males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or
                  females_pct[f] > males_pct[f]])  
                  
                  
    #print('found %d female and %d male names' % (len(male_names), len(female_names)))
    #print('male name sample:', list(male_names)[:5])
    #print('female name sample:', list(female_names)[:5])              
    return male_names, female_names

def robust_request(twitter, resource, params, max_tries=5):
   
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        elif request.status_code == 88:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)
        else:
            continue
            



def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()


from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression


def build_logisticR_model(X, y, nfolds):

    cv = KFold(len(y), nfolds)

    for train_idx, test_idx in cv:
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
      
      
    return clf

def make_feature_matrix(tweets,tokens_list, vocabulary):
    X = lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j = vocabulary[token]
            X[i,j] += 1
    return X.tocsr()
from collections import defaultdict

def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))  
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]  
    #print('%d unique terms in vocabulary' % len(vocabulary))
    return vocabulary


def main():
    
    

    male_names, female_names = get_census_names()
   
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f) 
    
              
  
     
    
    with open('tweets.pkl','rb') as tw:
        tweets=pickle.load(tw) 
    y = np.array([get_gender(t, male_names, female_names) for t in tweets])

    
       
    
    results['totaltweets']=len(tweets) 
    tokens_list = [tweet2tokens(t, use_descr=True, lowercase=True,
                            keep_punctuation=True, descr_prefix='d=',
                            collapse_urls=True, collapse_mentions=True)
              for t in tweets]
    vocabulary = make_vocabulary(tokens_list)  
    X = make_feature_matrix(tweets,tokens_list, vocabulary) 
    
    clf = build_logisticR_model(X, y, 2)
    malex=1
    femalex=1
    it = np.nditer(y, flags=['f_index'])
    while not it.finished:
        predicted = clf.predict(X[it.index])
        if(malex==0 and femalex==0):
            break
        if  it[0] == 0 and predicted==it[0] and malex:
            #print(it.index,X[it.index], "--Male ")
            test_tweet=tweets[it.index]
            results['male']=test_tweet
            #print('test tweet:\n\tscreen_name=%s\n\tname=%s\n\tdescr=%s\n\ttext=%s' %
            #(test_tweet['user']['screen_name'],
             #test_tweet['user']['name'],
             #test_tweet['user']['description'],
             #test_tweet['text']))
            malex=0
        if  it[0] == 1 and predicted ==it[0] and femalex:
            #print(it.index,X[it.index],"--Female")
            test_tweet=tweets[it.index]
            results['female']=test_tweet
            #print('test tweet:\n\tscreen_name=%s\n\tname=%s\n\tdescr=%s\n\ttext=%s' %
            #(test_tweet['user']['screen_name'],
             #test_tweet['user']['name'],
             #test_tweet['user']['description'],
             #test_tweet['text']))
            femalex=0
        it.iternext()    
    
    pickle.dump(results, open('results.pkl', 'wb'))         
if __name__ == '__main__':
    main()     