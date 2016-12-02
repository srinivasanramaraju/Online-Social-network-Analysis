# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:31:37 2016

@author: Nivash
"""

import sys
import time
import requests
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from TwitterAPI import TwitterAPI



consumer_key = 'JbNmHiCvktwSRFeWwvYbEgBIu'
consumer_secret = '6WpENU31DWOZuKgZ61jmRguRezynhQubKMDhYVLeWXliiwksNS'
access_token = '1376939035-dC2nd33K47GiCw3y8KtHZWAATrM2qgnS6fmUTEW'
access_token_secret = 'xCW918qTRl2vnWxaecQbSdVeuPho1MoD9xsJqU1EBb6ai'


def get_twitter():
  
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret) 


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
            
def read_screen_names(filename):
  
    with open(filename,'r') as scrnamesfile:
        scrnames=scrnamesfile.read().splitlines()
        return scrnames
    pass


def get_users(twitter, screen_names):
  
    ###TODO
    usrresults=robust_request(twitter,"users/lookup",{"screen_name":screen_names})        
    return usrresults
    
def create_graph(users, friend_counts):
   
    graph = nx.Graph()
    for user in users:
        graph.add_node(user['screen_name'])
        for each in user['followers'][:250]:
            #friend_counts[each] > 1:
                graph.add_node(each)
                graph.add_edge(user['screen_name'],each)
    nx.write_gpickle(graph,"twittergraph.gpickle")            
    return graph
    pass    

def count_friends(users):
 
    cnt=Counter()
    for user in users:
        for key in user['followers']:
            cnt[key]+=1
    return cnt    
    pass


def get_friends(twitter, screen_name):
   
    frndslistjsonresults = robust_request(twitter,"followers/ids",{"screen_name":screen_name,"Count":5000})          
    frndslistjson=frndslistjsonresults.json()
    frndslist=frndslistjson['ids']
    return sorted(frndslist,key=int)
    pass


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


def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()


def sample_tweets(twitter, male_names, female_names,userid,tweets):
    
    x=-1
    responses=robust_request(twitter,'statuses/user_timeline',{'user_id':userid,"count":10})
    if responses != None:
        for response in responses:
        #print('inside for loop')            
            if 'user' in response:
            #print("user in response")
                name = get_first_name(response)
                if name in male_names: 
                #print('user in census')
                    tweets.append(response)
                    print("Fetched %d Tweets" % len(tweets))
                    x=0
                if name in female_names: 
                    tweets.append(response)
                    print("Fetched %d Tweets" % len(tweets))
                    x=1
    #call=call+1          
    #print(call,len(tweets))                 
    
    return tweets,x

def add_all_friends(twitter, users):
 
    usrdic={}
    for user in users:
        usrdic['screen_name'] = user['screen_name']
        usrfrndresults = get_friends(twitter,usrdic['screen_name'])
        user['followers'] = usrfrndresults
    pass

def print_num_friends(users):
  
    printtuplelist=[]
    for user in users:
        printtuple=user['screen_name'],len(user['followers'])
        printtuplelist.append(printtuple)
    sorted(printtuplelist,key=lambda x:x[0])
    for tup in printtuplelist:
        print(tup[0], tup[1])
    pass


def draw_network(graph, users, filename):
   
   
    plt.figure(figsize=(12,12))
    labels={nodes: '' if isinstance(nodes,int) else nodes for nodes in graph.nodes()}
    nx.draw_networkx(graph,labels=labels, alpha=.5, width=.1,
                     node_size=100)
    plt.axis("off")
    plt.savefig(filename)
    pass

def main():
    male_names, female_names = get_census_names()
    limit=1800
    tweets=[] 
    followerslst=[]
    twitter = get_twitter()
    screen_names = read_screen_names('players.txt')
    print('Established Twitter connection.')
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    print('Followers per candidate: limiting to 5000 each')
    add_all_friends(twitter, users)
    print_num_friends(users)
    friend_counts = count_friends(users) 
    pickle.dump(users, open('users.pkl', 'wb'))
    graph = create_graph(users, friend_counts)
    draw_network(graph, users, 'network.png')
    print("Follower data collected and a graph object created- network .png ")
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f) 
    for user in users:
        for each in user['followers'][:250]:
            if each not in followerslst:
                followerslst.append(each)
                
    results['malecount']=0
    results['femalecount']=0             
    for eachuser in followerslst:
        #print(eachuser)
        tweets,x=sample_tweets(twitter, male_names, female_names,eachuser,tweets) 
        if x==0:
           results['malecount']+=1
        if x==1:
           results['femalecount']+=1 
        if(len(tweets)>limit):
            break 
    print("fetched %d tweets " % len(tweets))
    results['messages']=len(tweets)    
    pickle.dump(tweets, open('tweets.pkl', 'wb'))  
    pickle.dump(results, open('results.pkl', 'wb'))
    print("Data Collection Over")
if __name__ == '__main__':
    main()   