# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    tokenslist=[]
    for index,eachgenre in movies.iterrows():
        tokenslist+=[tokenize_string(eachgenre['genres'])]
    movies['tokens']=tokenslist
    
    return movies
    
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    allgenres=[]
    genredata=[]
    
    csr_matrixlist=[]
    docgenrecount=defaultdict(lambda:0)
    vocab={}
    i=0
    row=0
     
    for genreslist in movies['tokens'].tolist():
        genrecount=defaultdict(lambda:0)
        onedoc=defaultdict(lambda:1)
        for eachgenre in genreslist:
            genrecount[eachgenre]+=1
            if eachgenre not in docgenrecount or onedoc[eachgenre] == 1:
                docgenrecount[eachgenre]+=1
                onedoc[eachgenre] = 0
            if eachgenre not in allgenres:
                allgenres.append(eachgenre)
            maxkey=max(genrecount, key=genrecount.get) 
            tempdata=[]
        for genre,count in genrecount.items():
            tempdata+=[(genre,count,genrecount[maxkey])]
        genredata+=[tempdata]    
            
    allgenres=sorted(allgenres)
    for eachgenre in allgenres:
        if eachgenre not in vocab:
            vocab[eachgenre]=i
            i=i+1 
    
    for genretuplelist in genredata:
        matrixdata=[]
        matrixrow=[]
        matrixcol=[]
        for genretuple in genretuplelist:
            matrixdata.append((genretuple[1]/genretuple[2]) * math.log((len(movies['tokens'].tolist())/docgenrecount[genretuple[0]]),10))
            matrixrow.append(0)
            matrixcol.append(vocab[genretuple[0]])
        csrdata=np.array(matrixdata)
        csrrow=np.array(matrixrow) 
        csrcol=np.array(matrixcol)
        X=csr_matrix((csrdata, (csrrow, csrcol)),shape=(1,len(vocab))) 
        csr_matrixlist+=X
    movies['features']=pd.DataFrame(csr_matrixlist)  
    
    return movies,vocab       
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
   
    
    
    a_array=a.toarray()
    b_array=b.toarray()
    cosine_sim=(a_array.dot(b_array.transpose()))/(np.sqrt(a.multiply(a).sum(1)).dot(np.sqrt(b.multiply(b).sum(1)).transpose()))
    return cosine_sim
    ###TODO
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
   
    predicted_ratings=[]
    for index, testmovie in ratings_test.iterrows():
        
        testmoviefeature=movies[movies['movieId']==testmovie['movieId']]
        for index,testmoviefeaturerow  in testmoviefeature.iterrows():
            predictmoviefeature=testmoviefeaturerow['features']    
    
        userratedmovies=ratings_train[ratings_train['userId']==testmovie['userId']] 
       
  
        ratings_list=[]
        positive_ratings=0
        simvalue_list=[]
        for index,eachmovie in userratedmovies.iterrows():
            
            eachmoviefeature=movies[movies['movieId']==eachmovie['movieId']]
            for index,moviefeature in eachmoviefeature.iterrows():
                predictedmoviefeature=moviefeature['features']
                
                sim_value=cosine_sim(predictmoviefeature,predictedmoviefeature)
                
                if(sim_value > 0):
                    positive_ratings=1
                    ratings=ratings_train[(ratings_train['movieId']==eachmovie['movieId'])&(ratings_train['userId']==testmovie['userId'])]
                    for index,ratings in ratings.iterrows():
                        userrating=ratings['rating']
                    ratings_list.append(sim_value*userrating)
                    simvalue_list.append(sim_value)
        if positive_ratings == 1:           
            predicted_ratings.append(np.sum(ratings_list)/np.sum(simvalue_list) )
        else:
            predicted_ratings.append(np.mean(userratedmovies['rating']))
    
           
    return np.array(predicted_ratings)          
             
    
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
