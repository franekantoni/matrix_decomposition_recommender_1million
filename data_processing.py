#USE PYTHON2


import numpy as np
import pickle
from sklearn.preprocessing import normalize
import pandas as pd
import re

def strip_quote(astring):
	if astring[0] == '"':
		return astring[1:]
	else:
		return astring


####.   DATA  #####  PROCESSING.  ###############################################

#load the data
diction = {}

with open("ratings.dat", "r") as data:
	for line in data:
		UserID, MovieID, Rating, Timestamp= line.split("::")
		if not diction.get(int(UserID)):
			diction[int(UserID)] = {int(MovieID) : int(Rating)}
		else:
			diction[int(UserID)][int(MovieID)] = int(Rating)

#make data matrix sparse
data = pd.DataFrame(diction).T

#change NaN to zeros
data = data.fillna(0)

#create a dict with indexes as keys and movie ids as values
index_to_id = {}
for index, id_ in enumerate(data):
	index_to_id[index] = id_

#create a dict with movie ids as keys and indexes as values
id_to_index = {v: k for k, v in index_to_id.iteritems()}

#create a dict with indexes as keys and titles as values:
index_to_title = {}

with open("movies.dat", "r") as movies:
	for line in movies:
		MovieID, Title, Type = line.split("::")
		if id_to_index.get(int(MovieID)):
			index_to_title[id_to_index[int(MovieID)]] = Title

#easy bugfix
index_to_title[0] = "Toy Story (1995)"
#create a dict with titles as keys and indexes as values:
title_to_index = {v: k for k, v in index_to_title.iteritems()}



with open('index_to_title_dict.pickle', 'wb') as handle:
    pickle.dump(index_to_title, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('title_to_index_dict.pickle', 'wb') as handle2:
    pickle.dump(title_to_index, handle2, protocol=pickle.HIGHEST_PROTOCOL)


#normalize the ratings for each user
data = normalize(data, norm='l2', axis=1)
with open('pickled_normalized_matrix.sav', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

