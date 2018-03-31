#PYTHON2
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
import numpy as np 
import pickle
from helpers import strip_quote
import pandas as pd
import random
from sklearn.svm import SVC
import time


def compure_rec(users, model, n=10):
	"""
	takes np.array of users, model and number of recomendations to be returned
	returns list of n recommendations
	"""

	#get the dot product of user matrix and factofised one -> fill the recomendation matrix for those users
	users = model.transform(users)
	H = model.components_
	rec = (np.dot(users, H))

	#create a list of n highest reccommended movies for each user [lis of lists]

	lst_of_recs = []

	for user in rec:
		user_lst = []
		for i, score in enumerate(user):
			user_lst.append((score, i))
		user_lst = sorted(user_lst, reverse = True)
		lst_of_recs.append([s for i, s in user_lst[:n]])

	return lst_of_recs



def get_the_favs(users, n):
	"""
	takes a np.array of users and number of favourite movies to be returned
	returns a list of n favourite movies for each user [lis of lists]
	"""

	lst_of_real_scores = []

	for user in users:
		user_lst = []
		for i, score in enumerate(user):
			user_lst.append((score, i))
		user_lst = sorted(user_lst, reverse = True)
		lst_of_real_scores.append([s for i, s in user_lst[:n]])
		

	return lst_of_real_scores




def print_comarison(lst_of_real_scores, lst_of_recs, n=10):
	"""
	takes a list of lists of favourite movies [->get_the_favs()] of users (or a single user)
	a list of lists of recommendations [->compure_rec()] of users (or a single user)
	and an int n
	for each user prints theirs n favourite movies and n best recommendations
	"""
	user_cnt = 1
	for recomendation, favourites in zip(lst_of_recs, lst_of_real_scores):

		recomendation = recomendation[:n]
		favourites = favourites[:n]

		print "\n\n\nUSER", user_cnt 
		user_cnt+=1

		print("\nFAVOURITES:\n")
		for fav in favourites:
			title = index_to_title[fav]
			print(strip_quote(title))

		print("\nRECOMENDATIONS:\n")
		for movie in recomendation:
			title = index_to_title[movie]
			print(strip_quote(title))

	return 0


def train_and_pickle(data_file_name, model = NMF(n_components=40, init='random', random_state=0, max_iter=500)):
	"""
	takes a NMF as a default model
	takes data_file_name (file name of a saved normalized matrix) as input
	trains the model and pickles it for later
	returns a trained model
	to train model with 1m dataset: train_and_pickle("pickled_normalized_matrix.sav")
	"""

	with open(data_file_name, "rb") as f:
		data = pickle.load(f)
	model.fit(data)
	with open('pickled_model.pickle', 'wb') as handle:
	    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return model


def catalogue():
	"""
	prints title, index pairs in alphabetical order
	creates a file containing title, index pairs
	"""
	with open("katalog.txt", "w") as katalog:

		for title in sorted(title_to_index):
			print title, ":", title_to_index[title]
			katalog.write(strip_quote(title)+" : "+str(title_to_index[title])+"\n")
	return 0



def user_vector(favs, num_of_all_movies = 9066):
	"""
	Takes a list of favourite movies indexes and number of movies in the dataset as inputs
	returns a user vector suitable for computing the recommendation
	vector's length is the same as the number of movies in the data set
	"""
	vector = [0]*num_of_all_movies
	for fav in favs:
		vector[fav] = 5.0
	user = [np.array(vector)]
	return user

def load_pickles(data=True, index_to_title=True, title_to_index=True, model=True):
	#load the matrix
	if data:
		with open("pickled_normalized_matrix.sav", "rb") as f:
			data = pickle.load(f)
	#load the index to title dict
	if index_to_title:
		with open("index_to_title_dict.pickle", "rb") as f2:
			index_to_title = pickle.load(f2)
	#load title to index dict
	if title_to_index:
		with open("title_to_index_dict.pickle", "rb") as f2:
			title_to_index = pickle.load(f2)
	#load trained model
	if model:
		with open("pickled_model.pickle", "rb") as f3:
			model = pickle.load(f3)
	return data, index_to_title, title_to_index, model



def get_similar_movies(movie_index, model, n=10):
	"""
	takes movie index, data matrix and int n as a input
	returns a list (of length n) of indexes of movies with most similar features  
	"""
	t = time.time()
	#columns = pd.DataFrame(data)
	#movie = columns[movie_index]
	columns = model.components_.T
	movie = columns[movie_index]
	scores = []
	"""
	substract the movie vector from each column vector, compute the square norm of the new vector 
	(the lower the norm, the more similar two vectors are)
	save the norm and the column index to a list
	sort the list and return n indexes of most similar movie vectors
	"""
	for i, column in enumerate(columns):
		score = np.subtract(column, movie)
		scores.append((np.dot(score, score), i))
	scores = sorted(scores)
	print(time.time()-t)
	return [i for score, i in scores[1:n+1]]


def rec_from_terminal():
	"""
	terminal program that interacts with the user
	takes indexes of favourite movies from a user and prints recomendations
	"""
	user_movies = []
	print("\n\nEnter the indexes of your favourite movies to get movie recomendations\nThe more titles you provide, the more accurate the predictions will be (min = 8)\n\n\n")

	go = True
	while go:
		movie = unicode(raw_input("So far you have added: "+str(len(user_movies))+" movies\nprovide movie index (type \"q\" if you think that's enough):"), 'utf-8')
		if movie == "q":
			go = False
		else:
			if movie.isnumeric():
				movie = int(movie)
				if movie >= 0 and movie <= 9065:
					user_movies.append(movie)
					print("\nadded: {}: {}\n".format(str(movie), index_to_title[movie]))
				else:
					print("ERROR: looks like it is not a movie from the list")
			else:
				print("ERROR: you need to provide the index number")

	if len(user_movies) >= 8:

		print(user_movies)
		user = user_vector(user_movies)

		rec = compure_rec(user, model, 15)
		favs = get_the_favs(user, len(user_movies))
		print_comarison(favs, rec, 10)

	else:
		print("\nSorry, the number of movies you have provided is too small for a recommendation")


def random_index(movies_num=9066):
	"""
	returns random movie index between 0 and num_of movies in data
	9066 set as default
	"""
	index = random.randint(0,movies_num-1)
	return index

favs = []
def binary_likes(index_to_title, model, index = -1):
	#check if the list is ready for computing recommendations
	if len(favs) == 8:
		user = user_vector(favs)
		recs = compure_rec(user, model)[0]
		print("\n\nYOU MIGHT LIKE THOSE MOVIES: \n")
		for rec in recs:
			print(index_to_title[rec])

	#if the list is not ready for recommendations
	else:

		if index == -1:
			index = random_index()

		like = raw_input("do you like {}".format(index_to_title[index]))

		if not like or like[0] != "y":
			binary_likes(index_to_title, model)

		elif like == "q":
			return 0

		elif like[0] == "y":
			favs.append(index)
			similars = get_similar_movies(index, data, n=5)
			random_5 = random.randint(0,4)
			similar = similars[random_5]
			binary_likes(index_to_title, model, index= similar)



#data, index_to_title, title_to_index, model = load_pickles()

#train_and_pickle("pickled_normalized_matrix.sav")




