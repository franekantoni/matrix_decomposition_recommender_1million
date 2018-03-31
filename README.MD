RECOMMENDER SYSTEM PROJECT 1m ratings

®Franciszek Szombara

PProject uses files contain 1,000,209 anonymous ratings of approximately 3,900 movies 
made by 6,040 MovieLens users who joined MovieLens in 2000.


Program uses Pandas, NumPy and SKlearn libraries

to com

data_processing.py
	python program processing the data containded in files:
		1. ratings.csv
		2. movies.csv

	Program:
		1. Takes the data from csv files and turns it into a normalized, sprarse matrix:
		index='userId', columns='movieId'

		2. Creates two dictionaries: 
		index_to_title- a dict with column indexes as keys and titles as values
		title_to_index- a dict with titles as keys and column indexes as values

		3. Saves (pickles) the data matrix as pickled_normalized_matrix.sav
		Saves (pickles) the index_to_title dict as index_to_title_dict.pickle
		Saves (pickles) the title_to_index dict as title_to_index_dict.pickle

main.py
	python program containing functions for movie recommendations

	compure_rec(users, model, n):

		takes np.array of users, model and number of recomendations to be returned
		returns list of n recommendations


	get_the_favs(users, n):

		takes a np.array of users and number of favourite movies to be returned
		returns a list of n favourite movies for each user [lis of lists]


	print_comarison(lst_of_real_scores, lst_of_recs, n=10):

		takes a list of lists of favourite movies [->get_the_favs()] of users (or a single user)
		a list of lists of recommendations [->compure_rec()] of users (or a single user)
		and an int n
		for each user prints theirs n favourite movies and n best recommendations


	train_and_pickle(model = NMF(n_components=30, init='random', random_state=0, max_iter=300)):

		takes a NMF as a default model
		trains the model and pickles it for later
		returns a trained model


	catalogue():

		prints title, index pairs in alphabetical order
		creates a file containing title, index pairs


	user_vector(favs, num_of_all_movies = 9066):

		Takes a list of favourite movies indexes and number of movies in the dataset (default to the number in provided data set) as inputs 
		returns a user vector suitable for computing the recommendation
		vector's length is the same as the number of movies in the data set


	load_pickles():

		loads  and returns the data matrix, index to title dict, title to index dict and the trained model


	get_similar_movies(movie_index, data, n=10):

		takes movie index, data matrix and int n as a input
		returns a list (of length n) of indexes of movies with most similar features
		for each movie column substracts the movie vector from each column vector, compute the square norm of the new vector 
		(the lower the norm, the more similar two vectors are)
		saves the norm and the column index to a list
		sorts the list and return n indexes of most similar movie vectors


	rec_from_terminal():

		terminal program that interacts with the user
		takes indexes of favourite movies from a user and prints recomendations





