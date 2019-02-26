from pandas import DataFrame, read_csv
from os import path
from re import sub
# from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec


# Terminologies:
# 1. data carrying variables: Temporary variables that will store significant 
#                             results and will be returned in future.
# 
# Some basic description and file style guide:
# 1. Each function should define data carrying variables at start itself.
# 2. Temporary variables can be ambiguously defined anywhere.
# 


class Constants:
	# system wide file type constants:
	dataset_path = "../dataset"
	training_file_name = 'training.tsv'
	file_type='tsv'
	train_data_encoding = 'latin'

	# global constants
	file_ifs = dict(
		csv=', ',
		tsv='\t'
	)

	# processed constants:
	training_file_path = path.join(dataset_path, training_file_name)


def load_data(file_location, file_type='csv', encoding='utf8'):
	data = None

	# getting the ifs for the given dataset file.
	seperator = Constants.file_ifs.get(file_type)
	if seperator is None:
		raise NotImplementedError("file type not defined, use these {} \
				formats only.".format(list(Constants.file_ifs)))
	
	try:
		data = read_csv(file_location, sep=seperator, encoding=encoding)
	except FileNotFoundError:
		raise FileNotFoundError("given dataset file doesn't exist.")
	except UnicodeDecodeError:
		raise UnicodeDecodeError('Wrong encoding given')
	else:
		return data


def replace(data: DataFrame, replacee:str, replacer:str):
	re_replace = lambda word: sub(replacee, replacer, word)
	return data.apply(re_replace)


def main():
	ESSAY = 'essay'
	SCORE1 = 'rater2_domain1'
	SCORE2 = 'rater1_domain1'
	mSCORE = 'score'
	
	data = load_data(
		Constants.training_file_path, 
		Constants.file_type, 
		Constants.train_data_encoding 
	)
	
	# removing all the rows with atleast one attribute as NaN.
	data.dropna()
	
	# removing all the @ symbols in the dataset.
	data[ESSAY] = replace(data[ESSAY], "@", '')
	
	# defining final score to be average of rates of two rater.
	data[mSCORE] = data[SCORE1]+data[SCORE2]
	
	# waste try. build vocabulary before training.
	'''
	# Generating vectors for the given essays:
	model = Word2Vec(
		sentences=[i.split() for i in data [ESSAY]],
		workers=1,            # gives the number of threads used to train model
		min_count=1,          # min occurence of word below 
							  # which word will be ignored.
		window=50,             # max diff btwn curr word and predicted word.
		size=3               # output vector size 
	)
	'''
	tagged_essays = [TaggedDocument(essay.split(), essay) for idx, essay in enumerate(data[ESSAY])]
	tags = data[ESSAY]
	from time import time
	t1 = time()
	model = Doc2Vec(tagged_essays, vector_size=100, window=5, min_count=1, workers=5)
	print(time()-t1)
	with open("obj.pkl", 'wb') as f:
		import pickle
		pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	main()
