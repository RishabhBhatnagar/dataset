from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
import pickle
from itertools import repeat
from pandas import read_csv
from numpy import array
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import array


if  __name__ == '__main__':	
	model = None
	docvecs = []
	vector_size = 100
	nn = 200
	data_size = 0
	target = None

	data = read_csv('../dataset/training.tsv', sep='\t', encoding='latin')
	essays = data['essay']
	target = data['rater1_domain1']+data['rater2_domain1']

	documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(essays)]
	model = Doc2Vec(documents, vector_size=vector_size, window=2, min_count=1, workers=5)
	train_data = [model.infer_vector(essay.split()) for essay in essays]

	'''
	print(target)
	# loading the saved object of model that stores the vectors of all the essays.
	with open('obj.pkl', 'rb') as pkl_file:
		model = pickle.load(pkl_file)
	
	# getting all the features from the model trained to vectorise the essays.
	for _, key in enumerate(model.wv.vocab):
		docvecs.append(model.wv[key])
		data_size += 1
	'''

	# Creating the neural network.
	nn_model = Sequential()

	# Defining all the layers.
	ip_layer = Dense(100, kernel_initializer='normal', input_dim=vector_size, activation='relu')
	op_layer = Dense(1, kernel_initializer='normal', activation='linear')
	hidden_layer = Dense(100, kernel_initializer='normal')#, activation='relu')

	# adding all the layers:
	nn_model.add(ip_layer)
	for _ in range(5): nn_model.add(hidden_layer)
	nn_model.add(op_layer)
	
	nn_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
	print(nn_model.summary())
	
	callback = []#[callbacks.EarlyStopping(patience=7)]
	logger = callbacks.TensorBoard(log_dir='./logs', write_graph=True, histogram_freq=1)
	nn_model.fit(array(train_data), array(target), epochs=5000, batch_size=1000, validation_split=0.2, callbacks=callback)
	try:
		model.save('final_model.h5')
	except:
		import pickle
		with open('final1.pkl', 'wb') as asdf:
			pickle.dump(model, asdf)
