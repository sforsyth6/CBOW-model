import tensorflow as tf
import numpy as np
import zipfile
import collections
import random
import math
filename = 'text8.zip'

def readfile(filename):
	with zipfile.ZipFile(filename) as f:
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

text = readfile(filename)

def build_dataset(text):
	count = [['UNK', -1]]
	count.extend(collections.Counter(text).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in text:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count = unk_count + 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

def word_embed_batch(data, embed_batch_size):	
	global data_index 
	num_skips = 2*window 
	assert embed_batch_size % num_skips == 0
        batch = np.ndarray(shape=(embed_batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(embed_batch_size, 1), dtype=np.int32)
       	size = num_skips + 1
        buffer = collections.deque(maxlen=size)
        for _ in range(size):
                buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data)
        for i in range(embed_batch_size // num_skips):
                target = window  # target label at the center of the buffer
                targets_to_avoid = [ window ]
                for j in range(num_skips):
                        while target in targets_to_avoid:
                                target = random.randint(0, size - 1)
                        targets_to_avoid.append(target)
                        batch[i * num_skips + j] = buffer[window]
                        labels[i * num_skips + j, 0] = buffer[target]
                data_index = (data_index + 1) % len(data)
        
        return batch, labels



vocabulary_size = 150000

alpha = 0.5

neg_sample = 15
num_embed = 128


window = 2
size = 2*window + 1
data_index = 0
embed_size = 64

valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))

data, count, dictionary, reverse_dictionary = build_dataset(text)
del text


graph = tf.Graph()

with graph.as_default():
	
	#weights for word embeddings
	w_embed = tf.Variable(tf.truncated_normal([vocabulary_size, num_embed], stddev=1.0 / math.sqrt(num_embed)))
	b_embed = tf.Variable(tf.zeros([vocabulary_size]))

	embed_batch = tf.placeholder(shape=[embed_size], dtype = tf.int32)
	embed_labels = tf.placeholder(shape=[embed_size,1], dtype = tf.int32)
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, num_embed], -1, 1))	

	embed = tf.nn.embedding_lookup(embeddings, embed_batch)


	embedding_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights = w_embed, biases = b_embed, 
					inputs = embed, labels = embed_labels, 
					num_sampled = neg_sample, num_classes = vocabulary_size))



	optimizer = tf.train.AdagradOptimizer(alpha).minimize(embedding_loss)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


	saver = tf.train.Saver()



with tf.Session(graph = graph) as session:
	tf.global_variables_initializer().run()

	num_steps = 1000001
	average = 0
	summary_frequency = 3
	for step in range(num_steps):
		feed = dict()

		#generate batch for word embeddings
		e_batch, e_label = word_embed_batch(data, embed_size)
		feed[embed_batch] = e_batch
		feed[embed_labels] = e_label

		_,l = session.run([optimizer, embedding_loss], feed_dict = feed)	

		average += l
                if step % 2500 == 0:
                        print (average / 2501.0, step)
                        average = 0

		if step % 10000 == 0:
			sim = similarity.eval()
			for i in range(valid_size):
				valid_word = reverse_dictionary[valid_examples[i]]
				top_k = 8 # number of nearest neighbors
				nearest = (-sim[i, :]).argsort()[1:top_k+1]
				log = 'Nearest to %s:' % valid_word
				for k in range(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log = '%s %s,' % (log, close_word)
				print(log)
	save_path = saver.save(session, '/tmp/model.ckpt')
	print ('Model saved in file: %s' %save_path)
