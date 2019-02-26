import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2


x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

#weight
w = tf.Variable(tf.zeros([784, 10]))
#bias
b = tf.Variable(tf.zeros([10]))


with tf.name_scope("Wx_b") as scope:
	#construct linear model
	model = tf.nn.softmax(tf.matmul(x, w) + b)


# add summary operations to help visualize weights and biases later on
w_h = tf.summary.histogram("weights", w)
b_h = tf.summary.histogram("biases", w)


#to minimize error while training, another scope

with tf.name_scope("cost_function") as scope:
	#this is cross entropy
	cost_function = -tf.reduce_sum(y*tf.log(model))
	#create a summary to monitor cost function and visualize it later
	tf.summary.scalar("cost_function", cost_function)

#creates optimization function that improves model while training
with tf.name_scope("train") as scope:
	#gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

#initialize the variables
init = tf.initialize_all_variables()

#merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all	()

#launch the graph
with tf.Session() as sess:
	sess.run(init)

	#set the logs writer to the folder
	summary_writer = tf.summary.FileWriter('data/logs', graph_def=sess.graph_def)

	#now train the model
	for iteration in range(training_iteration):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples//batch_size)

		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			#fit training using batch data
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			#compute average loss
			avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})//total_batch

			#write logs for each iteration
			summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
			summary_writer.add_summary(summary_str, iteration*total_batch+i)
		if iteration % display_step == 0:
			print("Iteration:", '%04d' % (iteration+1), "cost=", "{:.9f}".format(avg_cost))

	print("training complete!")

	#test the model
	predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	#calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
	print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))





