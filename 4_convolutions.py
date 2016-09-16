# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves import cPickle as pickle

# Input configuration
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

# Model configuration
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

# Training configuration
starter_learning_rate = 0.1
decay_every = 2000
decay_rate = 0.96

# dataset file name
pickle_file = 'notMNIST.pickle'


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) 
            / predictions.shape[0])

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels


if __name__ == "__main__":

    # unpickle the data
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        
        del save  # hint to help gc free up memory
        
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)


    # reformat the dataset
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


    graph = tf.Graph()
    with graph.as_default():

        # Training step (for learning rate decay)
        global_step = tf.Variable(0, trainable=False)

        # Input data
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
      
        # Variables
        layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
      
        # Model
        def model(data):
            # hidden layer 1
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(pool + layer1_biases)
            hidden_drop = tf.nn.dropout(hidden, 0.7)
            
            # hidden layer 2
            conv = tf.nn.conv2d(hidden_drop, layer2_weights, [1, 1, 1, 1], padding='SAME')
            pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(pool + layer2_biases)
            hidden_drop = tf.nn.dropout(hidden, 0.7)
            
            # fully connected MLP on top
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden_drop, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            
            return tf.matmul(hidden, layer4_weights) + layer4_biases
      
        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        learning_rate = tf.train.exponential_decay(starter_learning_rate, 
                                           global_step,
                                           decay_every, 
                                           decay_rate, 
                                           staircase=True)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
        learning_step = (optimizer)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))


    num_steps = 1001

    with tf.Session(graph=graph) as session:
        
        tf.initialize_all_variables().run()
        print('Initialized')
        
        for step in range(num_steps):
            
            # compute the offset of the data
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            
            # get batches of data
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            
            # feeding dictionary
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            
            # Execute one run through the graph
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            # Print some feedback every 50 iterations
            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
            
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))