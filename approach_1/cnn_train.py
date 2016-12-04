import tensorflow as tf
import cnn_functions as cf
from audio_dataset import audio_dataset


# Parameters of the session and network
LOG_STEP = 200    # Log accuracy every 200 steps
SAVER_STEP = 100  # Save checkpoint every 100 steps
BATCH_SIZE = 1    # Per pattern training

x = tf.placeholder(tf.float32, [None,262144,1,1])
y_ = tf.placeholder(tf.float32, [None,10])

# Layers of the network. The network is comprised of a series of convolutions
# and max-pool layers, and ends with two fully connected layers. A softmax
# function is used for predictions.

h1 = cf.conv_layer(x, [7,1,1,3])      # 262144 x 3
h2 = cf.pool_layer(h1)                # 131072 x 3
h3 = cf.conv_layer(h2, [5,1,3,5])     # 131072 x 5
h4 = cf.pool_layer(h3)                # 65535  x 5
h5 = cf.conv_layer(h4, [5,1,5,5])     # 65535  x 5
h6 = cf.pool_layer(h5)                # 32768  x 5
h7 = cf.conv_layer(h6, [3,1,5,5])     # 32768  x 5
h8 = cf.pool_layer(h7)                # 16384  x 5
h9 = cf.conv_layer(h8, [3,1,5,5])     # 16384  x 5
h10 = cf.pool_layer(h9)               # 8192   x 5
h11 = cf.conv_layer(h10, [3,1,5,5])   # 8192   x 5
h12 = cf.pool_layer(h11)              # 4096   x 5
h13 = cf.conv_layer(h12, [3,1,5,1])   # 4096   x 1
h14 = cf.pool_layer(h13)              # 2048   x 1

hf = tf.reshape(h14, [-1, 2048])

# The fully connected layers
fc1 = cf.fc_layer(hf, [2048,100])
fc2 = cf.fc_layer(fc1, [100,10])

# The output is passed through a softmax function so it represents probabilities
y = tf.nn.softmax(fc2)

# The loss/energy function is the cross-entropy between the label and the output
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Adaptive Momentum Backpropogation
train_step = tf.train.AdamOptimizer(3e-4).minimize(loss)

# Classification accuracy is a better indicator of performance
correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

dataset = audio_dataset()
sess = tf.Session()
sess.run(tf.initialize_all_variables())

log = open('cnn/logs/log.txt','a')

saver = tf.train.Saver()
checkpoint = 0



with sess.as_default():
  for step in range(1+int(2e5)): 
    batch, labels = dataset.get_batch_train(BATCH_SIZE)

    # Print a message for every 50 steps
    if (step % 50 == 0): 
      print('step {}'.format(step))

    # Update the log with the newest performance results
    if (step%LOG_STEP==0):
      # Calculate acccuracy for the current batch.
      train_y = y.eval(feed_dict={x:batch})
      train_loss = loss.eval(feed_dict={y:train_y, y_:labels})
      train_acc = accuracy.eval(feed_dict={y:train_y, y_:labels})

      # Run through the entire validation set; accuracy is the mean of results.
      valid_loss = 0
      valid_acc = 0
      batch_count = 0

      while True:
        val_x, val_y_ = dataset.get_batch_valid(BATCH_SIZE)
        if val_x == None:
          break
        batch_count +=1
        
        val_y = y.eval(feed_dict={x:val_x})
        valid_loss += loss.eval(feed_dict={y:val_y, y_:val_y_})
        valid_acc += accuracy.eval(feed_dict={y:val_y, y_:val_y_})
        
      valid_loss = valid_loss/batch_count
      valid_acc = valid_acc/batch_count

      logline = 'Epoch {} Batch {} train_loss {} train_acc {} valid_loss {} valid_acc {} \n'
      logline = logline.format(dataset.completed_epochs, step, train_loss, train_acc, valid_loss, valid_acc)
      log.write(logline)
      print(logline)

    if step%SAVER_STEP==0:
      path = saver.save(sess, 'cnn/checkpoints/cnn_', global_step=checkpoint)
      print("Saved checkpoint to {}".format(path))
      checkpoint += 1
      
    train_step.run(feed_dict={x:batch, y_:labels})