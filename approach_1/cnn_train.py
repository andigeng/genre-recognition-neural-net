import tensorflow as tf
import cnn_functions as cf
from audio_dataset import audio_dataset

dataset = audio_dataset()
wave,l,bs = dataset.next_batch_valid(10)
print wave.shape

#Parameters of the loop
LOG_STEP = 200
SAVER_STEP = 100

# Hyper-parameters of the network
BATCH_SIZE = 1

x = tf.placeholder(tf.float32, [1,129,1170,1])
y_ = tf.placeholder(tf.float32, [1, 10])


h1 = cf.conv_layer(x, [3,3,1,4]) # feature map = 129 x 1170 x 4
print("\n {} \n".format(h1._shape))

h2 = cf.pool_layer(h1) # size = 65 x 585 x 4
print("\n {} \n".format(h2._shape))

h3 = cf.pool_layer(h2) # 33 x 293 x 4
print("\n {} \n".format(h3._shape))

h4 = cf.conv_layer(h3, [3,3,4,1]) # 33 x 293 x 1
print("\n {} \n".format(h4._shape))

h5 = cf.pool_layer(h4) #17 x 147 x 1
print("\n {} \n".format(h5._shape))

h6 = cf.pool_layer(h5) # 9 x 74 x 1
print("\n {} \n".format(h6._shape))

h7 = cf.pool_layer(h6) # 5 x 37 x 1
print("\n {} \n".format(h7._shape))

h8 = cf.pool_layer(h7) # 3 x 19 x 1
print("\n {} \n".format(h8._shape))

h9 = cf.pool_layer(h8) # 2 x 10 x 1
print("\n {} \n".format(h9._shape))

h10 = cf.pool_layer(h9) # 1 x 5 x 1

print("\n {} \n".format(h10._shape))

hf = tf.reshape(h10, [-1, 5])
print("\n {} \n".format(hf._shape))

fc1 = cf.fc_layer(hf, [5, 10])
print("\n {} \n".format(fc1._shape))

# We pass the output through softmax so it represents probabilities
y = tf.nn.softmax(fc1)

# Our loss/energy function is the cross-entropy between the label and the output
# We chose this as it offers better results for classification
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# We are using the Adam Optimiser because it is effective at managing the learning rate and momentum
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# Classification accuracy is a better indicator of performance
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

log = open('cnn/logs/log.txt','a')

saver = tf.train.Saver()
checkpoint = 0



with sess.as_default():
  for s in range(1+int(2e1)): 
    waves, labels, bs = dataset.next_batch_train(BATCH_SIZE)
    print('step {}'.format(s))

    # We update the log with the newest performance results
    if (s%LOG_STEP==0):
      # We calculate the performance results
      # for the training set on the current batch
      tr_y = y.eval(feed_dict={x:waves})
      train_loss = loss.eval(feed_dict={y:tr_y, y_:labels})
      train_acc = accuracy.eval(feed_dict={y:tr_y, y_:labels})

      # For the validation set, we do it on the whole thing
      # The final results are means of the results for each batch
      valid_loss = 0
      valid_acc = 0
      batch_count = 0
      while True:
        va_x, va_y_, bs = dataset.next_batch_valid(BATCH_SIZE)
        if bs == -1:
          break
        batch_count +=1
        
        va_y = y.eval(feed_dict={x:va_x})
        valid_loss += loss.eval(feed_dict={y:va_y, y_:va_y_})
        valid_acc += accuracy.eval(feed_dict={y:va_y, y_:va_y_})
        
      valid_loss = valid_loss/batch_count
      valid_acc = valid_acc/batch_count

      logline = 'Epoch {} Batch {} train_loss {} train_acc {} valid_loss {} valid_acc {} \n'
      logline = logline.format(dataset.completed_epochs, s, train_loss, train_acc, valid_loss, valid_acc)
      log.write(logline)
      print logline

    if s%SAVER_STEP==0:
      path = saver.save(sess, 'cnn/checkpoints/cnn_',global_step=checkpoint)
      print "Saved checkpoint to %s" % path
      checkpoint += 1
      
    train_step.run(feed_dict={x:waves, y_:labels})

"""
h2 = cf.conv_layer(h1, [3,3,4,4])   # feature map = 129 x 1170 x 4
p1 = cf.pool_layer(h2)              # output      = 65 x 585 x 4
h3 = cf.conv_layer(p1, [3,3,4,8])   # feature map = 65 x 585 x 8
p2 = cf.pool_layer(h3)              # output      = 33 x 293 x 8
h4 = cf.conv_layer(p2, [3,3,8,16])  # feature map = 33 x 292 x 16
p3 = cf.pool_layer(h4)              # output      = 17 x 146 x 16
h5 = cf.conv_layer(p3, [3,3,16,32]) # feature map = 17 x 146 x 32
p4 = cf.pool_layer(h5)              # output      = 9 x 73 x 32
h6 = cf.conv_layer(p4, [3,3,32,64]) # feature map = 9 x 73 x 64

h1 = cf.cnm2x1Layer(x, [3,3,1,5])   # feature map = 129 x 1170 x 5
                                    # output      = 
h2 = cf.cnm2x1Layer(h1, [])

h1 = cf.cnm2x1Layer(x, [7,1,1,3]) # size=131072x3
h3 = cf.cnm2x1Layer(h1, [5,1,3,5]) # size=65536x5
h4 = cf.cnm2x1Layer(h3, [5,1,5,5]) # size=32768x5
h5 = cf.cnm2x1Layer(h4, [3,1,5,5]) # size=16384x5
h6 = cf.cnm2x1Layer(h5, [3,1,5,5]) # size=81925x5
h7 = cf.cnm2x1Layer(h6, [3,1,5,5]) # size=4096x5
h8 = cf.cnm2x1Layer(h7, [3,1,5,1]) # size=2048x1

hf = tf.reshape(h8, [-1, 2048])

fc1 = cf.fc_nn(hf,[2048,100])
fc2 = cf.fc_nn(fc1,[100,10])
"""