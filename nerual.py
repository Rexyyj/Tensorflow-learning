import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data

print("package loaded")

mnist=input_data.read_data_sets('mnist_data/',one_hot=True)

n_hidden_1 =256
n_hidden_2 =128
n_input    =784
n_classes  =10

#n inputs and outputs
x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_classes])

#network parameters

stddev=0.1
weights={
    'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
    'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
}
biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes])),
}


print("Network ready")

##star to define the network function
def multilayer_perceptron(_X,_weights,_biases):
    layer1=tf.nn.sigmoid(tf.add(tf.matmul(_X,_weights['w1']),_biases['b1']))
    layer2=tf.nn.sigmoid(tf.add(tf.matmul(layer1,_weights['w2']),_biases['b2']))
    return (tf.matmul(layer2,_weights['out'])+_biases['out'])

#predition
pred=multilayer_perceptron(x,weights,biases)

#loss and optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
optm=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
corr=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accr=tf.reduce_mean(tf.cast(corr,"float"))

#initializer
init=tf.global_variables_initializer()
print("function ready")

##start training
training_epochs=100
batch_size=100
display_step=4
#lunch the graph
sess=tf.Session()
sess.run(init)
#optimize
for epoch in range(training_epochs):
    avg_cost=0.
    total_batch =int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        feeds={x: batch_xs, y: batch_ys}
        sess.run(optm,feed_dict=feeds)
        avg_cost+=sess.run(cost, feed_dict=feeds)
    avg_cost=avg_cost/total_batch
    #display
    if (epoch+1) % display_step==0:
        print("epoch:%03d/%03d cost:%.9f" % (epoch,training_epochs,avg_cost))
        feeds={x: batch_xs, y: batch_ys}
        train_acc=sess.run(accr,feed_dict=feeds)
        print("train accuracy: %.3f" % (train_acc))
        feeds={x:mnist.test.images,y:mnist.test.labels}
        test_acc=sess.run(accr,feed_dict=feeds)
        print("test accuracy: %.3f" % (test_acc))
print("optimization finished")

