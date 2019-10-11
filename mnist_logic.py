import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

print("package loaded")

mnist=input_data.read_data_sets('mnist_data/',one_hot=True)

print(" type of mnist is %s" % (type(mnist)))
print(" number of trian data is %d" % (mnist.train.num_examples))
print(" number of test data is %d" % (mnist.test.num_examples))

#what does teh data of mnist look like
trainimg    =mnist.train.images
trainlabel  =mnist.train.labels
testimg     =mnist.test.images
testlabel   =mnist.test.labels

print (trainlabel[0]) #the apperance of label:[1,0,0,0,0,0,0,0,0,0]

x=tf.placeholder("float",[None,784])
y=tf.placeholder("float",[None,10])#None is for infinite
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
#Logistic regression model
actv=tf.nn.softmax(tf.matmul(x,W)+b)
#cost function
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv),reduction_indices=1))
#optimizer
learning_rate=0.01
optm=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#prediction
#argmax(matrix,type) find the largest addr in type;type=0->column,type=1->row
pred=tf.equal(tf.argmax(actv,1),tf.argmax(y,1))
#accutacy
accr=tf.reduce_mean(tf.cast(pred,"float"))
#initializer
init=tf.global_variables_initializer()


#constants
training_epochs=10
batch_size=100
display_step=5

#session
sess=tf.Session()
sess.run(init)

#Mini-batch learning
for epoch in range(training_epochs):
    avg_cost=0.
    num_batch =int(mnist.train.num_examples/batch_size)
    for i in range(num_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(optm,feed_dict={x: batch_xs, y: batch_ys})
        feeds={x: batch_xs, y: batch_ys}
        avg_cost+=sess.run(cost, feed_dict=feeds)/num_batch

    #display
    if epoch % display_step==0:
        feeds_train={x: batch_xs, y: batch_ys}
        feeds_test={x:mnist.test.images,y:mnist.test.labels}
        train_acc=sess.run(accr,feed_dict=feeds_train)
        test_acc=sess.run(accr,feed_dict=feeds_test)
        print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
        % (epoch,training_epochs,avg_cost,train_acc,test_acc))
print("DONE!")

print("real pic")
nsample=10
randidx=np.random.randint(trainimg.shape[0],size=nsample)
for i in randidx:
    curr_img_temp=[trainimg[i,:]]
    curr_img=np.reshape(trainimg[i,:],(28,28))  #28 by 28 matrix
    curr_label=[np.argmax(trainlabel[i,:])]

    predition=sess.run(actv,feed_dict={x:curr_img_temp})
    predition_temp=predition[0,:]
    predition_rate=max(predition_temp)
    predition_real=np.argmax(predition_temp)
    #predition_real=tf.argmax(predition,1).eval



    plt.matshow(curr_img,cmap=plt.get_cmap('gray'))
    plt.title(  ""+"label is "+str(curr_label)+" predition is: "+str(predition_real)+" rate is: "+str(predition_rate))
    print(  ""+str(i)+"th training data"
            +"label is "+str(curr_label))
    plt.show()