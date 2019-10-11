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

print(" type of trainimg is %s" % (type(trainimg)))
print(" type of trainlabel is %s" % (type(trainlabel)))
print(" type of testimg is %s" % (type(testimg)))
print(" type of testlabel is %s" % (type(testlabel)))
print(" shape of trainimg is %s" %(trainimg.shape,))
print(" shape of trainlabel is %s" %(trainlabel.shape,))
print(" shape of testimg is %s" %(testimg.shape,))
print(" shape of testlabel is %s" %(testlabel.shape,))

print("real pic")
nsample=5
randidx=np.random.randint(trainimg.shape[0],size=nsample)

for i in randidx:
    curr_img=np.reshape(trainimg[i,:],(28,28))  #28 by 28 matrix
    curr_label=np.argmax(trainlabel[i,:])
    plt.matshow(curr_img,cmap=plt.get_cmap('gray'))
    plt.title(  ""+str(i)+"th training data"
                +"label is "+str(curr_label))
    print(  ""+str(i)+"th training data"
            +"label is "+str(curr_label))
    plt.show()