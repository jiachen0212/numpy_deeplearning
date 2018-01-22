#coding=utf-8
# https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d
import tensorflow as tf 
import numpy as np 

def depthwise_conv2d(
    input,
    filter,
    strides,
    rate=[1,1],
    padding=None
):
	ish = input.shape 
	fsh = filter.shape 
	output = np.zeros([ish[0],(ish[1]-fsh[0])//strides[1]+1,(ish[2]-fsh[1])//strides[2]+1,ish[3]*fsh[3]])
	osh = output.shape
	print(osh)
	for p in range(osh[0]):
		for i in range(osh[1]):
			for j in range(osh[2]):
				for k in range(fsh[2]):
					for di in range(fsh[0]):
						for dj in range(fsh[1]):
							t = np.dot(
									filter[di,dj,k,:],
									input[p,strides[1]*i+rate[0]*di,strides[2]*j+rate[1]*dj,k],
								)
							output[p,i,j,k*fsh[3]:(k+1)*fsh[3]] = np.sum(
									[
										t,
										output[p,i,j,k*fsh[3]:(k+1)*fsh[3]]
									],
									axis=0
								)
	return output
input_t = np.random.randint(10,size=(3,5,5,3))
filter_t = np.random.randint(10,size=(2,2,3,2))
strides_t = [1,1,1,1]
print('numpy depthwise_conv2d:')
res = depthwise_conv2d(input_t,filter_t,strides_t)
print(res)
print(res.shape)
print('tensorflow depthwise_conv2d')
a = tf.Variable(input_t,dtype=tf.float32)
b = tf.Variable(filter_t,dtype=tf.float32)
op = tf.nn.depthwise_conv2d(a,b,
	strides=strides_t,
	padding='VALID')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	res = sess.run(op)
	print(res)
	print(res.shape)

