from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    newX = tf.expand_dims(X, 1)
    newMU = tf.expand_dims(MU, 0)
    distance = tf.reduce_sum( tf.square(tf.subtract(newX, newMU)), 2)

    return distance

def log_GaussPDF(X, mu, sigma):
    mu_expanded = tf.expand_dims(mu, 1)
    #print mu_expanded.shape
    # convert this 2d tensor into 3D.                          
    X_expanded = tf.expand_dims(X,0)
    #print X_expanded.shape
    distance = (tf.math.subtract(X_expanded, mu_expanded))
    #distance = tf.reduce_sum(tf.square(distance), 2)
    print distance.shape
    #print distance.shape
    #print sigma.shape
    x_unpacked = tf.unstack(sigma) # defaults to axis 0, returns a list of tensors

    processed = [] # this will be the list of processed tensors
    for t in x_unpacked:
        D = tf.linalg.diag(tf.transpose([t,t]),name=None)
        print D.shape
        D1 = tf.reshape(D,[2,2])
        processed.append(D1)
    #print processed
    p = tf.convert_to_tensor(processed)

    #D = tf.linalg.diag(tf.transpose(sigma),name=None)
    #D1 = tf.reshape(D,[3,3])
    #print D1.shape
    m_dist_x = -tf.math.multiply((tf.math.multiply(tf.linalg.matmul((distance),tf.linalg.inv(p)),tf.transpose(distance,perm=[0,1,2]))),0.5)
    distance1 = tf.reduce_sum(m_dist_x,2)
    print distance1.shape
    matrix2 = tf.math.log(tf.math.multiply(1.0/(2.0*np.pi), tf.math.pow(tf.linalg.det(p),-0.5)))
    result = tf.transpose(distance1) + matrix2
    print result.shape
    return result
    #matrix1 = -(tf.math.divide((distance),sigma))
    #data = tf.linalg.tensor_diag_part(D1)
    #data1 = tf.sqrt(tf.math.reduce_prod(tf.square(sigma))) 
    #print m_dist_x.shape
    #matrix2 = tf.math.log(tf.math.multiply(tf.math.pow(2*3.14,1.5), data1))
    #matrix2 = tf.math.multiply(tf.math.log(tf.math.multiply(tf.math.pow(2*3.14,0.5), tf.linalg.det(D))),1/2)
    #print matrix2.shape
    #final = matrix1 - matrix2
    #print final.shape
    #return tf.transpose(final)
    #print tf.transpose(distance, perm=[0,1,2]).shape
    #print tf.reshape(distance,[1,30000,2])
    #print tf.linalg.inv(D).shape
    #m_dist_x = tf.linalg.matmul(tf.transpose(distance, perm=[0, 1, 2]),tf.linalg.inv(D))
    #m_dist_x = tf.math.multiply(m_dist_x, distance)
    #print m_dist_x.shape

    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    # TODO

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    return hlp.logsoftmax(tf.add(tf.transpose(log_pi),log_PDF))
    

def main():
    data = np.load('data2D.npy')
    [num_pts, dim] = np.shape(data)
    is_valid = False
    K = 5
    # For Validation set
    if is_valid:
        valid_batch = int(num_pts / 3.0)
        np.random.seed(45689)
        rnd_idx = np.arange(num_pts)
        np.random.shuffle(rnd_idx)
        val_data = data[rnd_idx[:valid_batch]]
        data = data[rnd_idx[valid_batch:]]

    X = tf.placeholder(tf.float32, [num_pts, dim], name='X')
    mu = tf.Variable(tf.truncated_normal([K, dim],stddev=0.05),name="mu")
    phi = tf.Variable(tf.zeros([K, 1]),name="sigma")
    sigma = tf.exp(phi)
    #clip_op = tf.assign(sigma, tf.clip_by_value(sigma, 0, np.infty))

    log_PDF = log_GaussPDF(X, mu,sigma)
    #temp_pi = np.float32(np.ones((K,1)))
    #temp_pi[:] = 0
    pi = tf.Variable(tf.truncated_normal([K, 1],stddev=0.05),name="pi")
    log_pi = hlp.logsoftmax(pi)
    log_Posterior = log_posterior(log_PDF, log_pi)
    logPX = tf.reduce_sum(hlp.reduce_logsumexp(tf.add(tf.transpose(log_pi),log_PDF)))
    print logPX.shape
    Loss = -logPX
    train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(Loss)
    sess = tf.Session()
    distances = distanceFunc(X, mu)
    nearestIndices = tf.argmax(log_Posterior, 1)
    partitions = tf.dynamic_partition(X,tf.to_int32(nearestIndices),K)
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(800):
            sess.run(train_op,feed_dict={X:np.float32(data)})
            #sess.run(clip_op)
            print sess.run(mu)
            print sess.run(tf.exp(log_pi))
            print(sess.run(Loss,feed_dict={X:np.float32(data)}))
        updated_centroid_value = sess.run(mu)
    part = sess.run(partitions,feed_dict={X:np.float32(data)})
    for data in part:
        print len(data)
    plt.figure()
    colour = plt.cm.rainbow(np.linspace(0,1,len(updated_centroid_value)))
    for i, centroid in enumerate(updated_centroid_value):
        print len(part[i])
        for j,point in enumerate(part[i]):

            plt.scatter(point[0], point[1], c=colour[i])
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k')
    plt.savefig( 'cluster5' + '.png')
    #plt.show()
    plt.figure()
    plt.scatter(data[:,0], data[:,1])
    plt.savefig('Originaldata' + '.png')



if __name__ == '__main__':
    main()