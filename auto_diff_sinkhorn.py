import tensorflow as tf
import numpy as np


def sinkhorn_tf(M, a, b, lambda_sh, numItermax=1000, stopThr=.5e-2):


    u = tf.ones(shape=(tf.shape(a)[0], tf.shape(a)[1]), dtype=tf.float32) / tf.cast(tf.shape(a)[0], tf.float32)
    v = tf.zeros_like(b, dtype=tf.float32)
    K = tf.exp(-M * lambda_sh)

    cpt = tf.constant(0, dtype=tf.float32)
    err = tf.constant(1.0, dtype=tf.float32)

    c = lambda cpt, u, v, err: tf.logical_and(cpt < numItermax, err > stopThr)

    def v_update(u_, v_):
        v_ = tf.divide(b, tf.matmul(tf.transpose(K), u_))  
        u_ = tf.divide(a, tf.matmul(K, v_)) 
        return u_, v_

    def no_v_update(u_, v_):
        return u_, v_

    def err_f1(K_, u_, v_, b_):
        bb = tf.multiply(v_, tf.matmul(tf.transpose(K_), u_))
        err_ = tf.norm(tf.reduce_sum(tf.abs(bb - b_), axis=0), ord=np.inf)
        return err_

    def err_f2(err_):
        return err_

    def loop_func(cpt_, u_, v_, err_):
        u_ = tf.divide(a, tf.matmul(K, tf.divide(b, tf.transpose(tf.matmul(tf.transpose(u_), K)))))
        cpt_ = tf.add(cpt_, 1)
        u_, v_ = tf.cond(tf.logical_or(tf.equal(cpt_ % 20, 1), tf.equal(cpt, numItermax)), lambda: v_update(u_, v_), lambda: no_v_update(u_, v_))
        err_ = tf.cond(tf.logical_or(tf.equal(cpt_ % 20, 1), tf.equal(cpt, numItermax)), lambda: err_f1(K, u_, v_, b), lambda: err_f2(err_))
        return cpt_, u_, v_, err_

    cpt, u, v, err = tf.while_loop(c, loop_func, loop_vars=[cpt, u, v, err])

    sinkhorn_divergences = tf.reduce_sum(tf.multiply(u, tf.matmul(tf.multiply(K, M), v)), axis=0)
    return sinkhorn_divergences
