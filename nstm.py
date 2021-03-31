import math

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import os
import time
from scipy import sparse


base_dir = "./"
sys.path.insert(1, base_dir)

import utils
from auto_diff_sinkhorn import sinkhorn_tf



from utils import load_data, batch_indices, print_topics, set_logger, save_flags, get_doc_topic

flags = tf.app.flags
flags.DEFINE_float('sh_epsilon', 0.001, 'sinkhorn epsilon')
flags.DEFINE_integer('sh_iterations', 50, 'sinkhorn iterations')
flags.DEFINE_string('dataset', 'TMN', 'dataset')
flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
flags.DEFINE_integer('batch_size', 200, 'batch_size')
flags.DEFINE_integer('K', 100, 'num topics')
flags.DEFINE_integer('random_seed', 1, 'random_seed')
flags.DEFINE_integer('n_epochs', 50, 'n_epochs')
flags.DEFINE_float('rec_loss_weight', 0.07, 'rec_loss_weight')
flags.DEFINE_float('sh_alpha', 20, 'sh_alpha')
FLAGS = flags.FLAGS

def run_ntsm(args):


    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    save_dir = os.path.join(base_dir, 'save', 'dataset%s_K%d_RW%0.3f_RS%d_L%0.3f' %
                            (FLAGS.dataset, FLAGS.K, FLAGS.rec_loss_weight, FLAGS.random_seed, FLAGS.sh_alpha))

    os.makedirs(save_dir, exist_ok=True)

    save_flags(save_dir)

    logger = set_logger(save_dir)

    data_dir = os.path.join(base_dir, 'datasets')
    data_dir = '%s/%s' % (data_dir, FLAGS.dataset)

    train_data, test_data, word_embeddings, voc = load_data('%s/data.mat' % data_dir, True)

    L = word_embeddings.shape[1]

    V = train_data.shape[1]
    N = train_data.shape[0]

    doc_word_ph = tf.placeholder(dtype=tf.float32, shape=[None, V])

    doc_word_tf = tf.nn.softmax(doc_word_ph)

    with tf.variable_scope('encoder'):

        doc_topic_tf = utils.mlp(doc_word_ph, [200], utils.myrelu)
        doc_topic_tf = tf.nn.dropout(doc_topic_tf, 0.75)
        doc_topic_tf = tf.contrib.layers.batch_norm(utils.linear(doc_topic_tf, FLAGS.K, scope='mean'))
        doc_topic_tf = tf.nn.softmax(doc_topic_tf)

    with tf.variable_scope('cost_function'):

        topic_embeddings_tf = tf.get_variable(name='topic_embeddings', shape=[FLAGS.K, L],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, seed=FLAGS.random_seed))
        word_embeddings_ph = tf.placeholder(dtype=tf.float32, shape=[V, L])

        topic_embedding_norm = tf.nn.l2_normalize(topic_embeddings_tf, dim=1)
        word_embedding_norm = tf.nn.l2_normalize(word_embeddings_ph, dim=1)
        topic_word_tf = tf.matmul(topic_embedding_norm, tf.transpose(word_embedding_norm))
        M = 1 - topic_word_tf


    sh_loss = sinkhorn_tf(M, tf.transpose(doc_topic_tf), tf.transpose(doc_word_tf), lambda_sh = FLAGS.sh_alpha)

    sh_loss = tf.reduce_mean(sh_loss)

    rec_log_probs = tf.nn.log_softmax(tf.matmul(doc_topic_tf, topic_word_tf))
    rec_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(rec_log_probs, doc_word_ph), 1))

    fullvars = tf.trainable_variables()
    enc_vars = utils.variable_parser(fullvars, 'encoder')
    cost_function_vars = utils.variable_parser(fullvars, 'cost_function')

    rec_loss_weight = tf.placeholder(tf.float32, ())

    joint_loss = rec_loss_weight * rec_loss + sh_loss
    trainer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(joint_loss, var_list=[enc_vars + cost_function_vars])

    saver = tf.train.Saver()

    is_stop = False

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    with session as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        nb_batches = int(math.ceil(float(N) / FLAGS.batch_size))
        assert nb_batches * FLAGS.batch_size >= N

        rec_losses = []

        sh_losses = []

        joint_losses = []

        running_times = []

        for epoch in range(FLAGS.n_epochs):

            logger.info('epoch: %d' % epoch)

            idxlist = np.random.permutation(N)

            rec_loss_avg, sh_loss_avg, joint_loss_avg = 0., 0., 0.

            for batch in tqdm(range(nb_batches)):

                start, end = batch_indices(batch, N, FLAGS.batch_size)
                X = train_data[idxlist[start:end]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                batch_start_time = time.time()
                _, rec_loss_batch, sh_rec_loss_batch, joint_loss_batch = \
                    sess.run([trainer, rec_loss, sh_loss, joint_loss],
                             feed_dict={doc_word_ph: X, word_embeddings_ph: word_embeddings, rec_loss_weight: FLAGS.rec_loss_weight})
                running_times.append(time.time() - batch_start_time)
                if np.isnan(joint_loss_batch):
                    is_stop = True

                rec_loss_avg += rec_loss_batch
                sh_loss_avg += sh_rec_loss_batch
                joint_loss_avg += joint_loss_batch

                rec_losses.append(rec_loss_batch)
                sh_losses.append(sh_rec_loss_batch)
                joint_losses.append(joint_loss_batch)

                
            logger.info('joint_loss: %f' % (joint_loss_avg / nb_batches))

            if is_stop:
                logger.info('early stop because of NaN at epoch %d' % epoch)
                break



        [topic_embeddings, topic_word_mat] = sess.run([topic_embeddings_tf, topic_word_tf], feed_dict={word_embeddings_ph: word_embeddings})

        train_doc_topic = get_doc_topic(sess, doc_topic_tf, doc_word_ph, train_data, FLAGS.K)

        test_doc_topic = get_doc_topic(sess, doc_topic_tf, doc_word_ph, test_data, FLAGS.K)

        vis_file = open(os.path.join(save_dir, 'vis.txt'), 'a')
        print_topics(topic_word_mat, voc, printer=vis_file.write)

        import scipy.io
        scipy.io.savemat(os.path.join(save_dir, 'save.mat'), {'phi': topic_word_mat,
                                                                'train_theta': train_doc_topic,
                                                                'test_theta': test_doc_topic,
                                                                'topic_embeddings': topic_embeddings,
                                                                'joint_losses': joint_losses})



if __name__ == '__main__':

    tf.app.run(run_ntsm)
