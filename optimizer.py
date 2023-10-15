import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):

    """
    Optimizer for GAE
    """

    def __init__(self, preds, labels, degree_matrix, num_nodes, pos_weight, norm, clusters_distance):

        preds_sub = preds
        labels_sub = labels

        # Reconstruction term (as in tkipf/gae)
        self.cost_adj =  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                                                 labels = labels_sub,
                                                                                 pos_weight = pos_weight))

        # Modularity-inspired term
        if FLAGS.simple: # simpler proxy of the modularity
            self.cost_mod = (1.0 / num_nodes) * tf.reduce_sum(labels_sub * clusters_distance)
        else:
            self.cost_mod = (1.0 / num_nodes) * tf.reduce_sum((labels_sub - degree_matrix) * clusters_distance)
        self.cost = self.cost_adj - FLAGS.beta * self.cost_mod

        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):

    """
    Optimizer for VGAE
    """

    def __init__(self, preds, labels, degree_matrix, model, num_nodes, pos_weight, norm, clusters_distance):

        preds_sub = preds
        labels_sub = labels
        self.labels_sub = labels
        self.clusters_distance = clusters_distance

        # ELBO term (as in tkipf/gae)
        self.cost_adj = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = tf.clip_by_value(preds_sub,1e-10,1.0),
                                                                                       labels = labels_sub,
                                                                                       pos_weight = pos_weight))
        print("print cost_adj:")
        print(self.cost_adj)
        self.log_lik = self.cost_adj
        self.kl = (0.5 / num_nodes) * \
                  tf.reduce_mean(tf.reduce_sum(1 \
                                               + 2 * model.z_log_std \
                                               - tf.square(model.z_mean) \
                                               - tf.square(tf.exp(model.z_log_std)), 1))
        self.cost_adj -= self.kl

        # Modularity-inspired term
        if FLAGS.simple: # simpler proxy of the modularity
            self.cost_mod = (1.0 / num_nodes) * tf.reduce_sum(labels_sub * clusters_distance)
            self.mod_Q = (1.0 / num_nodes) * tf.reduce_sum(labels_sub)
        else:
            self.cost_mod = (1.0 / num_nodes) * tf.reduce_sum((labels_sub - degree_matrix) * clusters_distance)
            self.mod_Q = (1.0 / num_nodes) * tf.reduce_sum(labels_sub - degree_matrix)
            #self.redu_sum=tf.reduce_sum(labels_sub - degree_matrix)
            #self.numnodes=num_nodes
        # Note: here, self.cost_adj corresponds to -ELBO. By minimizing (-ELBO-FLAGS.beta*self.cost_mod),
        # we actually maximize (ELBO+FLAGS.beta*self.cost_mod) as in the paper
        self.cost = self.cost_adj - FLAGS.beta * self.cost_mod
        print("print self.cost_mod:")
        print(self.cost_mod)
        print("print cost:")
        print(self.cost)

        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
