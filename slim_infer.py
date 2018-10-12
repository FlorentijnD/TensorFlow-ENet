import tensorflow as tf
from enet import ENet, ENet_arg_scope
slim = tf.contrib.slim

def main():

    graph = tf.Graph()
    with graph.as_default():

        with slim.arg_scope(ENet_arg_scope()):
            inputs = tf.placeholder(tf.float32, [None, 360, 480, 3], name="input") 
            logits, probabilities = ENet(inputs,
                                         12,
                                         batch_size=1,
                                         is_training=False,
                                         reuse=None,
                                         num_initial_blocks=1,
                                         stage_two_repeat=2,
                                         skip_connections=False)

        saver = tf.train.Saver(tf.global_variables())
        sess  = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, './checkpoint/model.ckpt-13800')
        saver.save(sess, './checkpoint/modelfinal.ckpt-13800')

        graphdef = graph.as_graph_def()
        tf.train.write_graph(graphdef, './checkpoint', 'semanticsegmentation_enet.pbtxt', as_text=True)

if __name__ == '__main__':
    main()
