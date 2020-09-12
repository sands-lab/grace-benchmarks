#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: PTB-LSTM.py
# Author: Yuxin Wu

import argparse
import numpy as np
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils import gradproc, optimizer, summary
from tensorpack.utils import logger
from tensorpack.utils.argtools import memoized_ignoreargs
from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.train import HorovodTrainer
import wandb

import reader as tfreader
from reader import ptb_producer

rnn = tf.contrib.rnn

TRAIN_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt'
VALID_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt'
TEST_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt'


@memoized_ignoreargs
def get_PennTreeBank(data_dir=None):
    if data_dir is None:
        data_dir = get_dataset_path('ptb_data')
    if not os.path.isfile(os.path.join(data_dir, 'ptb.train.txt')):
        download(TRAIN_URL, data_dir)
        download(VALID_URL, data_dir)
        download(TEST_URL, data_dir)
    word_to_id = tfreader._build_vocab(os.path.join(data_dir, 'ptb.train.txt'))
    data3 = [np.asarray(tfreader._file_to_word_ids(os.path.join(data_dir, fname), word_to_id))
             for fname in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']]
    return data3, word_to_id


class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec((None, args.seq_len), tf.int32, 'input'),
                tf.TensorSpec((None, args.seq_len), tf.int32, 'nextinput')]

    def build_graph(self, input, nextinput):
        initializer = tf.random_uniform_initializer(-0.05, 0.05)

        def get_basic_cell():
            cell = rnn.BasicLSTMCell(num_units=args.hidden_size, forget_bias=0.0, reuse=tf.get_variable_scope().reuse)
            if self.training:
                cell = rnn.DropoutWrapper(cell, output_keep_prob=args.keep_prob)
            return cell

        cell = rnn.MultiRNNCell([get_basic_cell() for _ in range(args.num_layers)])

        def get_v(n):
            return tf.get_variable(n, [args.batch_size, args.hidden_size],
                                   trainable=False,
                                   initializer=tf.constant_initializer())

        state_var = [rnn.LSTMStateTuple(
            get_v('c{}'.format(k)), get_v('h{}'.format(k))) for k in range(args.num_layers)]
        self.state = state_var = tuple(state_var)

        embeddingW = tf.get_variable('embedding', [args.vocab_size, args.hidden_size], initializer=initializer)
        input_feature = tf.nn.embedding_lookup(embeddingW, input)  # B x seqlen x hiddensize
        input_feature = Dropout(input_feature, keep_prob=args.keep_prob)

        with tf.variable_scope('LSTM', initializer=initializer):
            input_list = tf.unstack(input_feature, num=args.seq_len, axis=1)  # seqlen x (Bxhidden)
            outputs, last_state = rnn.static_rnn(cell, input_list, state_var, scope='rnn')

        # update the hidden state after a rnn loop completes
        update_state_ops = []
        for k in range(args.num_layers):
            update_state_ops.extend([
                tf.assign(state_var[k].c, last_state[k].c),
                tf.assign(state_var[k].h, last_state[k].h)])

        # seqlen x (Bxrnnsize)
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.hidden_size])  # (Bxseqlen) x hidden
        logits = FullyConnected('fc', output, args.vocab_size,
                                activation=tf.identity, kernel_initializer=initializer,
                                bias_initializer=initializer)
        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.reshape(nextinput, [-1]))

        with tf.control_dependencies(update_state_ops):
            cost = tf.truediv(tf.reduce_sum(xent_loss),
                              tf.cast(args.batch_size, tf.float32), name='cost')  # log-perplexity

        perpl = tf.exp(cost / args.seq_len, name='perplexity')
        summary.add_moving_summary(perpl, cost)
        return cost

    def reset_lstm_state(self):
        s = self.state
        z = tf.zeros_like(s[0].c)
        ops = []
        for k in range(args.num_layers):
            ops.append(s[k].c.assign(z))
            ops.append(s[k].h.assign(z))
        return tf.group(*ops, name='reset_lstm_state')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=args.init_lr, trainable=False)
        opt = tf.train.GradientDescentOptimizer(lr)
        return optimizer.apply_grad_processors(
            opt, [gradproc.GlobalNormClip(5)])


#def lr_schedule(epoch, lr):
#    if epoch < args.warmup_epochs:
#        new_lr = args.init_lr * ((epoch+1)/args.warmup_epochs)
#    else:
#        new_lr = args.init_lr * ((1 - (epoch / float(args.epochs))) ** 2.0)
#    wandb.log({"learning_rate": new_lr, "epoch": epoch}, step=trainer.loop._global_step)
#    return new_lr


def get_config():
    data3, wd2id = get_PennTreeBank(args.datadir)
    args.vocab_size = len(wd2id)
    steps_per_epoch = (data3[0].shape[0] // args.batch_size - 1) // args.seq_len

    train_data = TensorInput(
        lambda: ptb_producer(data3[0], args.batch_size, args.seq_len),
        steps_per_epoch)
    val_data = TensorInput(
        lambda: ptb_producer(data3[1], args.batch_size, args.seq_len),
        (data3[1].shape[0] // args.batch_size - 1) // args.seq_len)

    test_data = TensorInput(
        lambda: ptb_producer(data3[2], args.batch_size, args.seq_len),
        (data3[2].shape[0] // args.batch_size - 1) // args.seq_len)

    M = Model()
    return TrainConfig(
        data=train_data,
        model=M,
        callbacks=[
            # ModelSaver(),
            HyperParamSetterWithFunc(
                'learning_rate',
#                lambda e, lr: lr_schedule(e, lr)),
#                lambda e, _: args.init_lr * (e/args.warmup_epochs) if e < args.warmup_epochs else args.init_lr * ((1 - (e / float(args.epochs))) ** 2.0)),
                lambda e, x: x * 0.8 if e > 6 else x),
            RunOp(lambda: M.reset_lstm_state()),
            InferenceRunner(val_data, [ScalarStats(['cost'])]),
            RunOp(lambda: M.reset_lstm_state()),
            InferenceRunner(
                test_data,
                [ScalarStats(['cost'], prefix='test')], tower_name='InferenceTowerTest'),
            RunOp(lambda: M.reset_lstm_state()),
            CallbackFactory(
                trigger=lambda self:
                [self.trainer.monitors.put_scalar(
                    'validation_perplexity',
                    np.exp(self.trainer.monitors.get_latest('validation_cost') / args.seq_len)),
                 self.trainer.monitors.put_scalar(
                     'test_perplexity',
                     np.exp(self.trainer.monitors.get_latest('test_cost') / args.seq_len))]
            ),
        ],
        max_epoch=args.epochs,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='the GPU to use')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--datadir', help='data set directory')
    parser.add_argument('--logdir', help='logging directory')
    parser.add_argument('--seq-len', type=int, help='sequence length', default=35)
    parser.add_argument('--hidden-size', type=int, default=650)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--keep-prob', type=float, default=0.5)
    parser.add_argument('--init-lr', type=float, default=1.0)
#    parser.add_argument('--warmup-epochs', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--vocab-size', type=int)
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    config = get_config()
    config.session_init = SmartInit(args.load)
    global trainer
    trainer = HorovodTrainer()
    # if not trainer.is_chief:
    #     os.environ['WANDB_MODE'] = 'dryrun'
    # elif not args.logdir:
    #     logger.auto_set_dir(action="d")
    # else:
    #     logger.set_logger_dir(args.logdir, action="d")

    launch_train_with_config(config, HorovodTrainer())
