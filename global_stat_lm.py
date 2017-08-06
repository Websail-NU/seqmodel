import sys
import os
from collections import ChainMap
from functools import partial

import tensorflow as tf
import kenlm

sys.path.insert(0, '../')

import seqmodel as sq  # noqa


def ngram_stat_data():
    ngram_path = partial(os.path.join, '../experiment/lm/ngram_lm/')
    f_lm = kenlm.Model(ngram_path('train_bigram.arpa'))
    p_lm = kenlm.Model(ngram_path('current_bigram.arpa'))
    ngram_set = sq.get_union_ngram_set([ngram_path('train_bigram.count'),
                                        ngram_path('current_bigram.count')])
    vocab = sq.Vocabulary.from_vocab_file('data/ptb/vocab.txt')
    CU, C = sq.compute_ngram_constraints(ngram_set, f_lm, p_lm, vocab)

    def eps_feed(mode, features, labels, **kwargs):
        inputs = features.inputs
        return sq.get_sparse_scalers(inputs, C, max_order=1)

    return eps_feed


def load_data(opt):
    dpath = partial(os.path.join, opt['data_dir'])
    vocab = sq.Vocabulary.from_vocab_file(dpath('vocab.txt'))
    data_fn = partial(sq.read_seq_data, in_vocab=vocab, out_vocab=vocab,
                      keep_sentence=opt['sentence_level'], seq_len=opt['seq_len'])
    data = [data_fn(sq.read_lines(dpath(f), token_split=' '))
            for f in (opt['train_file'], opt['valid_file'], opt['eval_file'])]

    batch_iter = partial(sq.seq_batch_iter, batch_size=opt['batch_size'],
                         shuffle=opt['sentence_level'],
                         keep_sentence=opt['sentence_level'])
    return data, batch_iter, (vocab, vocab)


def begin_epoch():
    pass


def main(opt, model_opt, data_fn, run_fn, logger, train_opt=None):

    logger.info('Loading data...')
    data, batch_iter, vocabs = data_fn()
    if opt['set_vocab_size']:
        model_vocab_opt = model_class.get_vocab_opt(*(v.vocab_size for v in vocabs))
        model_opt = ChainMap(model_vocab_opt, model_opt)

    logger.info('Building graph...')
    train_batch_iter = partial(batch_iter, *data[0])
    valid_batch_iter = partial(batch_iter, *data[1])
    train_model = sq.SeqModel(check_feed_dict=True)
    init_lr = train_opt['train:init_lr']
    _nodes = train_model.build_graph(model_opt)
    train_model.set_default_feed('train_loss_denom', opt['batch_size'])
    train_model.set_default_feed('eps', ngram_stat_data())
    lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    train_op = sq.create_train_op(
        train_model.training_loss, optim_class=train_opt['train:optim_class'],
        learning_rate=lr, clip_gradients=train_opt['train:clip_gradients'])

    eval_batch_iter = partial(batch_iter, *data[-1])
    eval_model = sq.SeqModel()
    _nodes = eval_model.build_graph(model_opt, reuse=is_training, no_dropout=True)

    logger.debug('Trainable Variables:')
    for v in tf.trainable_variables():
        logger.debug(f'{v.name}, {v.get_shape()}')

    sess_conf = tf.ConfigProto() if opt['gpu'] else tf.ConfigProto(device_count={'GPU': 0})  # noqa

    with tf.Session(config=sess_conf) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if is_training:
            logger.info('Training...')
            success, train_state = sq.load_exp(sess, saver, opt['exp_dir'], latest=True,
                                               checkpoint=opt['load_checkpoint'])
            if success:
                logger.info('Loaded model from checkpoint.')
            if train_state is None:
                logger.info('No experiment to resume.')
            else:
                logger.info('Resume experiment.')
            train_fn = partial(run_fn, sess, train_model, train_batch_iter, train_op)
            valid_fn = partial(run_fn, sess, eval_model, valid_batch_iter)
            begin_epoch_fn = partial(
                sq.update_learning_rate, partial(train_model.set_default_feed, lr),
                **sq.dict_with_key_startswith(train_opt, 'lr:'))

            def end_epoch_fn(train_state):
                sq.save_exp(sess, saver, opt['exp_dir'], train_state)
                return sq.is_done_training_early(train_state, train_opt['lr:imp_wait'])

            sq.train(train_fn, logger, max_epoch=train_opt['train:max_epoch'],
                     train_state=train_state, init_lr=init_lr,
                     valid_run_epoch_fn=valid_fn, begin_epoch_fn=begin_epoch_fn,
                     end_epoch_fn=end_epoch_fn)

        checkpoint = None if is_training else opt['load_checkpoint']
        if checkpoint is not None:
            logger.info(f'Loading parameters from `{checkpoint}` ...')
        else:
            _m = 'latest' if opt['eval_latest'] else 'best'
            logger.info(f'Loading parameters from {_m} checkpoint...')
        success, __ = sq.load_exp(sess, saver, opt['exp_dir'], latest=opt['eval_latest'],
                                  checkpoint=checkpoint)
        if not success:
            logger.warn('Loading model from checkpoint failed.')
        logger.info('Evaluating...')
        info = eval_run_fn(sess, eval_model, eval_batch_iter)
        logger.info(info.summary('eval'))

if __name__ == '__main__':
    start_time = time.time()
    group_default = {'model': sq.SeqModel.default_opt(),
                     'train': sq.default_training_opt(),
                     'decode': sq.default_decoding_opt()}
    parser = sq.get_common_argparser('main_lm.py')
    parser.add_argument('--seq_len', type=int, default=20, help=' ')
    parser.add_argument('--sentence_level', action='store_true', help=' ')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default)
    logger, all_opt = sq.init_exp_opts(opt, groups, group_default)
    opt, model_opt, train_opt, decode_opt = all_opt
    mle(opt, model_opt, train_opt, logger, data_fn, sq.SeqModel)
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
