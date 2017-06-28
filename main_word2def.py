import time
import os
from functools import partial

import kenlm
import numpy as np

from _main import sq
from _main import mle
from _main import decode
from _main import policy_gradient


def get_reward_fn(opt, pg_opt):
    # XXX: LM Score
    # lm = kenlm.Model('../experiment/dm/ngram_lm/train_no_wbdef.arpa')
    # vocab = sq.Vocabulary.from_vocab_file(
    #     'data/common_wordnet_defs/lemma_senses/dec_vocab.txt')
    # reward_fn = partial(sq.reward_ngram_lm, lm=lm, vocab=vocab)
    # return reward_fn

    # XXX: BLEU, not efficent (load data twice)
    dpath = partial(os.path.join, opt['data_dir'])
    enc_vocab = sq.Vocabulary.from_vocab_file(dpath('enc_vocab.txt'))
    dec_vocab = sq.Vocabulary.from_vocab_file(dpath('dec_vocab.txt'))
    file_list = (opt['train_file'], opt['valid_file'], opt['eval_file'])
    line_fn = partial(sq.read_lines, token_split=' ', part_split='\t', part_indices=(0, -1))  # noqa
    read_fn = partial(sq.read_seq2seq_data, in_vocab=enc_vocab, out_vocab=dec_vocab)
    data = [read_fn(line_fn(dpath(f))) for i, f in enumerate(file_list)]
    group_fn = partial(sq.group_data, key=lambda e: e[0][0], entry=lambda e: e[1][1:])
    refereces = group_fn(zip(*data[0]))
    refereces.update(group_fn(zip(*data[1])))
    refereces.update(group_fn(zip(*data[2])))
    refereces[0] = None

    def ref_fn(batch):
        refs = []
        for wbdef in batch.features.enc_inputs[0, :]:
            refs.append(refereces[wbdef])
        return refs
    return partial(sq.reward_bleu, ref_fn=ref_fn, reward_incomplete=True)

    # XXX: Constant
    # return sq.reward_constant


def pack_data(batch, sample, ret):
    pg_batch = sq.get_batch_data(batch, sample, input_key='dec_inputs',
                                 seq_len_key='dec_seq_len')
    full_batch = sq.concat_word2def_batch(pg_batch, batch)
    full_ret = np.copy(full_batch.labels.label_weight)
    full_ret[:ret.shape[0], :ret.shape[1]] = ret
    return full_batch, full_ret


if __name__ == '__main__':
    start_time = time.time()
    group_default = {'model': sq.Word2DefModel.default_opt(),
                     'train': sq.default_training_opt(),
                     'pg': sq.policy_gradient_opt(),
                     'decode': sq.default_decoding_opt()}
    parser = sq.get_common_argparser('main_word2word.py')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default, dup_replaces=('enc:', 'dec:'))
    logger, all_opt = sq.init_exp_opts(opt, groups, group_default)
    opt, model_opt, train_opt, decode_opt, pg_opt = all_opt

    def data_fn():
        dpath = partial(os.path.join, opt['data_dir'])
        enc_vocab = sq.Vocabulary.from_vocab_file(dpath('enc_vocab.txt'))
        dec_vocab = sq.Vocabulary.from_vocab_file(dpath('dec_vocab.txt'))
        char_vocab = sq.Vocabulary.from_vocab_file(dpath('char_vocab.txt'))
        file_list = (opt['train_file'], opt['valid_file'], opt['eval_file'])
        line_fn = partial(sq.read_lines, token_split=' ', part_split='\t',
                          part_indices=(0, -1))
        read_fn = partial(sq.read_word2def_data, in_vocab=enc_vocab,
                          out_vocab=dec_vocab, char_vocab=char_vocab)
        data = [read_fn(line_fn(dpath(f)), freq_down_weight=i != 2)
                for i, f in enumerate(file_list)]

        # XXX: SoXC
        # print('load ngram data')
        # data_ngram = read_fn(line_fn(
        #     '../experiment/dm2/ngram_lm/decode/sampling_train.txt'),
        #     freq_down_weight=True, init_seq_weight=0.5)
        # for T, N in zip(data[0], data_ngram):
        #     T.extend(N)
        # XXX: EoXC

        batch_iter = partial(sq.word2def_batch_iter, batch_size=opt['batch_size'])
        return data, batch_iter, (enc_vocab, dec_vocab, char_vocab)

    if opt['command'] == 'decode':
        with open(decode_opt['decode:outpath'], 'w') as ofp:
            def decode_batch(batch, samples, vocabs):
                words = vocabs[0].i2w(batch.features.words)
                for b_samples in samples:
                    b_seq_len = sq.find_first_min_zero(b_samples)
                    for word, sample, seq_len in zip(words, b_samples.T, b_seq_len):
                        if word == '</s>':
                            continue
                        definition = ' '.join(vocabs[1].i2w(sample[0: seq_len]))
                        ofp.write(f'{word}\t{definition}\n')
            decode(opt, model_opt, decode_opt, decode_batch, logger,
                   data_fn, sq.Word2DefModel)
    else:
        if pg_opt['pg:enable']:
            reward_fn = get_reward_fn(opt, pg_opt)
            policy_gradient(opt, model_opt, train_opt, pg_opt, logger, data_fn,
                            sq.Word2DefModel, reward_fn=reward_fn,
                            pack_data_fn=pack_data)
        else:
            mle(opt, model_opt, train_opt, logger, data_fn, sq.Word2DefModel)
            # with open('tmp.txt', 'w') as ofp:
            #     def write_score(batch, collect):
            #         enc = batch.features.enc_inputs
            #         dec = batch.features.dec_inputs
            #         score = collect[0]
            #         for i in range(len(score)):
            #             _e = enc[0, i]
            #             _d = ' '.join([str(_x) for _x in dec[:, i]])
            #             ofp.write(f'{_e}\t{_d}\t{score[i]}\n')
            #     eval_fn = partial(sq.run_collecting_epoch,
            #                       collect_keys=['dec.batch_loss'],
            #                       collect_fn=write_score)
            #     mle(opt, model_opt, train_opt, logger, data_fn, sq.Word2DefModel,
            #         eval_run_fn=eval_fn)
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
