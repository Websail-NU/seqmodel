"""
Experiment context:

Creating agent and data iterators from configuration.
"""
import copy
import os
from pydoc import locate

from seqmodel.bunch import Bunch
from seqmodel.data import Vocabulary


class Context(object):
    def __init__(self, sess, agent_class, agent_opt, data_dir,
                 iterator_class, iterator_opt,
                 vocab_files=Bunch(in_vocab='vocab.txt',
                                   out_vocab='vocab.txt'),
                 data_files=Bunch(train="train.txt",
                                  valid="valid.txt",
                                  test="test.txt")):
        self.config = Bunch(
            agent_opt=agent_opt,
            agent_class=agent_class,
            iterator_class=iterator_class,
            data_dir=data_dir,
            iterator_opt=iterator_opt,
            vocab_files=vocab_files,
            data_files=data_files)
        self.agent = Context.create_agent(agent_class, agent_opt, sess)
        self.vocabs = Context.create_vocabs(data_dir, vocab_files)
        self.iterators = Context.create_iterators(
            data_dir, self.vocabs, iterator_class, iterator_opt, data_files)

    @staticmethod
    def create_agent(agent_class, agent_opt, sess):
        agent_class_ = locate(agent_class)
        return agent_class_(agent_opt, sess)

    @staticmethod
    def create_vocabs(data_dir, vocab_files):
        vocabs = Bunch()
        vocab_file_map = {}
        for key in vocab_files:
            path = os.path.join(data_dir, vocab_files[key])
            if path in vocab_file_map:
                vocabs[key] = vocab_file_map[path]
            else:
                vocab = Vocabulary.from_vocab_file(path)
                vocabs[key] = vocab
                vocab_file_map[path] = vocab
        return vocabs

    @staticmethod
    def create_iterators(data_dir, vocabs, iterator_class,
                         iterator_opt, data_files):
        iterators = Bunch()
        iter_class_ = locate(iterator_class)
        kwargs = Bunch(vocabs)
        for key in data_files:
            opt = copy.deepcopy(iterator_opt)
            opt.data_source = os.path.join(data_dir, data_files[key])
            iterators[key] = iter_class_(opt, **kwargs)
        return iterators

    @staticmethod
    def from_config_file(sess, filepath):
        config = Bunch.from_json_file(filepath)
        out_context = Context(sess, **config)
        return out_context

    def write_config(self, filepath):
        with open(filepath, 'w') as ofp:
            ofp.write(self.config.to_pretty_json())

    def initialize_iterators(self):
        for key in self.iterators:
            self.iterators[key].initialize()
