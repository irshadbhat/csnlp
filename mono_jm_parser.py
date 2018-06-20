#!/usr/bin/python -*- coding: utf-8 -*-

import dynet_config
dynet_config.set(random_seed=127, autobatch=1)

import io
import os
import re
import sys
import copy
import string
import random
import pickle
import socket
import argparse
import StringIO
import threading

import numpy as np
from collections import Counter, namedtuple, defaultdict

import dynet as dy
from gensim.models.word2vec import Word2Vec

from algorithms.swap import Swap
from utils.inorder_traversal import * #get_in_order(graph) without dummies
from utils.pseudoProjectivity import *
from algorithms.arc_eager import ArcEager
from algorithms.arc_standard import ArcStandard


_MAX_BUFFER_SIZE_ = 102400 #100KB


class ClientThread(threading.Thread):
    def __init__(self, ip, port, clientsocket, parser):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.parser = parser
        self.csocket = clientsocket

    def run(self):
        data = self.csocket.recv(_MAX_BUFFER_SIZE_)
        dummyInputFile = StringIO.StringIO(data)
        dummyOutputFile = StringIO.StringIO("")
        for line in dummyInputFile:
            line = line.decode('utf-8').split()
            if not line:
                continue
            dummyOutputFile.write(parse_sent(self.parser, line).encode('utf-8')+'\n\n')
        dummyInputFile.close()
        self.csocket.send(dummyOutputFile.getvalue())
        dummyOutputFile.close()
        self.csocket.close()


class Meta:
    def __init__(self, palgo='swap'):
        self.palgo = palgo  # parsing algorithm
        self.c_dim = 32  # character-rnn input dimension
        self.add_words = 1  # additional lookup for missing/special words
        self.p_hidden = 64  # pos-mlp hidden layer dimension
        self.n_hidden = 128  # parser-mlp hidden layer dimension
        self.lstm_wc_dim = 128  # LSTM (word-char concatenated input) output dimension
        self.lstm_char_dim = 64  # char-LSTM output dimension
        self.window = 2 
        self.transitions = {'SHIFT':0,'LEFTARC':1,'RIGHTARC':2}  # parser core transitions
        if palgo == 'eager':
            self.transitions['REDUCE'] = 3
        elif palgo == 'swap':
            self.window = 3
            self.transitions['SWAP'] = 3

class Configuration(object):
    def __init__(self, nodes=[], standard=False):
        self.b0 = 1
        self.stack = list()
        self.queue = range(len(nodes))[1:]
        if standard:
            self.nodes = nodes[:1] + get_in_order(nodes[1:-1]) + nodes[-1:]
            for tnode in range(1, len(self.nodes[1:-1])+1):
                tnodeparent = self.nodes[self.nodes[tnode].parent]
                self.nodes[tnodeparent.id] = self.nodes[tnodeparent.id]._replace(children=tnodeparent.children+[tnode])
        else:
            self.nodes = nodes


class Parser(object):
    def __init__(self, model=None, meta=None):
        self.meta = pickle.load(open('%s.meta' %model, 'rb')) if model else meta
        self.model = dy.Model()
        if not getattr(self.meta, 'palgo', None):
            self.meta.palgo = 'eager'
        if self.meta.palgo == 'eager':
            self.transitionSystem = ArcEager()
        elif self.meta.palgo == 'swap':
            self.transitionSystem = Swap()
        else: 
            self.transitionSystem = ArcStandard()

        # define pos-mlp
        self.ps_pW1 = self.model.add_parameters((self.meta.p_hidden, self.meta.lstm_wc_dim*2))
        self.ps_pb1 = self.model.add_parameters(self.meta.p_hidden) 
        self.ps_pW2 = self.model.add_parameters((self.meta.n_tags, self.meta.p_hidden)) 
        self.ps_pb2 = self.model.add_parameters(self.meta.n_tags)

        # define parse-mlp
        self.pr_pW1 = self.model.add_parameters((self.meta.n_hidden, self.meta.lstm_wc_dim*2*self.meta.window))
        self.pr_pb1 = self.model.add_parameters(self.meta.n_hidden) 
        self.pr_pW2 = self.model.add_parameters((self.meta.n_outs, self.meta.n_hidden)) 
        self.pr_pb2 = self.model.add_parameters(self.meta.n_outs)

        # define char-rnns
        self.cfwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        self.cbwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)

        # define base Bi-LSTM for input word sequence (takes word+char-rnn embeddings as input)
        self.fwdRNN = dy.LSTMBuilder(1, self.meta.w_dim+self.meta.lstm_char_dim*2, self.meta.lstm_wc_dim, self.model)
        self.bwdRNN = dy.LSTMBuilder(1, self.meta.w_dim+self.meta.lstm_char_dim*2, self.meta.lstm_wc_dim, self.model)

        # define Bi-LSTM for POS feature representation (takes base Bi-LSTM output as input)
        self.ps_fwdRNN = dy.LSTMBuilder(1, self.meta.lstm_wc_dim*2, self.meta.lstm_wc_dim, self.model)
        self.ps_bwdRNN = dy.LSTMBuilder(1, self.meta.lstm_wc_dim*2, self.meta.lstm_wc_dim, self.model)

        # define Bi-LSTM for parser feature representation (takes base Bi-LSTM output and pos-hidden-state as input)
        self.pr_fwdRNN = dy.LSTMBuilder(1, self.meta.lstm_wc_dim*2+self.meta.p_hidden, self.meta.lstm_wc_dim, self.model)
        self.pr_bwdRNN = dy.LSTMBuilder(1, self.meta.lstm_wc_dim*2+self.meta.p_hidden, self.meta.lstm_wc_dim, self.model)

        # pad-node for missing nodes in partial parse tree
        self.PAD = self.model.add_parameters(self.meta.lstm_wc_dim*2)

        # define lookup tables
        self.LOOKUP_WORD = self.model.add_lookup_parameters((self.meta.n_words+self.meta.add_words, self.meta.w_dim))
        self.LOOKUP_CHAR = self.model.add_lookup_parameters((self.meta.n_chars, self.meta.c_dim))

        # load pretrained embeddings
        if args.embd and model is None:
            for word, V in wvm.vocab.iteritems():
                self.LOOKUP_WORD.init_row(V.index+self.meta.add_words, wvm.syn0[V.index])

        # load pretrained dynet model
        if model:
            self.model.populate('%s.dy' %model)

    def enable_dropout(self):
        self.fwdRNN.set_dropout(0.3)
        self.bwdRNN.set_dropout(0.3)
        self.cfwdRNN.set_dropout(0.3)
        self.cbwdRNN.set_dropout(0.3)
        self.ps_fwdRNN.set_dropout(0.3)
        self.ps_bwdRNN.set_dropout(0.3)
        self.pr_fwdRNN.set_dropout(0.3)
        self.pr_bwdRNN.set_dropout(0.3)
        self.ps_W1 = dy.dropout(self.ps_W1, 0.3)
        self.ps_b1 = dy.dropout(self.ps_b1, 0.3)
        self.pr_W1 = dy.dropout(self.pr_W1, 0.3)
        self.pr_b1 = dy.dropout(self.pr_b1, 0.3)

    def disable_dropout(self):
        self.fwdRNN.disable_dropout()
        self.bwdRNN.disable_dropout()
        self.cfwdRNN.disable_dropout()
        self.cbwdRNN.disable_dropout()
        self.ps_fwdRNN.disable_dropout()
        self.ps_bwdRNN.disable_dropout()
        self.pr_fwdRNN.disable_dropout()
        self.pr_bwdRNN.disable_dropout()

    def initialize_graph_nodes(self):
        #  convert parameters to expressions
        self.pad = dy.parameter(self.PAD)
        self.ps_W1 = dy.parameter(self.ps_pW1)
        self.ps_b1 = dy.parameter(self.ps_pb1)
        self.ps_W2 = dy.parameter(self.ps_pW2)
        self.ps_b2 = dy.parameter(self.ps_pb2)
        self.pr_W1 = dy.parameter(self.pr_pW1)
        self.pr_b1 = dy.parameter(self.pr_pb1)
        self.pr_W2 = dy.parameter(self.pr_pW2)
        self.pr_b2 = dy.parameter(self.pr_pb2)

        # apply dropout
        if self.eval:
            self.disable_dropout()
        else:
            self.enable_dropout() 

        # initialize the RNNs
        self.f_init = self.fwdRNN.initial_state()
        self.b_init = self.bwdRNN.initial_state()

        self.cf_init = self.cfwdRNN.initial_state()
        self.cb_init = self.cbwdRNN.initial_state()

        self.ps_f_init = self.ps_fwdRNN.initial_state()
        self.ps_b_init = self.ps_bwdRNN.initial_state()

        self.pr_f_init = self.pr_fwdRNN.initial_state()
        self.pr_b_init = self.pr_bwdRNN.initial_state()

    def word_rep(self, w):
        if not self.eval and random.random() < 0.3:
            return self.LOOKUP_WORD[0]
        if self.meta.lang == 'eng':
            idx = self.meta.w2i.get(w, self.meta.w2i.get(w.lower(), 0))
        else:
            idx = self.meta.w2i.get(w, 0)
        return self.LOOKUP_WORD[idx]

    def char_rep(self, w, f, b):
        no_c_drop = False
        if self.eval or random.random()<0.9:
            no_c_drop = True
        bos, eos, unk = self.meta.c2i["bos"], self.meta.c2i["eos"], self.meta.c2i['unk']
        char_ids = [bos] + [self.meta.c2i.get(c, unk) if no_c_drop else unk for c in w] + [eos]
        char_embs = [self.LOOKUP_CHAR[cid] for cid in char_ids]
        fw_exps = f.transduce(char_embs)
        bw_exps = b.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def get_char_embds(self, sentence, cf, cb):
        char_embs = []
        for node in sentence:
            char_embs.append(self.char_rep(node.form, cf, cb))
        return char_embs

    def get_word_embds(self, sentence):
        word_embs = []
        for node in sentence:
            word_embs.append(self.word_rep(node.form))
        return word_embs

    def basefeatures(self, nodes, stack, i):
        #NOTE Stack nodes
        #s3 = nodes[stack[-4]] if stack[3:] else nodes[0].left
        #s2 = nodes[stack[-3]] if stack[2:] else nodes[0].left
        s1 = nodes[stack[-2]] if stack[1:] else nodes[0].left
        s0 = nodes[stack[-1]] if stack else nodes[0].left

        #NOTE Buffer nodes
        n0 = nodes[i] if nodes[i:] else nodes[0].left # i here is the first node in the queue (impt. for swap action)
        #n0left = n0.left if i else [None]

        #NOTE Leftmost and Rightmost children of s2,s1,s0 and b0(only leftmost)
        #s3l = nodes[s3.left [-1]] if s3.left [-1] != None else nodes[0].left
        #s3r = nodes[s3.right[-1]] if s3.right[-1] != None else nodes[0].left
        #s2l = nodes[s2.left [-1]] if s2.left [-1] != None else nodes[0].left
        #s2r = nodes[s2.right[-1]] if s2.right[-1] != None else nodes[0].left
        #s1l = nodes[s1.left [-1]] if s1.left [-1] != None else nodes[0].left
        #s1r = nodes[s1.right[-1]] if s1.right[-1] != None else nodes[0].left
        #s0l = nodes[s0.left [-1]] if s0.left [-1] != None else nodes[0].left
        #s0r = nodes[s0.right[-1]] if s0.right[-1] != None else nodes[0].left
        #n0l = nodes[n0left [-1]]  if n0left  [-1] != None else nodes[0].left
        #n0r = nodes[n0.right[-1]] if n0.right[-1] != None else nodes[0].left

        if self.meta.palgo == 'swap':
            return [(nd.id, nd.form) for nd in s1,s0,n0]
        elif self.meta.palgo == "eager":
            return [(nd.id, nd.form) for nd in s0, n0]
        else:
            return [(nd.id, nd.form) for nd in s1,s0]

    def feature_extraction(self, sentence):
        self.initialize_graph_nodes()

        # get word/char embeddings
        wembs = self.get_word_embds(sentence)
        cembs = self.get_char_embds(sentence, self.cf_init, self.cb_init)
        lembs = [dy.concatenate([w,c]) for w,c in zip(wembs, cembs)]

        # feed word vectors into base biLSTM
        fw_exps = self.f_init.transduce(lembs)
        bw_exps = self.b_init.transduce(reversed(lembs))
        bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]
    
        # feed biLSTM embeddings into POS biLSTM
        ps_fw_exps = self.ps_f_init.transduce(bi_exps)
        ps_bw_exps = self.ps_b_init.transduce(reversed(bi_exps))
        ps_bi_exps = [dy.concatenate([f,b]) for f,b in zip(ps_fw_exps, reversed(ps_bw_exps))]

        # get pos-hidden representation and pos loss
        pos_errs, pos_hidden = [], []
        for xi,node in zip(ps_bi_exps, sentence):
            xh = self.ps_W1 * xi
            pos_hidden.append(xh)
            xh = self.meta.activation(xh) + self.ps_b1
            xo = self.ps_W2*xh + self.ps_b2
            err = dy.softmax(xo).npvalue() if self.eval else dy.pickneglogsoftmax(xo, self.meta.p2i[node.tag])
            pos_errs.append(err)

        # concatenate pos hidden-layer with base biLSTM 
        wcp_exps = [dy.concatenate([w,p]) for w,p in zip(bi_exps, pos_hidden)]
        # feed concatenated embeddings into parse biLSTM
        pr_fw_exps = self.pr_f_init.transduce(wcp_exps)
        pr_bw_exps = self.pr_b_init.transduce(reversed(wcp_exps))
        pr_bi_exps = [dy.concatenate([f,b]) for f,b in zip(pr_fw_exps, reversed(pr_bw_exps))]

        return  pr_bi_exps, pos_errs

    def predict(self, configuration, pr_bi_exps):
        rfeatures = self.basefeatures(configuration.nodes, configuration.stack, configuration.b0)
        xi = dy.concatenate([pr_bi_exps[id-1] if id > 0 else self.pad for id, rform in rfeatures])
        xh = self.pr_W1 * xi
        xh = self.meta.activation(xh) + self.pr_b1
        return self.pr_W2*xh + self.pr_b2
    
def Train(sentence, epoch, dynamic=True):
    parser.eval = False
    if parser.meta.palgo in ['standard', 'swap']:
        configuration = Configuration(sentence, standard=True)
    else:
        configuration = Configuration(sentence)
    pr_bi_exps, pos_errs = parser.feature_extraction(sentence[1:-1])
    while not parser.transitionSystem.inFinalState(configuration):
        xo = parser.predict(configuration, pr_bi_exps)
        if parser.meta.palgo in ['swap', 'standard']: # Static Oracle
            goldTransitionFunc, goldLabel = parser.transitionSystem.LabelledAction(configuration)
            goldTransition = goldTransitionFunc.__name__
            parser.loss.append(dy.pickneglogsoftmax(xo, parser.meta.td2i[(goldTransition, goldLabel)]))
            goldTransitionFunc(configuration, goldLabel)
        else: # Dynamic Oracle
            output_probs = dy.softmax(xo).npvalue()
            ranked_actions = sorted(zip(output_probs, range(len(output_probs))), reverse=True)
            pscore, paction = ranked_actions[0]

            #{0: <bound method arceager.SHIFT>}
            validTransitions, allmoves = parser.transitionSystem.get_valid_transitions(configuration)
            while parser.transitionSystem.action_cost(\
                    configuration, parser.meta.i2td[paction], parser.meta.transitions, validTransitions) > 500:
                ranked_actions = ranked_actions[1:]
                pscore, paction = ranked_actions[0]

            gaction = None
            for i,(score, ltrans) in enumerate(ranked_actions):
                cost = parser.transitionSystem.action_cost(\
                    configuration, parser.meta.i2td[ltrans], parser.meta.transitions, validTransitions)
                if cost == 0:
                    gaction = ltrans
                    need_update = (i > 0)
                    break

            gtransitionstr, goldLabel = parser.meta.i2td[gaction]
            ptransitionstr, predictedLabel = parser.meta.i2td[paction]
            if dynamic and (epoch > 2) and (np.random.random() < 0.9):
               predictedTransitionFunc = allmoves[parser.meta.transitions[ptransitionstr]]
               predictedTransitionFunc(configuration, predictedLabel)
            else:
               goldTransitionFunc = allmoves[parser.meta.transitions[gtransitionstr]]
               goldTransitionFunc(configuration, goldLabel)
            parser.loss.append(dy.pickneglogsoftmax(xo, parser.meta.td2i[(gtransitionstr, goldLabel)])) #NOTE original

    parser.loss.extend(pos_errs)

def build_dependency_graph(parser, graph):
    dy.renew_cg()
    pred_pos = []
    parser.eval = True
    pr_bi_exps, pos_errs = parser.feature_extraction(graph[1:-1])
    for xo, node in zip(pos_errs, graph[1:-1]):
        p_tag = parser.meta.i2p[np.argmax(xo)]
        pred_pos.append(p_tag)

    configuration = Configuration(graph)
    while not parser.transitionSystem.inFinalState(configuration):
        output_probs = parser.predict(configuration, pr_bi_exps)
        output_probs = dy.softmax(output_probs).npvalue()
        validTransitions, _ = parser.transitionSystem.get_valid_transitions(configuration) #{0: <bound method arceager.SHIFT>}
        sortedPredictions = sorted(zip(output_probs, range(len(output_probs))), reverse=True)
        for score, action in sortedPredictions:
            transition, predictedLabel = parser.meta.i2td[action]
            if parser.meta.transitions[transition] in validTransitions:
                predictedTransitionFunc = validTransitions[parser.meta.transitions[transition]]
                predictedTransitionFunc(configuration, predictedLabel)
                break

    if parser.meta.palgo == 'swap':
        return graph[1:-1], pred_pos
    else:
        return deprojectivize(graph[1:-1]), pred_pos


def test_conll(parser, dev_file, ofp=None):
    with io.open(dev_file, encoding='utf-8') as fp:
        inputGenTest = re.finditer("(.*?)\n\n", fp.read(), re.S)

    scores = defaultdict(int)
    correct_pos, incorrect_pos = 0.0, 0.0
    for idx, sentence in enumerate(inputGenTest):
        dy.renew_cg()
        graph = list(depenencyGraph(sentence.group(1)))
        graph, ppos = build_dependency_graph(parser, graph)
        scores = tree_eval(graph, scores)
        for node,p_tag in zip(graph, ppos):
            if node.tag == p_tag:
                correct_pos += 1
            else:
                incorrect_pos += 1
        if ofp:
            for node,pos in zip(graph, ppos):
                ofp.write('\t'.join([unicode(node.id), node.form, u'_', pos, u'_', u'_',
                            unicode(node.pparent), node.pdrel.strip('%'), u'_', u'_'])+'\n')
            ofp.write(u'\n')

    pos_score = round(100. * correct_pos/(correct_pos+incorrect_pos), 2)
    UAS = round(100. * scores['rightAttach']/(scores['rightAttach']+scores['wrongAttach']),2)
    LS  = round(100. * scores['rightLabel']/(scores['rightLabel']+scores['wrongLabel']), 2)
    LAS = round(100. * scores['rightLabeledAttach']/(scores['rightLabeledAttach']+scores['wrongLabeledAttach']),2)
    return pos_score, UAS, LS, LAS

def parse_sent(parser, sentence):
    leaf = namedtuple('leaf', ['id','form','lemma','tag','ctag','features','parent','pparent', 'drel','pdrel','left','right', 'visit'])
    PAD = leaf._make([-1,'PAD','PAD','PAD','PAD','PAD',-1,-1,'PAD','PAD',[None],[None], False])
    graph = [leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', 'ROOT_T', -1, -1, 'ROOT', 'ROOT', PAD, [None], False])]
    graph += [leaf._make([i,w,'_','_','_','_',-1,-1,'_','_',[None],[None], False]) for i,w in enumerate(sentence, 1)]
    graph += [leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', 'ROOT_T', -1, -1, 'ROOT', 'ROOT', [None], [None], False])]
    graph, ppos = build_dependency_graph(parser, graph)
    return '\n'.join(['\t'.join([unicode(node.id), node.form, u'_', pos, u'_', u'_', unicode(node.pparent),
                      node.pdrel.strip('%'), u'_', u'_']) for node,pos in zip(graph, ppos)])

def test_raw_sents(parser, test_file, ofp):
    with io.open(test_file, encoding='utf-8') as ifp:
        for line in ifp:
            line = line.split()
            if not line:
                continue
            parsed_sent = parse_sent(parser, line)
            ofp.write(parsed_sent+'\n\n')

def tree_eval(sentence, scores):
    for node in sentence:
        if node.parent == node.pparent:
            scores['rightAttach'] += 1
            if node.drel.strip('%') == node.pdrel.strip('%'):
                scores['rightLabeledAttach'] += 1
            else:
                scores['wrongLabeledAttach'] += 1
        else:
            scores['wrongAttach'] += 1
            scores['wrongLabeledAttach'] += 1

        if node.drel.strip('%') == node.pdrel.strip('%'):
            scores['rightLabel'] += 1
        else:
            scores['wrongLabel'] += 1
    return scores

def train_parser(dataset):
    n_samples = len(dataset)
    sys.stdout.write("Started training ...\n")
    sys.stdout.write("Training Examples: %s Classes: %s Epochs: %d\n\n" % (n_samples, parser.meta.n_outs, args.iter))
    psc, num_tagged, cum_loss = 0., 0, 0.
    for epoch in range(args.iter):
        random.shuffle(dataset)
        parser.loss = []
        dy.renew_cg()
        for sid, sentence in enumerate(dataset, 1):
            if sid % 500 == 0 or sid == n_samples:   # print status
                trainer.status()
                print(cum_loss / num_tagged)
                cum_loss, num_tagged = 0, 0
                sys.stdout.flush()
            csentence = copy.deepcopy(sentence)
            Train(csentence, epoch+1)
            num_tagged += 2 * len(sentence[1:-1]) - 1
            if len(parser.loss) > 75:
                batch_loss = dy.esum(parser.loss)
                cum_loss += batch_loss.scalar_value()
                batch_loss.backward()
                trainer.update()
                parser.loss = []
                dy.renew_cg()
        sys.stderr.flush()
        if parser.loss:
            batch_loss = dy.esum(parser.loss)
            cum_loss += batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()
            parser.loss = []
            dy.renew_cg()
        CLAS = 0.0
        for dfile in args.dev:
            POS, UAS, LS, LAS = test_conll(parser, dfile)
            sys.stderr.write(\
                "{} > POS ACCURACY: {}% UAS: {}%, LS: {}% and LAS: {}%\n".format(dfile.rsplit('/')[-1][:3], POS, UAS, LS, LAS))
            CLAS += LAS
        if CLAS > psc:
            sys.stderr.write('SAVE POINT %d\n' %epoch)
            psc = CLAS
            if args.save_model:
                parser.model.save('%s.dy' %args.save_model)

def MPC(node, parent, tree):
    if parent == -1:
        return node
    else:
        node, parent = tree[parent].id, tree[parent].pparent
        return MPC(node, parent,tree)

def MPCs(graph, stparser):
    configuration = Configuration(graph, standard=True)
    while not stparser.inFinalState(configuration):
        goldTransitionFunc, goldLabel = stparser.LabelledAction(configuration)
        goldTransition = goldTransitionFunc.__name__
        goldTransitionFunc(configuration, goldLabel)

    for p in range(1,len(configuration.nodes[1:])+1):
        pN = configuration.nodes[p]
        pNParent = pN.pparent
        if pNParent == -1:
            maxProjection = pN.id
        else:
            maxProjection = MPC(pN,pNParent, configuration.nodes)
        configuration.nodes[p] = configuration.nodes[p]._replace(
                                        pdrel="__PAD__",
                                        inorder=-1,
                                        left=[None],
                                        right=[None],
                                        children=[],
                                        mpc=maxProjection)
    configuration.nodes[-1] = configuration.nodes[-1]._replace(inorder=p)
    return [cNode._replace(pparent=-1) for cNode in configuration.nodes]

def projective(nodes):
    """Identifies if a tree is non-projective or not."""
    for leaf1 in nodes:
        v1,v2 = sorted([int(leaf1.id), int(leaf1.parent)])
        for leaf2 in nodes:
            v3, v4 = sorted([int(leaf2.id), int(leaf2.parent)])
            if leaf1.id == leaf2.id:continue
            if (v1 < v3 < v2) and (v4 > v2): return False
    return True


def depenencyGraph(sentence):
    """Representation for dependency trees"""
    leaf = namedtuple('leaf', ['id','form','lemma','tag','ctag','features','parent','pparent',
                               'drel','pdrel','left','right', 'visit','inorder','children','mpc'])
    PAD = leaf._make([-1,'PAD','PAD','PAD','PAD','PAD',-1,-1,'PAD','PAD',[None],[None], False,-1, [],-1])
    yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', 'ROOT_T', -1, -1, 'ROOT', 'ROOT', PAD, [None], False,-1, [], 0])

    for idx, node in enumerate(sentence.split("\n"), 1):
        id_,form,lemma,tag,ctag,features,parent,drel = node.split("\t")[:8]
        if not id_.isdigit():
            continue
        #drel = drel.replace('-', '_')
        if args.ud:
            node = leaf._make([int(id_),form,lemma,tag,ctag,features,int(parent),-1,drel,drel,[None],[None], False, -1, [], idx])
        else:
            node = leaf._make([int(id_),form,lemma,ctag,tag,features,int(parent),-1,drel,drel,[None],[None], False, -1, [], idx])
        yield node
    yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', 'ROOT_T', -1, -1, 'ROOT', 'ROOT', [None], [None], False, -1, [], idx+1])


def read(fname, palgo):
    data = []
    mpcparser = ArcStandard()
    with io.open(fname, encoding='utf-8') as fp:
        inputGenTrain = re.finditer("(.*?)\n\n", fp.read(), re.S)

    for i,sentence in enumerate(inputGenTrain):
        graph = list(depenencyGraph(sentence.group(1)))
        if palgo == 'swap':
            if not projective(graph[1:-1]):
                graph = MPCs(graph, mpcparser)
            data.append(graph)
        else:
            pgraph = graph[:1] + projectivize(graph[1:-1]) + graph[-1:]
            data.append(pgraph)
    return data


def set_label_map(train_sents):
    meta.c2i = {'bos':0, 'eos':1, 'unk':2}
    cid = len(meta.c2i)
    for graph in train_sents:
        for node in graph[1:-1]:
            plabels.add(node.tag)
            if node.parent == 0:
                tdlabels.add(('LEFTARC', node.drel))
            elif node.id < node.parent:
                tdlabels.add(('LEFTARC', node.drel))
            else:
                tdlabels.add(('RIGHTARC', node.drel))
            for c in node.form:
                if not meta.c2i.has_key(c):
                    meta.c2i[c] = cid
                    cid += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Neural Network Parser.", description="Bi-LSTM Parser")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dynet-gpu')
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-devices')
    parser.add_argument('--dynet-autobatch')
    parser.add_argument('--dynet-seed', dest='seed', type=int, default='127')
    parser.add_argument('--train', nargs='+', help='CONLL Train file')
    parser.add_argument('--dev', nargs='+', help='CONLL Dev/Test file')
    parser.add_argument('--test', help='Raw Test file')
    parser.add_argument('--pretrained-embds', dest='embd', help='Pretrained word2vec Embeddings')
    parser.add_argument('--elimit', type=int, default=None, help='load only top-n pretrained word vectors (default=all vectors)')
    parser.add_argument('--lang', help='3-letter ISO language code e.g., eng for English, hin for Hindi')
    parser.add_argument('--trainer', default='momsgd', help='Trainer [momsgd|adam|adadelta|adagrad]')
    parser.add_argument('--activation-fn', dest='act_fn', default='tanh', help='Activation function [tanh|rectify|logistic]')
    parser.add_argument('--ud', type=int, default=1, help='1 if UD treebank else 0')
    parser.add_argument('--iter', type=int, default=100, help='No. of Epochs')
    parser.add_argument('--algo', dest='palgo', action='store', choices=['eager','standard','swap'], default='swap', help='Parsing Algorithm')
    parser.add_argument('--bvec', type=int, help='1 if binary embedding file else 0')
    group.add_argument('--save-model', dest='save_model', help='Specify path to save model')
    group.add_argument('--load-model', dest='load_model', help='Load Pretrained Model')
    parser.add_argument('--output-file', dest='outfile', default='/tmp/out.conll', help='Output File')
    parser.add_argument('--daemonize', dest='isDaemon', action='store_true', default=False, help='Daemonize parser')
    parser.add_argument('--port', type=int, dest='daemonPort', help='Specify a port number')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not args.load_model:
        meta = Meta(palgo=args.palgo)
        plabels = set()
        tdlabels = set()
        meta.lang = args.lang
        tdlabels.add(('SHIFT', None))
        if args.palgo == "eager":
            tdlabels.add(('REDUCE', None))
        elif args.palgo == "swap":
            tdlabels.add(('SWAP', None))

        meta.w2i = {}
        train_sents = []
        for tfile in args.train:
            train_sents += read(tfile, args.palgo)
        set_label_map(train_sents)

        if args.embd:
            wvm = Word2Vec.load_word2vec_format(args.embd, binary=args.bvec, limit=args.elimit)
            meta.w_dim = wvm.syn0.shape[1]
            meta.n_words = wvm.syn0.shape[0]+meta.add_words
            for w in wvm.vocab:
                meta.w2i[w] = wvm.vocab[w].index + meta.add_words
        else:
            for graph in train_sents:
                for node in graph[1:-1]:
                    meta.w2i.setdefault(node.form, 0)
                    meta.w2i[node.form] += 1
            meta.w2i = {w:i for i,w in enumerate([w for w,c in meta.w2i.items() if c > 9], 1)}
            meta.w_dim = 64
            meta.n_words = len(meta.w2i) + 1

        meta.i2p = dict(enumerate(plabels))
        meta.i2td = dict(enumerate(tdlabels))
        meta.p2i = {v: k for k,v in meta.i2p.iteritems()}
        meta.td2i = {v: k for k,v in meta.i2td.iteritems()}
        meta.n_outs = len(meta.i2td)
        meta.n_tags = len(meta.p2i)
        meta.n_chars = len(meta.c2i)

        trainers = {
            'momsgd'   : dy.MomentumSGDTrainer,
            'adam'     : dy.AdamTrainer,
            'simsgd'   : dy.SimpleSGDTrainer,
            'adagrad'  : dy.AdagradTrainer,
            'adadelta' : dy.AdadeltaTrainer
            }
        act_fn = {
            'sigmoid' : dy.logistic,
            'tanh'    : dy.tanh,
            'relu'    : dy.rectify,
            }
        meta.trainer = trainers[args.trainer]
        meta.activation = act_fn[args.act_fn] 

    if args.save_model:
        pickle.dump(meta, open('%s.meta' %args.save_model, 'wb'))
    if args.load_model:
        sys.stderr.write('Loading Models ...\n')
        parser = Parser(model=args.load_model)
        sys.stderr.write('Done!\n')
        if args.isDaemon and args.daemonPort:
            host = "0.0.0.0" #Listen on all interfaces
            port = args.daemonPort #Port number
            tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            tcpsock.bind((host, port))
            while True:
                tcpsock.listen(4)
                # Listening for incoming connections
                (clientsock, (ip, port)) = tcpsock.accept()
                # pass clientsock to the ClientThread thread object being created
                newthread = ClientThread(ip, port, clientsock, parser)
                newthread.start()
        else:
            with io.open(args.outfile, 'w', encoding='utf-8') as ofp:
                if args.test:
                    test_raw_sents(parser, args.test, ofp)
                else:
                    for dfile in args.dev:
                        POS, UAS, LS, LAS = test_conll(parser, dfile, ofp)
                        sys.stderr.write("TEST-SET POS: {}%, UAS: {}%, LS: {}% and LAS: {}%\n".format(POS, UAS, LS, LAS))
    else:
        parser = Parser(meta=meta)
        trainer = meta.trainer(parser.model)
        train_parser(train_sents)
