#)!/usr/bin/python -*- coding: utf-8 -*-

import io
import os
import re
import sys
import copy
import random
import pickle

import argparse
import numpy as np
from collections import Counter, namedtuple, defaultdict

import dynet as dy
from gensim.models.word2vec import Word2Vec

from arc_eager import ArcEager
from pseudoProjectivity import *

random.seed(37)
np.random.seed(37)

class Meta:
    def __init__(self):
        self.c_dim = 32  # character-rnn input dimension
        self.window = 2  # arc-eager feature window
        self.add_words = 1  # additional lookup for missing/special words
        self.p_hidden = 64  # pos-mlp hidden layer dimension
        self.n_hidden = 128  # parser-mlp hidden layer dimension
        self.lstm_wc_dim = 128  # LSTM (word-char concatenated input) output dimension
        self.lstm_char_dim = 64  # char-LSTM output dimension
        self.transitions = {'SHIFT':0,'LEFTARC':1,'RIGHTARC':2,'REDUCE':3}  # parser transitions
        ################################# STACKING-MODEL-DIMS ##################################
        self.xc_dim = 32
        self.xp_hidden = 64
        self.xn_hidden = 128
        self.xlstm_wc_dim = 128
        self.xlstm_char_dim = 64


class Configuration(object):
    def __init__(self, nodes=[]):
        self.stack = list()
        self.b0 = 1
        self.nodes = nodes


class Parser(ArcEager):
    def __init__(self, model=None, meta=None, new_meta=None, test=False):
        self.model = dy.Model()
        if new_meta:
            self.meta = new_meta
        else:
            self.meta = pickle.load(open('%s.meta' %model, 'rb')) if model else meta

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
        self.hcfwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        self.hcbwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        self.ecfwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        self.ecbwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)

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
        self.ELOOKUP_WORD = self.model.add_lookup_parameters((self.meta.n_words_eng, self.meta.w_dim))
        self.HLOOKUP_WORD = self.model.add_lookup_parameters((self.meta.n_words_hin, self.meta.w_dim))
        self.ELOOKUP_CHAR = self.model.add_lookup_parameters((self.meta.n_chars_eng, self.meta.c_dim))
        self.HLOOKUP_CHAR = self.model.add_lookup_parameters((self.meta.n_chars_hin, self.meta.c_dim))

        # load pretrained embeddings
        if model is None:
            for word, V in ewvm.vocab.iteritems():
                self.ELOOKUP_WORD.init_row(V.index+self.meta.add_words, ewvm.syn0[V.index])
            for word, V in hwvm.vocab.iteritems():
                self.HLOOKUP_WORD.init_row(V.index+self.meta.add_words, hwvm.syn0[V.index])

        # load pretrained dynet model
        if not test and model:
            self.model.populate('%s.dy' %model)
        ######################################### STACKING ##############################################
        self.xps_pW1 = self.model.add_parameters((self.meta.xp_hidden, self.meta.xlstm_wc_dim*2))
        self.xps_pb1 = self.model.add_parameters(self.meta.xp_hidden)
        self.xps_pW2 = self.model.add_parameters((self.meta.xn_tags, self.meta.xp_hidden))
        self.xps_pb2 = self.model.add_parameters(self.meta.xn_tags)

        self.xpr_pW1 = self.model.add_parameters((self.meta.xn_hidden, self.meta.xlstm_wc_dim*2*self.meta.window+self.meta.xn_hidden))
        self.xpr_pb1 = self.model.add_parameters(self.meta.xn_hidden)
        self.xpr_pW2 = self.model.add_parameters((self.meta.xn_outs, self.meta.xn_hidden))
        self.xpr_pb2 = self.model.add_parameters(self.meta.xn_outs)

        self.xhcfwdRNN = dy.LSTMBuilder(1, self.meta.xc_dim, self.meta.xlstm_char_dim, self.model)
        self.xhcbwdRNN = dy.LSTMBuilder(1, self.meta.xc_dim, self.meta.xlstm_char_dim, self.model)
        self.xecfwdRNN = dy.LSTMBuilder(1, self.meta.xc_dim, self.meta.xlstm_char_dim, self.model)
        self.xecbwdRNN = dy.LSTMBuilder(1, self.meta.xc_dim, self.meta.xlstm_char_dim, self.model)

        self.xps_fwdRNN = dy.LSTMBuilder(1, self.meta.w_dim+self.meta.xlstm_char_dim*2+self.meta.xp_hidden, self.meta.xlstm_wc_dim, self.model)
        self.xps_bwdRNN = dy.LSTMBuilder(1, self.meta.w_dim+self.meta.xlstm_char_dim*2+self.meta.xp_hidden, self.meta.xlstm_wc_dim, self.model)

        self.xpr_fwdRNN = dy.LSTMBuilder(1, self.meta.xlstm_wc_dim*2+self.meta.w_dim+self.meta.xlstm_char_dim*2+self.meta.xp_hidden,
                            self.meta.xlstm_wc_dim, self.model)
        self.xpr_bwdRNN = dy.LSTMBuilder(1, self.meta.xlstm_wc_dim*2+self.meta.w_dim+self.meta.xlstm_char_dim*2+self.meta.xp_hidden,
                            self.meta.xlstm_wc_dim, self.model)

        self.XPAD = self.model.add_parameters(self.meta.xlstm_wc_dim*2)
        if test and model:
            self.model.populate('%s.dy' %model)

    def enable_dropout(self):
        self.fwdRNN.set_dropout(0.3)
        self.bwdRNN.set_dropout(0.3)
        self.ecfwdRNN.set_dropout(0.3)
        self.ecbwdRNN.set_dropout(0.3)
        self.hcfwdRNN.set_dropout(0.3)
        self.hcbwdRNN.set_dropout(0.3)
        self.ps_fwdRNN.set_dropout(0.3)
        self.ps_bwdRNN.set_dropout(0.3)
        self.pr_fwdRNN.set_dropout(0.3)
        self.pr_bwdRNN.set_dropout(0.3)
        self.ps_W1 = dy.dropout(self.ps_W1, 0.3)
        self.ps_b1 = dy.dropout(self.ps_b1, 0.3)
        self.pr_W1 = dy.dropout(self.pr_W1, 0.3)
        self.pr_b1 = dy.dropout(self.pr_b1, 0.3)
        ########################################
        self.xecfwdRNN.set_dropout(0.3)
        self.xecbwdRNN.set_dropout(0.3)
        self.xhcfwdRNN.set_dropout(0.3)
        self.xhcbwdRNN.set_dropout(0.3)
        self.xps_fwdRNN.set_dropout(0.3)
        self.xps_bwdRNN.set_dropout(0.3)
        self.xpr_fwdRNN.set_dropout(0.3)
        self.xpr_bwdRNN.set_dropout(0.3)
        self.xps_W1 = dy.dropout(self.xps_W1, 0.3)
        self.xps_b1 = dy.dropout(self.xps_b1, 0.3)
        self.xpr_W1 = dy.dropout(self.xpr_W1, 0.3)
        self.xpr_b1 = dy.dropout(self.xpr_b1, 0.3)

    def disable_dropout(self):
        self.fwdRNN.disable_dropout()
        self.bwdRNN.disable_dropout()
        self.ecfwdRNN.disable_dropout()
        self.ecbwdRNN.disable_dropout()
        self.hcfwdRNN.disable_dropout()
        self.hcbwdRNN.disable_dropout()
        self.ps_fwdRNN.disable_dropout()
        self.ps_bwdRNN.disable_dropout()
        self.pr_fwdRNN.disable_dropout()
        self.pr_bwdRNN.disable_dropout()
        ################################
        self.xecfwdRNN.disable_dropout()
        self.xecbwdRNN.disable_dropout()
        self.xhcfwdRNN.disable_dropout()
        self.xhcbwdRNN.disable_dropout()
        self.xps_fwdRNN.disable_dropout()
        self.xps_bwdRNN.disable_dropout()
        self.xpr_fwdRNN.disable_dropout()
        self.xpr_bwdRNN.disable_dropout()

    def initialize_graph_nodes(self):
        #  convert parameters to expressions
        self.pad = dy.parameter(self.PAD)
        #################################
        self.xpad = dy.parameter(self.XPAD)

        self.ps_W1 = dy.parameter(self.ps_pW1)
        self.ps_b1 = dy.parameter(self.ps_pb1)
        self.ps_W2 = dy.parameter(self.ps_pW2)
        self.ps_b2 = dy.parameter(self.ps_pb2)
        self.pr_W1 = dy.parameter(self.pr_pW1)
        self.pr_b1 = dy.parameter(self.pr_pb1)
        self.pr_W2 = dy.parameter(self.pr_pW2)
        self.pr_b2 = dy.parameter(self.pr_pb2)
        #######################################
        self.xpad = dy.parameter(self.XPAD)
        self.xps_W1 = dy.parameter(self.xps_pW1)
        self.xps_b1 = dy.parameter(self.xps_pb1)
        self.xps_W2 = dy.parameter(self.xps_pW2)
        self.xps_b2 = dy.parameter(self.xps_pb2)
        self.xpr_W1 = dy.parameter(self.xpr_pW1)
        self.xpr_b1 = dy.parameter(self.xpr_pb1)
        self.xpr_W2 = dy.parameter(self.xpr_pW2)
        self.xpr_b2 = dy.parameter(self.xpr_pb2)

        # apply dropout
        if self.eval:
            self.disable_dropout()
        else:
            self.enable_dropout() 

        # initialize the RNNs
        self.f_init = self.fwdRNN.initial_state()
        self.b_init = self.bwdRNN.initial_state()

        self.cf_init_eng = self.ecfwdRNN.initial_state()
        self.cb_init_eng = self.ecbwdRNN.initial_state()
        self.cf_init_hin = self.hcfwdRNN.initial_state()
        self.cb_init_hin = self.hcbwdRNN.initial_state()
        ################################################
        self.xcf_init_eng = self.xecfwdRNN.initial_state()
        self.xcb_init_eng = self.xecbwdRNN.initial_state()
        self.xcf_init_hin = self.xhcfwdRNN.initial_state()
        self.xcb_init_hin = self.xhcbwdRNN.initial_state()

        self.ps_f_init = self.ps_fwdRNN.initial_state()
        self.ps_b_init = self.ps_bwdRNN.initial_state()
        ###############################################
        self.xps_f_init = self.xps_fwdRNN.initial_state()
        self.xps_b_init = self.xps_bwdRNN.initial_state()

        self.pr_f_init = self.pr_fwdRNN.initial_state()
        self.pr_b_init = self.pr_bwdRNN.initial_state()
        ###############################################
        self.xpr_f_init = self.xpr_fwdRNN.initial_state()
        self.xpr_b_init = self.xpr_bwdRNN.initial_state()

    def word_rep_eng(self, w):
        if not self.eval and random.random() < 0.3:
            return self.ELOOKUP_WORD[0]
        idx = self.meta.ew2i.get(w, self.meta.ew2i.get(w.lower(), 0))
        return self.ELOOKUP_WORD[idx]

    def word_rep_hin(self, w):
        if not self.eval and random.random() < 0.3:
            return self.HLOOKUP_WORD[0]
        idx = self.meta.hw2i.get(w, 0)
        return self.HLOOKUP_WORD[idx]

    def char_rep_eng(self, w, f, b):
        no_c_drop = False
        if self.eval or random.random()<0.9:
            no_c_drop = True
        bos, eos, unk = self.meta.ec2i["bos"], self.meta.ec2i["eos"], self.meta.ec2i['unk']
        char_ids = [bos] + [self.meta.ec2i.get(c, unk) if no_c_drop else unk for c in w] + [eos]
        char_embs = [self.ELOOKUP_CHAR[cid] for cid in char_ids]
        fw_exps = f.transduce(char_embs)
        bw_exps = b.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def char_rep_hin(self, w, f, b):
        no_c_drop = False
        if self.eval or random.random()<0.9:
            no_c_drop = True
        bos, eos, unk = self.meta.hc2i["bos"], self.meta.hc2i["eos"], self.meta.hc2i["unk"]
        char_ids = [bos] + [self.meta.hc2i.get(c, unk) if no_c_drop else unk for c in w] + [eos]
        char_embs = [self.HLOOKUP_CHAR[cid] for cid in char_ids]
        fw_exps = f.transduce(char_embs)
        bw_exps = b.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def get_char_embds(self, sentence, hf, hb, ef, eb):
        char_embs = []
        for node in sentence:
            if node.lang == 'hi':
                char_embs.append(self.char_rep_hin(node.form, hf, hb))
            elif node.lang == 'en':
                char_embs.append(self.char_rep_eng(node.form, ef, eb))
        return char_embs

    def get_word_embds(self, sentence):
        word_embs = []
        for node in sentence:
            if node.lang == 'hi':
                word_embs.append(self.word_rep_hin(node.form))
            elif node.lang == 'en':
                word_embs.append(self.word_rep_eng(node.form))
        return word_embs

    def basefeaturesEager(self, nodes, stack, i):
	#NOTE Stack nodes
        #s2 = nodes[stack[-3]] if stack[2:] else nodes[0].left
        #s1 = nodes[stack[-2]] if stack[1:] else nodes[0].left
        s0 = nodes[stack[-1]] if stack else nodes[0].left

	#NOTE Buffer nodes
	n0 = nodes[ i ] if nodes[ i: ] else nodes[0].left

	#NOTE Leftmost and Rightmost children of s2,s1,s0 and b0(only leftmost)
	#s2l = nodes[s2.left [-1]] if s2.left [-1] != None else nodes[0].left
	#s2r = nodes[s2.right[-1]] if s2.right[-1] != None else nodes[0].left
	#s1l = nodes[s1.left [-1]] if s1.left [-1] != None else nodes[0].left
	#s1r = nodes[s1.right[-1]] if s1.right[-1] != None else nodes[0].left
	#s0l = nodes[s0.left [-1]] if s0.left [-1] != None else nodes[0].left
	#s0r = nodes[s0.right[-1]] if s0.right[-1] != None else nodes[0].left
	#n0l = nodes[n0.left [-1]] if n0.left [-1] != None else nodes[0].left
	
	return [(nd.id, nd.form) for nd in s0,n0]
	
    def basefeaturesStandard(self, nodes, stack, i):
	#NOTE Stack nodes
        #s3 = nodes[stack[-4]] if stack[3:] else nodes[0].left
        #s2 = nodes[stack[-3]] if stack[2:] else nodes[0].left
        #s1 = nodes[stack[-2]] if stack[1:] else nodes[0].left
        s0 = nodes[stack[-1]] if stack else nodes[0].left

	#NOTE Buffer nodes
	n0 = nodes[ i ] if nodes[ i: ] else nodes[0].left
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
	
	return [(nd.id, nd.form) for nd in s0,n0]

    def feature_extraction(self, sentence):
        self.initialize_graph_nodes()

        # get word/char embeddings
        wembs = self.get_word_embds(sentence)
        cembs = self.get_char_embds(sentence, self.cf_init_hin, self.cb_init_hin, self.cf_init_eng, self.cb_init_eng)
        lembs = [dy.concatenate([w,c]) for w,c in zip(wembs, cembs)]

        # feed word vectors into base biLSTM
        fw_exps = self.f_init.transduce(lembs)
        bw_exps = self.b_init.transduce(reversed(lembs))
        bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]
    
        # feed biLSTM embeddings into POS biLSTM (pretrained)
        ps_fw_exps = self.ps_f_init.transduce(bi_exps)
        ps_bw_exps = self.ps_b_init.transduce(reversed(bi_exps))
        ps_bi_exps = [dy.concatenate([f,b]) for f,b in zip(ps_fw_exps, reversed(ps_bw_exps))]

        # get pos-hidden representation and pos loss
        pos_errs, pos_hidden = [], []
        for xi,node in zip(ps_bi_exps, sentence):
            xh = self.ps_W1 * xi
            #xh = self.meta.activation(xh) + self.ps_b1
            pos_hidden.append(xh)

        # get word/char embeddings (stacked)
        xcembs = self.get_char_embds(sentence, self.xcf_init_hin, self.xcb_init_hin,
                                               self.xcf_init_eng, self.xcb_init_eng)
        xwcp_exps = [dy.concatenate([w,c,p]) for w,c,p in zip(wembs, xcembs, pos_hidden)]
        xfw_exps = self.xps_f_init.transduce(xwcp_exps)
        xbw_exps = self.xps_b_init.transduce(reversed(xwcp_exps))
        xbi_exps = [dy.concatenate([f,b]) for f,b in zip(xfw_exps, reversed(xbw_exps))]

        pos_errs, xpos_hidden = [], []
        for xi,node in zip(xbi_exps, sentence):
            xh = self.xps_W1 * xi
            #xh = self.meta.activation(xh) + self.xps_b1
            xpos_hidden.append(xh)
            xo = self.xps_W2*xh + self.xps_b2
            tid = self.meta.p2i[node.tag]
            err = dy.softmax(xo).npvalue() if self.eval else dy.pickneglogsoftmax(xo, tid)
            pos_errs.append(err)
 
        # concatenate pos hidden-layer with base biLSTM 
        wcp_exps = [dy.concatenate([w,p]) for w,p in zip(bi_exps, pos_hidden)]
        # feed concatenated embeddings into parse biLSTM
        pr_fw_exps = self.pr_f_init.transduce(wcp_exps)
        pr_bw_exps = self.pr_b_init.transduce(reversed(wcp_exps))
        pr_bi_exps = [dy.concatenate([f,b]) for f,b in zip(pr_fw_exps, reversed(pr_bw_exps))]

        xwcp_exps = [dy.concatenate(list(z)) for z in zip(wembs, xcembs, pr_bi_exps, xpos_hidden)]
        xbi_fw_exps = self.xpr_f_init.transduce(xwcp_exps)
        xbi_bw_exps = self.xpr_b_init.transduce(reversed(xwcp_exps))
        xpr_bi_exps = [dy.concatenate([f,b]) for f,b in zip(xbi_fw_exps, reversed(xbi_bw_exps))]

        return pr_bi_exps, xpr_bi_exps, pos_errs

def Train(sentence, epoch, dynamic=True):
    loss = []
    totalError = 0
    parser.eval = False
    configuration = Configuration(sentence)
    pr_bi_exps, xpr_bi_exps, pos_errs = parser.feature_extraction(sentence[1:-1])
    while not parser.isFinalState(configuration):
        rfeatures = parser.basefeaturesEager(configuration.nodes, configuration.stack, configuration.b0)
        xi = dy.concatenate([pr_bi_exps[id-1] if id > 0 else parser.pad for id, rform in rfeatures])
        yi = dy.concatenate([xpr_bi_exps[id-1] if id > 0 else parser.pad for id, rform in rfeatures])
        xh = parser.pr_W1 * xi
        xi = dy.concatenate([yi, xh])
        xh = parser.xpr_W1 * xi
        xh = parser.meta.activation(xh) + parser.xpr_b1
        xo = parser.xpr_W2*xh + parser.xpr_b2
        output_probs = dy.softmax(xo).npvalue()
        ranked_actions = sorted(zip(output_probs, range(len(output_probs))), reverse=True)
        pscore, paction = ranked_actions[0]

        validTransitions, allmoves = parser.get_valid_transitions(configuration) #{0: <bound method arceager.SHIFT>}
        while parser.action_cost(configuration, parser.meta.i2td[paction], parser.meta.transitions, validTransitions) > 500:
           ranked_actions = ranked_actions[1:]
           pscore, paction = ranked_actions[0]

        gaction = None
        for i,(score, ltrans) in enumerate(ranked_actions):
           cost = parser.action_cost(configuration, parser.meta.i2td[ltrans], parser.meta.transitions, validTransitions)
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


def Test(test_file, ofp=None, lang=None):
    with io.open(test_file, encoding='utf-8') as fp:
        inputGenTest = re.finditer("(.*?)\n\n", fp.read(), re.S)

    parser.eval = True
    scores = defaultdict(int)
    good, bad = 0.0, 0.0
    for idx, sentence in enumerate(inputGenTest):
        graph = list(depenencyGraph(sentence.group(1), lang))
        pr_bi_exps, xpr_bi_exps, pos_errs = parser.feature_extraction(graph[1:-1])
	pred_pos = []
        for xo, node in zip(pos_errs, graph[1:-1]):
	    p_tag = parser.meta.i2p[np.argmax(xo)]
            pred_pos.append(p_tag)
            if node.tag == p_tag:
                good += 1
            else:
                bad += 1

        configuration = Configuration(graph)
        while not parser.isFinalState(configuration):
            rfeatures = parser.basefeaturesEager(configuration.nodes, configuration.stack, configuration.b0)
            xi = dy.concatenate([pr_bi_exps[id-1] if id > 0 else parser.pad for id, rform in rfeatures])
            yi = dy.concatenate([xpr_bi_exps[id-1] if id > 0 else parser.pad for id, rform in rfeatures])
            xh = parser.pr_W1 * xi
            xi = dy.concatenate([yi, xh])
            xh = parser.xpr_W1 * xi
            xh = parser.meta.activation(xh) + parser.xpr_b1
            xo = parser.xpr_W2*xh + parser.xpr_b2
            output_probs = dy.softmax(xo).npvalue()
    	    validTransitions, _ = parser.get_valid_transitions(configuration) #{0: <bound method arceager.SHIFT>}
    	    sortedPredictions = sorted(zip(output_probs, range(len(output_probs))), reverse=True)
    	    for score, action in sortedPredictions:
    	        transition, predictedLabel = parser.meta.i2td[action]
    	        if parser.meta.transitions[transition] in validTransitions:
    	            predictedTransitionFunc = validTransitions[parser.meta.transitions[transition]]
    	            predictedTransitionFunc(configuration, predictedLabel)
    	            break
        dgraph = deprojectivize(graph[1:-1])
	scores = tree_eval(dgraph, scores)
        #sys.stderr.write("Testing Instances:: %s\r"%idx)
    sys.stderr.write('\n')

    UAS = round(100. * scores['rightAttach']/(scores['rightAttach']+scores['wrongAttach']),2)
    LS  = round(100. * scores['rightLabel']/(scores['rightLabel']+scores['wrongLabel']), 2)
    LAS = round(100. * scores['rightLabeledAttach']/(scores['rightLabeledAttach']+scores['wrongLabeledAttach']),2)
    
    return good/(good+bad), UAS, LS, LAS

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
        POS, UAS, LS, LAS = Test(args.cdev)
        sys.stderr.write("\nCM POS ACCURACY: {}% UAS: {}%, LS: {}% and LAS: {}%\n".format(POS, UAS, LS, LAS))
	sys.stderr.flush()
        if LAS > psc:
            sys.stderr.write('SAVE POINT %d\n' %epoch)
            psc = LAS
            if args.save_model:
                parser.model.save('%s.dy' %args.save_model)


def projective(nodes):
    """Identifies if a tree is non-projective or not."""
    for leaf1 in nodes:
        v1,v2 = sorted([int(leaf1.id), int(leaf1.parent)])
        for leaf2 in nodes:
            v3, v4 = sorted([int(leaf2.id), int(leaf2.parent)])
            if leaf1.id == leaf2.id:continue
            if (v1 < v3 < v2) and (v4 > v2): return False
    return True

def depenencyGraph(sentence, lang=None):
    """Representation for dependency trees"""
    leaf = namedtuple('leaf', ['id','form','lemma','tag','ctag','lang','parent','pparent', 'drel','pdrel','left','right', 'visit'])
    PAD = leaf._make([-1,'__PAD__','__PAD__','__PAD__','__PAD__',defaultdict(lambda:'__PAD__'),-1,-1,'__PAD__','__PAD__',[None],[None], False])
    yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', defaultdict(str), -1, -1, '__ROOT__', '__ROOT__', PAD, [None], False])

    for node in sentence.split("\n"):
        if lang:
            id_,form,lemma,tag,ctag,_,parent,drel = node.split("\t")[:8]
            tlang = lang
        else:
            id_,lemma,form,tag,ctag,_,parent,drel,tlang = node.split("\t")[:9]
            tlang = tlang.split('|')[0]
            if tlang != 'hi':
                tlang = 'en'
        if ':' in drel and drel != 'acl:relcl':
            drel = drel.split(':')[0]
        if drel == 'obl':
            drel = 'nmod'
        node = leaf._make([int(id_),form,lemma,tag,ctag,tlang,int(parent),-1,drel,drel,[None],[None], False])
        yield node
    yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', defaultdict(str), -1, -1, '__ROOT__', '__ROOT__', [None], [None], False])


def read(fname, lang=None):
    with io.open(fname, encoding='utf-8') as fp:
        inputGenTrain = re.finditer("(.*?)\n\n", fp.read(), re.S)

    data = []
    for i,sentence in enumerate(inputGenTrain):
        graph = list(depenencyGraph(sentence.group(1), lang))
        try:
            pgraph = graph[:1]+projectivize(graph[1:-1])+graph[-1:]
        except:
            sys.stderr.write('Error Sent :: %d\n' %i)
            #print(sentence.group(1))
            sys.stdout.flush()
            continue
        data.append(pgraph)
    return data


def set_class_map(data):
      for graph in data:
          for pnode in graph[1:-1]:
              plabels.add(pnode.tag)
              if pnode.parent == 0:
                  tdlabels.add(('LEFTARC', pnode.drel))
              elif pnode.id < pnode.parent:
                  tdlabels.add(('LEFTARC', pnode.drel))
              else:
                  tdlabels.add(('RIGHTARC', pnode.drel))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Neural Network Parser.", description="Bi-LSTM Parser")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dynet-gpu')
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-devices')
    parser.add_argument('--dynet-autobatch')
    parser.add_argument('--dynet-seed', dest='seed', type=int, default='127')
    parser.add_argument('--ctrain', help='Hindi-English CS CONLL Train file')
    parser.add_argument('--cdev', help='Hindi-English CS CONLL Dev/Test file')
    parser.add_argument('--trainer', default='momsgd', help='Trainer [momsgd|adam|adadelta|adagrad]')
    parser.add_argument('--activation-fn', dest='act_fn', default='tanh', help='Activation function [tanh|rectify|logistic]')
    parser.add_argument('--ud', type=int, default=1, help='1 if UD treebank else 0')
    parser.add_argument('--iter', type=int, default=100, help='No. of Epochs')
    parser.add_argument('--bvec', type=int, help='1 if binary embedding file else 0')
    group.add_argument('--save-model', dest='save_model', help='Specify path to save model')
    group.add_argument('--load-model', dest='load_model', help='Load Pretrained Model')
    parser.add_argument('--base-model', dest='base_model', help='build a stacking model on this pretrained model')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cdev:
        cdev = read(args.cdev)
    if not args.load_model:
        xmeta = Meta()
        meta = pickle.load(open('%s.meta' %args.base_model, 'rb'))
        plabels = set()
        tdlabels = set()
        tdlabels.add(('SHIFT', None))
        tdlabels.add(('REDUCE', None))

        train_sents = read(args.ctrain)
        set_class_map(train_sents)
        meta.i2p = dict(enumerate(plabels))
        meta.i2td = dict(enumerate(tdlabels))
        meta.p2i = {v: k for k,v in meta.i2p.iteritems()}
        meta.td2i = {v: k for k,v in meta.i2td.iteritems()}
        meta.xn_outs = len(meta.i2td)
        meta.xn_tags = len(meta.p2i)
        meta.xc_dim = xmeta.xc_dim
        meta.xp_hidden = xmeta.xp_hidden
        meta.xn_hidden = xmeta.xn_hidden
        meta.xlstm_wc_dim = xmeta.xlstm_wc_dim
        meta.xlstm_char_dim = xmeta.xlstm_char_dim

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
        parser = Parser(model=args.load_model, test=True)
        sys.stderr.write('Done!\n')
        POS, UAS, LS, LAS = Test(args.cdev)
        sys.stderr.write("TEST-SET POS: {}%, UAS: {}%, LS: {}% and LAS: {}%\n".format(POS, UAS, LS, LAS))
    elif args.base_model:
        parser = Parser(model=args.base_model, new_meta=meta)
        trainer = meta.trainer(parser.model)
        train_parser(train_sents)
