#!/usr/bin/env python

import re
import sys

from collections import namedtuple as nt, defaultdict as dfd

def eval_parses(gold, system, ignore_punct=False):

    rightAttach = wrongAttach = 0.0
    rightLabel = wrongLabel = 0.0
    rightLabeledAttach = wrongLabeledAttach = 0.0

    for man,auto in zip(gold, system):
	man, auto = list(man)[1:-1],list(auto)[1:-1]
        for node_man,node_auto in zip(man,auto):
	    if ignore_punct and node_man.tag == "PUNCT":continue
            if node_man.parent == node_auto.parent:
                rightAttach += 1
                if node_man.drel.split(":")[0] == node_auto.drel.split(":")[0]:
                    rightLabeledAttach += 1
                else:
                    wrongLabeledAttach += 1
            else:
                wrongAttach += 1
                wrongLabeledAttach += 1
            if node_man.drel.split(":")[0] == node_auto.drel.split(":")[0]:
                rightLabel += 1
            else: 
                wrongLabel += 1
    
    return "UAS: {}%, LS: {}% and LAS: {}%".format(
             round(100 * rightAttach/(rightAttach+wrongAttach),2), 
             round(100 * rightLabel/(rightLabel+wrongLabel), 2),
             round(100 * rightLabeledAttach/(rightLabeledAttach+wrongLabeledAttach),2)
             )

def depenencyGraph(sentence):
        leaf = nt('leaf', ['id','form','lemma','ctag','tag','features','parent','pparent', 'drel','pdrel','left','right'])
        yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_C', 'ROOT_P', dfd(str), -1, -1, '_', '_', [], []])
        for node in sentence.split("\n"):
                id_,form,lemma,ctag,tag,features,parent,drel = node.split("\t")[:-2]
                node = leaf._make([int(id_),form,lemma,ctag,tag,features,int(parent),-1, drel,'__PAD__',[],[]])
                features = dfd(str)
                for feature in node.features.split("|"):
                        try:
                                features[feature.split("-")[0]] = feature.split("-")[1]
                        except IndexError: features[feature.split("-")[0]] = ''
                yield node._replace(features=features)
        yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_C', 'ROOT_P', dfd(str), -1, -1, '_', '_', [], []])

def main(gold, system, ignore_punct):
    with open(gold) as inpg, open(system) as inps:
        goldSents = [depenencyGraph(gs.group(1)) for gs in re.finditer("(.*?)\n\n", inpg.read(), re.S)]
        systemSents = [depenencyGraph(ss.group(1)) for ss in re.finditer("(.*?)\n\n", inps.read(), re.S)]
    assert len(systemSents) == len(goldSents)
    accuracy = eval_parses(goldSents, systemSents, ignore_punct)
    return accuracy

if __name__ == '__main__':

    import argparse as arg

    parser = arg.ArgumentParser(description="Dependecy Parsing Evaluation.")
    parser.add_argument('--gold-file', dest='gold'   , required=True,  help='Gold parse trees.')
    parser.add_argument('--system-file' , dest='system' , required=True, help='Autmatically parsed trees.')
    parser.add_argument('--ignore-punct' , dest='punct' , action='store_true', help='Ignore scoring punctuations.')

    args = parser.parse_args()
    print "\tHere are the accuracies -> {}".format(main(args.gold, args.system, args.punct))
