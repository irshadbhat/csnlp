#!/usr/bin/env python

import sys
import copy

from collections import defaultdict, OrderedDict, namedtuple

def is_projective(graph):
   proj=True
   edges = int()
   spans = set()
   for node in graph:
      s = tuple(sorted([node.id, node.parent]))
      spans.add(s)
   for l,h in sorted(spans):
      for l1,h1 in sorted(spans):
         if (l,h)==(l1,h1): continue
         if l < l1 < h and h1 > h:
            #print "non proj:",(l,h),(l1,h1)
            edges += 1
            proj = False
   return edges

def make_tree(treelets, node):
    tree = dict()
    tree.setdefault(node,{})
    if treelets[node] != []:
        for child in treelets[node]:
            tree[node].update(make_tree(treelets, child))
    return tree

def InOrder(tree, order):
    root = tree.keys()[0]
    children = OrderedDict(sorted(tree[root].items()))
    left_sub_trees = [{x:children[x]} for x in children if x < root]
    right_sub_trees =  [{x:children[x]} for x in children if x > root]
    for sub_tree in left_sub_trees: # Traverse the left subtrees 
        if not sub_tree[sub_tree.keys()[0]]:
            order.append(sub_tree.keys()[0])
            continue
        InOrder(sub_tree, order)
    order.append(root) # Visit Root
    for sub_tree in right_sub_trees: # Traverse the right subtree
        if not sub_tree[sub_tree.keys()[0]]:
            order.append(sub_tree.keys()[0])
            continue
        InOrder(sub_tree, order)
    return order

def get_in_order(graph):
    roots = [node.parent for node in graph]
    treelets = defaultdict(list)
    for i,node in enumerate(roots):
        treelets[int(node)].append(i+1)
    tree = make_tree(treelets, 0)
    inOrder = InOrder(tree[0], [0])
    return sorted([graph[ino-1]._replace(inorder=idx) for idx,ino in enumerate(inOrder[1:],1)], key=lambda x:x.id)
    #return [node._replace(inorder=idx) for node,idx in zip(graph, inOrder[1:])]

def depenencyGraph(sentence):
    """Representation for dependency trees"""
    leaf = namedtuple('leaf', ['id','form','lemma','tag','ctag','features','parent','pparent', 'drel','pdrel','inorder','left','right'])
    PAD = leaf._make([-1,'__PAD__','__PAD__','__PAD__','__PAD__',defaultdict(lambda:'__PAD__'),-1,-1,'__PAD__','__PAD__',-1,[None],[None]])
    yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', defaultdict(str), -1, -1, '__ROOT__', '__ROOT__', -1,PAD, [None]])

    for node in sentence.split("\n"):
        id_,form,lemma,ctag,tag,features,parent,drel = node.split("\t")[:-2]
        node = leaf._make([int(id_),form,lemma,tag,ctag,features,int(parent),-1,drel,'__PAD__',-1, [None],[None]])
        features = defaultdict(str)
        for feature in node.features.split("|"):
            try:
                features[feature.split("-")[0]] = feature.split("-")[1]
            except IndexError: features[feature.split("-")[0]] = ''
        yield node._replace(features=features)
    yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_P', 'ROOT_C', defaultdict(str), -1, -1, '__ROOT__', '__ROOT__', -1, [None], [None]])


if __name__ == "__main__":

	with open(sys.argv[1]) as inp:
		sentences = inp.read().strip().split("\n\n")

	for sent in sentences:
		graph = list(depenencyGraph(sent))[1:-1]
		inorder = get_in_order(graph)
		for ind in inorder:print "\t".join(map(str, (ind.id,ind.form,ind.parent,ind.inorder)))
		print
