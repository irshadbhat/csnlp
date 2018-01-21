#!/usr/bin/env python -*- coding: utf-8 -*-

import re
import sys
import numpy as np

"""
Implementation of tree transformation algorithms for handling non-projective trees in transition based systems.
The procedure is defined in Joakim Nivre and Jens Nilsson, ACL 2005. http://stp.lingfil.uu.se/~nivre/docs/acl05.pdf
"""

def get_projection(node, adMat):
    children = list()
    #if np.all(adMat[node]==0):NOTE slower
    if not len(np.nonzero(adMat[node])[0]): # if no nonzero in the array
        return children
    #for c in np.where(adMat[node]!=0)[0]:NOTE slower
    #children += list(np.nonzero(adMat[node])[0])
    for c in np.nonzero(adMat[node])[0]:
        children += get_projection(c,adMat)
        children.append(c+1) # +1 -> tally with root nodes
    return children

def adjacency_matrix(nodes, training=True):
    """Builds an adjacency matrix of a dependency graph"""
    adMat = np.zeros((len(nodes), len(nodes)), int)
    for node in nodes:
	if training:
            if (node.parent == 0):continue
            parent, child = node.parent - 1, node.id - 1 # -1 -> tally with list indices
	else:
            if (node.pparent == 0):continue
            parent, child = node.pparent - 1, node.id - 1 # -1 -> tally with list indices
        adMat[parent, child] = 1
    return adMat

def non_projectivity(nodes, tree=None):
    """Extracts non-projective arcs from a given tree, if any."""
    #if not tree: tree = adjacency_matrix(nodes)
    np_arcs = set()
    for leaf in sorted(nodes):
        if leaf.parent == 0: continue # no node can interfer in the root to dummy root arc.
        head, dependent = leaf.parent, leaf.id
        projection = set(get_projection(head-1, tree)) # -1 -> tally with array indices
        for inter in range(head+1, dependent) if head < dependent else range(dependent+1 , head): # +1 -> ignore nodes in the arc
            if (head < inter < dependent or head > inter > dependent) and inter in projection:continue
            else:np_arcs.add((dependent, head, abs(dependent-head)))
    return np_arcs

def projectivize(nodes):
    """PseudoProjectivisation: Lift non-projective arcs by moving their head upwards one step at a time"""
    tree = adjacency_matrix(nodes,True)
    non_projective_arcs = sorted(non_projectivity(nodes, tree), key=lambda x:x[-1]) #sorted np arcs by distance.
    while non_projective_arcs:
        dependent, head, distance = non_projective_arcs.pop(0)
        npDepNode = nodes[dependent-1]
        npHeadNode = nodes[head-1] # syntacticHead
        modifieddrel = npDepNode.drel if npDepNode.visit else re.sub(r"(%|$)",r'|%s\1' % (npHeadNode.pdrel),npDepNode.drel)
        nodes[npDepNode.id-1] = nodes[npDepNode.id-1]._replace(drel=modifieddrel, parent=npHeadNode.parent, visit=True)
        nodes[npHeadNode.id-1] = nodes[npHeadNode.id-1]._replace(drel=re.sub(r"[%]*$",r'%',npHeadNode.drel))    
        tree = adjacency_matrix(nodes,True)
        non_projective_arcs = sorted(non_projectivity(nodes, tree), key=lambda x:x[-1])
    #return nodes
    return [node._replace(pparent=-1,pdrel='__PAD__') for node in nodes]
    #return [node._replace(pparent=node.parent,pdrel=node.drel) for node in nodes]

def ulParent(nodes, stack, linearHeadLabel):
    while stack:
        imdParent = stack.pop()
        if linearHeadLabel.strip("%")==nodes[imdParent].pdrel.split("|")[0].strip("%"):
            syntacticHead = imdParent
            return syntacticHead
    return 0

def BSF(nodes, tree, linearHead, linearHeadLabel, node):
    """Breadth First Search to locate syntactic head of a non-projective node."""
    #TODO bit messy, improve!
    syntacticHead = linearHead
    adjList = np.nonzero(tree[linearHead])[0]
    queue = [j for j in adjList if re.search(r"%", nodes[j].pdrel) and node != j] #NOTE original tree
    #queue = [j for j in adjList if re.search(r"%", nodes[j].copy) and node.id != j] # Note tree re-oriented
    stack = []
    while queue:
        queueNode = queue.pop(0)
        if queueNode == node:continue
        if linearHeadLabel.strip("%")==nodes[queueNode].pdrel.split("|")[0].strip("%"):
            lookDownQueueNode = [j for j in np.nonzero(tree[queueNode])[0] if re.search(r"%", nodes[j].pdrel)]
            #lookDownQueueNode = [j for j in np.nonzero(tree[queueNode])[0] if re.search(r"%", nodes[j].copy)]
            if (lookDownQueueNode == []):
                syntacticHead = queueNode
                break
            else:
                queue.extend([j for j in np.nonzero(tree[queueNode])[0] if re.search(r"%", nodes[j].pdrel)])
                #queue.extend([j for j in np.nonzero(tree[queueNode])[0] if re.search(r"%", nodes[j].copy)])
        else:
            adjList = [j for j in np.nonzero(tree[queueNode])[0] if re.search(r"%", nodes[j].pdrel)]
            if queue == [] and adjList == []: 
                _head = ulParent(nodes, stack, linearHeadLabel)
                if _head:syntacticHead = _head
            queue.extend([j for j in np.nonzero(tree[queueNode])[0] if re.search(r"%", nodes[j].pdrel)])
            #queue.extend([j for j in np.nonzero(tree[queueNode])[0] if re.search(r"%", nodes[j].copy)])
        stack.append(queueNode)
    return syntacticHead

def deprojectivize(nodes, scheme="head+path"):
    """PseudoProjectivisation: Reverse transformation of pseudoProjective arcs into non-projective arcs using BFS."""
    tree = adjacency_matrix(nodes, training=False)
    solutions = dict()
    for nC in range(0,len(nodes)):
        node = nodes[nC]
        parent, child, drel = node.pparent, node.id, node.pdrel
        if re.search(r"\|", drel):
            syntacticLabel,linearHeadLabel,linearHead = drel.split("|") + [parent-1]
            syntacticLabel = syntacticLabel + "%" if re.search(r"%", drel) else syntacticLabel

            syntacticHead = BSF(nodes, tree, linearHead, linearHeadLabel, node.id-1) # -1 -> to tally
            #if nodes[linearHead].parent is 0 -> lifting is undefined as per the Nivre and Nelson.
            if syntacticHead == linearHead and nodes[linearHead].pparent: # parent should be other the dummy root i.e. > 0
                syntacticHead = BSF(nodes,tree,nodes[linearHead].pparent-1,linearHeadLabel,linearHead) # -1 -> tally
            nodes[nC] = nodes[nC]._replace(pparent=syntacticHead + 1, pdrel = syntacticLabel)
            #nodes[nC].drel = syntacticLabel
            tree = adjacency_matrix(nodes, training=False)
    return nodes
