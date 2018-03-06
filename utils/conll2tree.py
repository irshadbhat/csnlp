#!/usr/bin/env python

def flatten_tree(tree):
        flatTree = []
        for key, value in tree.iteritems():
                flatTree.append(key)
                flatTree.extend(flatten_tree(value))
        return flatTree

def treeTraversal(tree, node):
        if node in tree:
                return tree[node]
        else:
                for key, value in tree.items():
                        if not value:continue
                        output = treeTraversal(value, node)
                        if output:return output
        return {}

def buildTree(treelets, node):
        tree = {}
        tree.setdefault(node,{})
        if treelets[node] != []:
                for child in treelets[node]:
                        tree[node].update(buildTree(treelets, child))
        return tree

