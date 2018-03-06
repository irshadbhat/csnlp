#!/usr/env python
# Copyright Riyaz Ahmad 2015 - present
#
# This file is part of IL-Parser library.
# 
# IL-Parser library is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# IL-Parser Library is distributed in the hope that it will be useful,
#        but WITHOUT ANY WARRANTY; without even the implied warranty of
#        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#        GNU General Public License for more details.
# 
#        You should have received a copy of the GNU General Public License
#        along with IL-Parser Library.  If not, see <http://www.gnu.org/licenses/>.
#

# Program for dependency parsing  
#
# @author Riyaz Ahmad
#

import numpy as np

class Swap(object):
    
    def SWAP(self, configuration, label=None):
        """
        Inverts top two nodes on the stack.
        """
	s0 = configuration.stack.pop() #j
	s1 = configuration.stack.pop() #i
        configuration.stack.append(s0)
	configuration.queue.insert(0,s1)
        configuration.b0 = s1

    def SHIFT(self, configuration, label=None):
        """
        Moves the input from buffer to stack.
        """
	b0 = configuration.queue.pop(0)
        configuration.stack.append(b0)
        #if configuration.nodes[b0].ptag == '__PAD__':
	#    configuration.nodes[b0] = configuration.nodes[b0]._replace(ptag=label)

	configuration.b0 = configuration.queue[0] if configuration.queue[0:] else b0+1

    def RIGHTARC(self, configuration, label=None):
        """
        Right reduces the top two tokens at the stack. s1 -> s0
        """
	#b0 = configuration.b0
        s1 = configuration.stack[-2]
        s0 = configuration.stack.pop(-1)
        s0N = configuration.nodes[s0]
        s1N = configuration.nodes[s1]

	configuration.nodes[s0] = configuration.nodes[s0]._replace(pparent=s1N.id, pdrel=label)
	configuration.nodes[s1] = configuration.nodes[s1]._replace(right=configuration.nodes[s1].right+[s0])
        #configuration.stack.append(b0)
	#configuration.b0 = b0+1

    def LEFTARC(self, configuration, label=None):
        """
        Left reduces the top two tokens at the stack. s0 -> s1
        """
	#b0 = configuration.b0
        s0 = configuration.stack[-1]
        s1 = configuration.stack.pop(-2)
        s0N = configuration.nodes[s0]
        s1N = configuration.nodes[s1]
	configuration.nodes[s1] = configuration.nodes[s1]._replace(pparent=s0N.id, pdrel=label)
	configuration.nodes[s0] = configuration.nodes[s0]._replace(left=configuration.nodes[s0].left+[s1])

    def inFinalState(self, configuration):
        """
        Checks if the parser is in final configuration i.e. all the input is 
	consumed and both the stack and queue are empty.
        """
	#return (len(configuration.stack) == 1) and (len(configuration.nodes[b0:]) == 0)
	return (len(configuration.stack) == 1) and (configuration.queue == [])

    def get_valid_transitions(self, configuration):
	moves = dict()
        allmoves = moves.copy() 
	b0 = configuration.b0
	
	#if configuration.nodes[b0:]: moves[0] = self.SHIFT
	if configuration.queue: moves[0] = self.SHIFT
	if len(configuration.stack) > 1:
		s0 = configuration.stack[-1] #j
		s1 = configuration.stack[-2] #i
		moves[1] = self.LEFTARC
		moves[2] = self.RIGHTARC
		#if (configuration.nodes[s0].id != 0) and (s1 < s0): moves[3] = self.SWAP
		if (s1 < s0): moves[3] = self.SWAP
	return moves, moves

    def LabelledAction(self, configuration):
        if len(configuration.stack) < 2:
                #b0 = configuration.queue[0]
                b0 = configuration.b0
                return self.SHIFT, None #configuration.nodes[b0].tag
        else:
		b0Nmpc = configuration.nodes[configuration.queue[0]].mpc if configuration.queue else None
		s0 = configuration.stack[-1] #j
		s1 = configuration.stack[-2] #i
                s0N = configuration.nodes[s0]
                s1N = configuration.nodes[s1] 
                if (s1N.parent == s0N.id) and not (set(s1N.children) - set(s1N.left+s1N.right)): return self.LEFTARC, s1N.drel
                elif (s1N.id == s0N.parent) and not (set(s0N.children) - set(s0N.left+s0N.right)): return self.RIGHTARC, s0N.drel
		elif (s0N.inorder < s1N.inorder) and (s0N.mpc != b0Nmpc): return self.SWAP, None
                else:return self.SHIFT, None #configuration.nodes[configuration.b0].tag
