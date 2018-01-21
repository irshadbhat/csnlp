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

class ArcEager(object):
    
    def SHIFT(self, configuration, label=None):
        """
        Moves the input from buffer to stack.
        """
	b0 = configuration.b0
        configuration.stack.append(b0)
        configuration.b0 = b0+1

    def RIGHTARC(self, configuration, label=None):
        """
        Right reduces the tokens at the buffer and stack. s0 -> b0
        """
	b0 = configuration.b0
        s0 = configuration.stack[-1]
        s0N = configuration.nodes[s0]
        b0N = configuration.nodes[b0]

	configuration.nodes[b0] = configuration.nodes[b0]._replace(pparent=s0N.id, pdrel=label)
        if b0 < s0:
	    configuration.nodes[s0] = configuration.nodes[s0]._replace(left=configuration.nodes[s0].left+[b0])
        else:
	    configuration.nodes[s0] = configuration.nodes[s0]._replace(right=configuration.nodes[s0].right+[b0])
        configuration.stack.append(b0)
	configuration.b0 = b0+1

    def LEFTARC(self, configuration, label=None):
        """
        Left reduces the tokens at the stack and buffer. b0 -> s0
        """
	b0 = configuration.b0
        s0 = configuration.stack.pop()
        s0N = configuration.nodes[s0]
        b0N = configuration.nodes[b0]
	configuration.nodes[s0] = configuration.nodes[s0]._replace(pparent=b0N.id, pdrel=label)
        if s0 < b0:
	    configuration.nodes[b0] = configuration.nodes[b0]._replace(left=configuration.nodes[b0].left+[s0])
        else:
	    configuration.nodes[b0] = configuration.nodes[b0]._replace(right=configuration.nodes[b0].right+[s0])

    def REDUCE(self, configuration, label=None):
        """
        pops the top of the stack if it has got its head.
        """
	b0 = configuration.b0
        configuration.stack.pop()

    def isFinalState(self, configuration):
        """
        Checks if the parser is in final configuration i.e. all the input is 
	consumed and both the stack and queue are empty.
        """
	b0 = configuration.b0
	return (len(configuration.stack) == 0) and (len(configuration.nodes[b0:]) == 1)

    def get_valid_transitions(self, configuration):
        moves = {0:self.SHIFT,1:self.LEFTARC,2:self.RIGHTARC,3:self.REDUCE}
        allmoves = {0:self.SHIFT,1:self.LEFTARC,2:self.RIGHTARC,3:self.REDUCE}
	b0 = configuration.b0
        if len(configuration.nodes[b0:])==1:
            assert(configuration.nodes[b0].id == 0)
            del moves[0]
            del moves[2]

        if len(configuration.stack) == 0: del moves[3]
        elif (configuration.nodes[b0:]):
            if configuration.nodes[configuration.stack[-1]].pparent == -1: del moves[3]

        if len(configuration.stack) < 1:
            del moves[1]
            del moves[2]
        else:
            if configuration.nodes[configuration.stack[-1]].pparent > -1: del moves[1] #['LEFTARC'] # if s0 has parent no LEFT ARC
            if configuration.nodes[b0].pparent > -1: del moves[2] #['RIGHTARC'] # b0 has parent no RIGHT ARC unnecessary condition
        return moves, allmoves

    def predict(self, configuration):
        if not configuration.stack:
                return self.SHIFT, None
        else:
                s0 = configuration.nodes[configuration.stack[-1]]
                b0 = configuration.nodes[configuration.b0]
                if s0.parent == b0.id: return self.LEFTARC, s0.drel
                elif s0.id == b0.parent: return self.RIGHTARC, b0.drel
                elif self.dependencyLink(configuration): return self.REDUCE, None
                else: return self.SHIFT, None

    def dependencyLink(self, configuration):
        """
        Resolves ambiguity between shift and reduce actions.
        if a dependency exits between any node (<s0) and (b0) then reduce else shift.
        """
        for Sx in configuration.stack[:-1]:
                Sx = configuration.nodes[Sx]
                b0 = configuration.nodes[configuration.b0]
                if (Sx.parent == b0.id) or (Sx.id == b0.parent): return True
        return False

    def action_cost(self, configuration, labeled_transition, transitions, valid_transitions):
       stack, nodes, b0 = configuration.stack, configuration.nodes, configuration.b0
       transition, label = labeled_transition

       if transitions[transition] not in valid_transitions: return 1000

       lost = 0
       if transition == 'SHIFT':
          # b0 can no longer have children or parents on stack

          for s in stack:
             if nodes[s].parent == b0 and nodes[s].pparent == -1:
                lost += 1
             if nodes[b0].parent == s:
                if s != 0: # if real parent is ROOT and is on stack,
                           # we will get it by post-proc at the end.
                   lost += 1

       elif transition == 'REDUCE':
          # s0 can no longer have deps on buffer

	  s0 = stack[-1]
          for tok in nodes[b0:]:
             if tok.parent == s0:
                lost += 1

       elif transition == 'LEFTARC':
          # s0 can no longer have deps on buffer
          # s0 can no longer have parents on buffer[1:]
		
	  s0 = stack[-1]
          for (idx, tok) in enumerate(nodes[b0:]):
             if nodes[s0].parent == tok.id:
                if (idx > 0):
                   lost += 1
                elif nodes[s0].drel != label:
                   lost += 1
             if tok.parent == s0:
                lost += 1

       elif transition == 'RIGHTARC':
          # b0 can no longer have parents in stack[:-1]
          # b0 can no longer have deps in stack
          # b0 can no longer have parents in buffer

	  s0 = stack[-1]
          b0parent = nodes[b0].parent
          for s in stack:
             if nodes[s].parent == b0 and nodes[s].pparent == -1:
                lost += 1
             if (b0parent == s):
                if s != s0:
                   lost += 1
                elif nodes[b0].drel != label:
                   lost += 1
          # If root-at-end representation, lose the correct parent of b0 if it is root.
          if b0parent > b0 or (b0parent == 0 and stack and stack[0] != 0):
             lost += 1

       else:
          assert(False), ("Invalid action", transition)

       return lost
