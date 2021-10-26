"""
COMS W4705 - Natural Language Processing
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        n = len(tokens)
        table = defaultdict(set)

        for i in range(0, n):
            items = self.grammar.rhs_to_rules[(tokens[i],)]
            for m in items:
                table[(i, i + 1)].add(m[0])

        for l in range(2, n + 1):
            for i in range(0, n - l + 1):
                j = i + l
                for k in range(i + 1, j):
                    # TODO: Can we use itertools to simplify?
                    for p in table[(i, k)]:
                        for q in table[(k, j)]:
                            if (p, q) in list(self.grammar.rhs_to_rules.keys()):
                                items = self.grammar.rhs_to_rules[(p, q)]
                                for m in items:
                                    table[(i, j)].add(m[0])

        if self.grammar.startsymbol in table[(0, n)]:
            return True

        return False
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table= defaultdict(dict)
        probs = defaultdict(dict)

        n = len(tokens)
        for i in range(0, n):
            items = self.grammar.rhs_to_rules[(tokens[i],)]
            table[(i, i + 1)] = defaultdict(str)
            probs[(i, i + 1)] = defaultdict(float)
            for m in items:
                table[(i, i + 1)][m[0]] = tokens[i]
                probs[(i, i + 1)][m[0]] = math.log(m[2])

        for l in range(2, n + 1):
            for i in range(0, n - l + 1):
                # initialize table
                j = i + l
                # iterate through k
                for k in range(i + 1, j):
                    # TODO: Can we use itertools to simplify?
                    for p in list(table[(i, k)].keys()):
                        for q in list(table[(k, j)].keys()):
                            if (p, q) in list(self.grammar.rhs_to_rules.keys()):
                                # initialize table
                                if table[(i, j)] is None:
                                    table[(i, j)] = defaultdict(tuple)
                                if probs[(i, j)] is None:
                                    probs[(i, j)] = defaultdict(float)

                                # fill table
                                items = self.grammar.rhs_to_rules[(p, q)]
                                for m in items:
                                    tmpp = float(probs[(i, k)][p]) + float(probs[(k, j)][q]) + math.log(m[2])
                                    if m[0] not in table[(i, j)] or tmpp > probs[(i, j)][m[0]]:
                                        table[(i, j)][m[0]] = ((p, i, k), (q, k, j))
                                        probs[(i, j)][m[0]] = tmpp

        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if i + 1 == j:
        return (nt, chart[i,j][nt])

    c1 = chart[(i,j)][nt][0]
    c2 = chart[(i,j)][nt][1]

    return (nt, get_tree(chart, c1[1], c1[2], c1[0]), get_tree(chart, c2[1], c2[2], c2[0]))


if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)