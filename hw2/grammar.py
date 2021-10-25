"""
COMS W4705 - Natural Language Processing
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """

        for rules in list(self.lhs_to_rules.values()):
            list_p = []
            for r in rules:
                # check CNF
                leaves = r[1]
                if len(leaves) == 1:
                    if leaves[0] == '0' or leaves[0] == '.':
                        continue
                    if leaves[0].islower() is False:
                        return False
                elif len(leaves) == 2:
                    for s in leaves:
                        if s.isupper() is False:
                            return False
                else:
                    return False

                list_p.append(r[2])

            # check probability
            psum = fsum(list_p)
            if psum == 1:
                return True

        return False 


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)

    if grammar.verify_grammar():
        print("Grammar is valid.")
    else:
        print("Grammar is invalid.")
