import sys
import math
import random
import itertools
from collections import namedtuple, defaultdict, Counter
import json
import csv
from nltk import ParentedTree
from nltk import Tree

import tqdm

Rule = namedtuple("Rule", ['lhs', 'rhs'])

ROOT = 'ROOT'
NONE = '-NONE-'
TERMINAL_MARKER = '_'

### TESTING ###
def save_chart_to_json(chart, filename):
    with open(filename, "w") as f:
        json.dump(chart, f, indent=4)

def save_chart_to_txt(chart, filename):
    with open(filename, "w") as f:
        T = len(chart)
        for i in range(T):
            for j in range(i, T):
                f.write(f"Span ({i}, {j}):\n")
                for nt, prob in chart[i][j].items():
                    f.write(f"  {nt}: {prob:.6f}\n")
                f.write("\n")  # Add space for readability

def extract_tree(backpointers, i, j, nt):
    """ Recursively reconstruct the best parse tree in parentheses notation """
    if i == j:  # Base case: Single word (terminal)
        return f"({nt} {backpointers[i][j][nt]})"
    
    if nt not in backpointers[i][j]:  # No parse for this non-terminal
        return ""
    
    B, C, k = backpointers[i][j][nt]
    left_subtree = extract_tree(backpointers, i, k, B)
    right_subtree = extract_tree(backpointers, k+1, j, C)
    
    return f"({nt} {left_subtree} {right_subtree})"

def deepest_cnf_node(p_tree): 
    for pos in reversed(p_tree.treepositions()): # want to start with the deepest CNF nodes so that my assumption of the X node being in the second child position holds  
        subtree = p_tree[pos]
        if isinstance(subtree, ParentedTree) and subtree.label().startswith("X"):
            return pos
        
    return None

def remove_cnf_nodes(paren_str):
    tree = ParentedTree.fromstring(paren_str)
    pos = deepest_cnf_node(tree) # treeposition of top CNF node
    if pos:
        t = tree[pos].parent() # parent tree to fix 

        label = t.label() 
        children = t[:]
        if not children[1].label().startswith("X"):
            print(str(t))
            raise ValueError("Node must be a CNF node") # I'm assuming the X... node is on the right 

        new_tree = Tree(label, [children[0], *tree[pos][:]])
        new_tree = ParentedTree.convert(new_tree)

        top_pos = tree[pos].parent().treeposition()
        if top_pos == (): # if we are changing the root tree, we cannot reassign the top treeposition tree, so we pass the new tree
            output_parens = new_tree.pformat(margin=1000)
            # print("ROOT tree", output_parens, "\n")
            return remove_cnf_nodes(output_parens)
        else:    
            tree[top_pos] = new_tree
            output_parens = tree.pformat(margin=1000)
            # print("NON ROOT", output_parens, "\n")
            return remove_cnf_nodes(output_parens)
    else:
        # print("WE DID IT")
        return paren_str
### TESTING ###



def safelog(x):
    if x == 0:
        return -float('inf')
    else:
        return math.log(x)

class PCFG:
    """ PCFG in Chomsky Normal Form. """
    def __init__(self, nonterminal_rules, terminal_rules, root):
        self.nt_rules = nonterminal_rules
        self.t_rules = terminal_rules
        self.root = root

        # Build mappings from rhs to rules with probabilities
        self.nt_rules_inv = defaultdict(list)
        self.t_rules_inv = defaultdict(list)
        for rule, p in self.t_rules.items():
            self.t_rules_inv[rule.rhs[0]].append((rule.lhs, p))
        for rule, p in self.nt_rules.items():
            self.nt_rules_inv[rule.rhs].append((rule.lhs, p))
        
    # def score(self, xs):
    #     T = len(xs)
    #     chart = [[{} for _ in range(T)] for _ in range(T)]
    #     for i, word in enumerate(xs):
    #         for nt, p in self.t_rules_inv[word]:
    #             chart[i][i][nt] = p
    #     for span in range(2, T + 1):
    #         for i in range(T - span + 1):
    #             j = i + span - 1
    #             cell = Counter()
    #             for k in range(i, j):
    #                 left_cell = chart[i][k]
    #                 right_cell = chart[k+1][j]
    #                 for B, bscore in left_cell.items():
    #                     for C, cscore in right_cell.items():
    #                         for nt, p in self.nt_rules_inv[B, C]:
    #                             cell[nt] += bscore * cscore * p
    #             chart[i][j] = cell

    #     # save_chart_to_txt(chart, "chart_output.txt")
    #     return safelog(chart[0][T-1][self.root])
    
    def score(self, xs):  # probabilities in addition to the structure
        '''
        if there's multiple grammars possible that could generate this sentence,
        sum over the probabilities of each tree
        '''
        T = len(xs)
        chart = [[{} for _ in range(T)] for _ in range(T)]

        for i, word in enumerate(xs):
            if word == "_<WUG>":
                # print(f"Substituting <WUG> at position {i}:")
                # Sum over all possible terminals
                for wug_word, entries in self.t_rules_inv.items():
                    # print(f"  {wug_word} -> {entries}")
                    for nt, p in entries:
                        chart[i][i][nt] = chart[i][i].get(nt, 0) + p
            else:
                for nt, p in self.t_rules_inv.get(word, []):
                    chart[i][i][nt] = p

        for span in range(2, T + 1):
            for i in range(T - span + 1):
                j = i + span - 1
                cell = Counter()
                for k in range(i, j):
                    left_cell = chart[i][k]
                    right_cell = chart[k+1][j]
                    for B, bscore in left_cell.items():
                        for C, cscore in right_cell.items():
                            for nt, p in self.nt_rules_inv.get((B, C), []):
                                cell[nt] += bscore * cscore * p
                chart[i][j] = cell

        return safelog(chart[0][T-1].get(self.root, 0))

    # def best_parse(self, xs):
    #     T = len(xs)
    #     chart = [[{} for _ in range(T)] for _ in range(T)]
    #     backpointers = [[{} for _ in range(T)] for _ in range(T)]
        
    #     # Base case: Fill in words with their lexical probabilities
    #     for i, word in enumerate(xs):
    #         for nt, p in self.t_rules_inv[word]:  # Terminal rules
    #             chart[i][i][nt] = p
    #             backpointers[i][i][nt] = word  # Store the word as a leaf

    #     # Recursive case: Fill in larger spans
    #     for span in range(2, T + 1):  # Span size (2 to T)
    #         for i in range(T - span + 1):  
    #             j = i + span - 1  # Span endpoint
    #             cell = {}  # Stores non-terminals for this span
    #             backpointer = {}  # Stores split information
                
    #             for k in range(i, j):  # Split point
    #                 left_cell = chart[i][k]
    #                 right_cell = chart[k+1][j]
                    
    #                 for B, bscore in left_cell.items():
    #                     for C, cscore in right_cell.items():
    #                         for nt, p in self.nt_rules_inv.get((B, C), []):
    #                             score = bscore * cscore * p
    #                             if nt not in cell or score > cell[nt]:  # Maximize probability
    #                                 cell[nt] = score
    #                                 backpointer[nt] = (B, C, k)  # Store best split
                
    #             chart[i][j] = cell
    #             backpointers[i][j] = backpointer

    #     # Get the best parse tree for the full sentence
    #     if self.root in chart[0][T-1]:  # Check if the sentence can be parsed
    #         return extract_tree(backpointers, 0, T-1, self.root)
    #     else:
    #         return "No valid parse found"

    def best_parse(self, xs):
        T = len(xs)
        chart = [[{} for _ in range(T)] for _ in range(T)]
        backpointers = [[{} for _ in range(T)] for _ in range(T)]

        for i, word in enumerate(xs):
            if word == '_<WUG>':
                # Try all possible terminal substitutions and keep best-scoring one
                for terminal_word, entries in self.t_rules_inv.items():
                    for nt, p in entries:
                        if p > chart[i][i].get(nt, 0):
                            chart[i][i][nt] = p
                            backpointers[i][i][nt] = terminal_word
            else:
                for nt, p in self.t_rules_inv.get(word, []):
                    chart[i][i][nt] = p
                    backpointers[i][i][nt] = word

        # for i, word in enumerate(xs):
        #     if word == '_<WUG>':
        #         # For best_parse, pick the *most probable* terminal substitution
        #         best_nt = None
        #         best_word = None
        #         best_score = 0

        #         for terminal_word, entries in self.t_rules_inv.items():
        #             for nt, p in entries:
        #                 if p > best_score:
        #                     best_score = p
        #                     best_nt = nt
        #                     best_word = terminal_word

        #         if best_nt:
        #             chart[i][i][best_nt] = best_score
        #             backpointers[i][i][best_nt] = best_word
        #     else:
        #         for nt, p in self.t_rules_inv.get(word, []):
        #             chart[i][i][nt] = p
        #             backpointers[i][i][nt] = word

        for span in range(2, T + 1):
            for i in range(T - span + 1):
                j = i + span - 1
                cell = {}
                backpointer = {}

                for k in range(i, j):
                    left_cell = chart[i][k]
                    right_cell = chart[k+1][j]

                    for B, bscore in left_cell.items():
                        for C, cscore in right_cell.items():
                            for nt, p in self.nt_rules_inv.get((B, C), []):
                                score = bscore * cscore * p
                                if nt not in cell or score > cell[nt]:
                                    cell[nt] = score
                                    backpointer[nt] = (B, C, k)

                chart[i][j] = cell
                backpointers[i][j] = backpointer

        if self.root in chart[0][T-1]:
            return extract_tree(backpointers, 0, T-1, self.root)
        else:
            return "No valid parse found"


def gensym(_state=itertools.count()):
    return 'X' + str(next(_state))

def preterminal_for(x):
    return 'P' + x

def nonterminal_for_sequence(xs, rules, _suffix_dict):
    if xs in _suffix_dict:
        return _suffix_dict[xs]
    elif len(xs) == 2:
        new_nt = gensym()
        rule = Rule(new_nt, (xs[0], xs[1]))
        rules[rule] = 1.0
        _suffix_dict[xs] = new_nt
        return new_nt
    else:
        new_nt = gensym()
        first, *rest = xs
        next_nt = nonterminal_for_sequence(tuple(rest), rules, {})
        rule = Rule(new_nt, (first, next_nt))
        rules[rule] = 1.0
        _suffix_dict[xs] = new_nt
        return new_nt

def is_terminal(symbol):
    return symbol.startswith(TERMINAL_MARKER) or symbol == NONE

def convert_to_cnf(rules):
    # remove unary productions
    nonterminals = {rule.lhs for rule in rules.keys()}
    unit_paths = defaultdict(Counter)
    for A in nonterminals:
        unit_paths[A][A] = 1.0
        queue = [A]
        while queue:
            x = queue.pop()
            for rule, p in rules.items():
                if rule.lhs == x and len(rule.rhs) == 1:
                    y, = rule.rhs
                    new_prob = unit_paths[A][x] * p
                    if y not in unit_paths[A]:
                        unit_paths[A][y] = new_prob
                        queue.append(y)
    deunarized_rules = Counter()
    for A in nonterminals:
        for B in unit_paths[A]:
            for rule, p in rules.items():
                if rule.lhs == B and (len(rule.rhs) > 1 or is_terminal(rule.rhs[0])):
                    new_prob = unit_paths[A][B] * p
                    new_rule = Rule(A, rule.rhs)
                    deunarized_rules[new_rule] += new_prob
        
    # add preterminals
    pt_rules = {}
    for (lhs, rhs), p in deunarized_rules.items():
        new_rhs = []
        for symbol in rhs:
            assert len(rhs) > 1 or is_terminal(rhs[0])
            if is_terminal(symbol) and len(rhs) > 1:
                preterminal = preterminal_for(symbol)
                new_preterminal_rule = Rule(preterminal, (symbol,))
                pt_rules[new_preterminal_rule] = 1.0
                new_rhs.append(preterminal)
            else:
                new_rhs.append(symbol)
        pt_rules[Rule(lhs, tuple(new_rhs))] = p

    # binarize by introducing new nonterminals
    nt_rules = {}
    t_rules = {}
    for rule, p in pt_rules.items():
        if len(rule.rhs) == 1: # terminal rule
            t_rules[rule] = p
        elif len(rule.rhs) == 2: # binary rule
            nt_rules[rule] = p
        else: # ternary+ rule
            first, *rest = rule.rhs
            new_nt = nonterminal_for_sequence(tuple(rest), nt_rules, {})
            rule = Rule(rule.lhs, (first, new_nt))
            nt_rules[rule] = p

    return nt_rules, t_rules

def test():
    '''
    a^n * b^n
    underscore denotes a lexical item (terminal node)
    We are testing if the score function works correctly.

    getting -inf from score function means probability was 0
    '''
    for i in range(100):
        p_continue = random.random() * .4
        rules = {
            Rule('S', ('_a', '_b')) : 1 - p_continue,  # probability associated with this rule
            Rule('S', ('_a', 'S', '_b')) : p_continue,
        }
        nt_rules, t_rules = convert_to_cnf(rules)
        pcfg = PCFG(nt_rules, t_rules, 'S')
        assert pcfg.score(['_a', '_b']) == math.log(1 - p_continue)
        n_2_case_score = pcfg.score(['_a', '_a', '_b', '_b'])
        assert n_2_case_score == math.log(p_continue * (1 - p_continue))  # probability of second rule times first rule
        n_3_case_score = pcfg.score(['_a', '_a', '_a', '_b', '_b', '_b'])
        assert round(n_3_case_score, 5) == round(math.log(p_continue **2 * (1 - p_continue)), 5)

    # (ab)^n
    for i in range(100):
        p_continue = random.random() * .4
        rules = {
            Rule('S', ('_a', '_b')): 1 - p_continue,
            Rule('S', ('_a', 'S1')): p_continue,
            Rule('S1', ('S', '_b')): 1.0  # helper to enforce (a S b)
        }
        nt_rules, t_rules = convert_to_cnf(rules)
        pcfg = PCFG(nt_rules, t_rules, 'S')

        assert pcfg.score(['_a', '_b']) == math.log(1 - p_continue)
        assert round(pcfg.score(['_a', '_a', '_b', '_b']), 5) == round(math.log(p_continue * (1 - p_continue)), 5)
        assert round(pcfg.score(['_a', '_a', '_a', '_b', '_b', '_b']), 5) == round(math.log(p_continue ** 2 * (1 - p_continue)), 5)

    # (abc)^n
    for i in range(100):
        p_continue = random.random() * .4
        rules = {
            Rule('S', ('_a', '_b', '_c', 'S')): p_continue,
            Rule('S', ('_a', '_b', '_c')): 1 - p_continue,
        }
        nt_rules, t_rules = convert_to_cnf(rules)
        pcfg = PCFG(nt_rules, t_rules, 'S')
        # print("what it is:", pcfg.score(['_a', '_b', '_c']), "\nWhat it should be:", math.log(1 - p_continue))
        assert pcfg.score(['_a', '_b', '_c']) == math.log(1 - p_continue)
        # print("what it is:", round(pcfg.score(['_a', '_b', '_c', '_a', '_b', '_c']), 5), "\nWhat it should be:", round(math.log(p_continue * (1 - p_continue)), 5))
        assert round(pcfg.score(['_a', '_b', '_c', '_a', '_b', '_c']), 5) == round(math.log(p_continue * (1 - p_continue)), 5)
        # print("what it is:", round(pcfg.score(['_a', '_b', '_c', '_a', '_b', '_c', '_a', '_b', '_c']), 5), "\nWhat it should be:", round(math.log(p_continue ** 2 * (1 - p_continue)), 5))
        assert round(pcfg.score(['_a', '_b', '_c', '_a', '_b', '_c', '_a', '_b', '_c']), 5) == round(math.log(p_continue ** 2 * (1 - p_continue)), 5)



    # for i in range(100):
    #     p_continue = random.random() * .4
    #     rules = {
    #         Rule('S', ('_a')): 1 - p_continue,  # probability associated with this rule
    #         Rule('S', ('_a', 'S')): p_continue,
    #     }
    #     nt_rules, t_rules = convert_to_cnf(rules)
    #     pcfg = PCFG(nt_rules, t_rules, 'S')
    #     print(nt_rules)
    #     print(t_rules)
    #     n_case_score = pcfg.score(['_a'])
    #     print(n_case_score)
    #     print(math.log(1 - p_continue))
    #     # assert n_case_score == math.log(1 - p_continue)
    #     n_2_case_score = pcfg.score(['_a', '_a'])
    #     assert n_2_case_score == math.log(p_continue * (1 - p_continue))  # probability of second rule times first rule
    #     n_3_case_score = pcfg.score(['_a', '_a', '_a'])
    #     assert round(n_3_case_score, 5) == round(math.log(p_continue ** 2 * (1 - p_continue)), 5)

def read_grammar(grammar_filename):
    rules = {}
    with open(grammar_filename) as infile:
        for line in infile:
            if line.startswith('#'):
                continue
            else:
                logprob, lhs, *rhs = line.strip().split()
                if rhs:
                    rule = Rule(lhs, tuple(rhs))
                    rules[rule] = math.exp(float(logprob))
    nt_rules, t_rules = convert_to_cnf(rules)
    return PCFG(nt_rules, t_rules, ROOT)


def main(grammar_filename, text_filename, output_csv):
    print("Processing grammar...", file=sys.stderr)
    grammar = read_grammar(grammar_filename)
    grammar_terminals = [terminal for terminal, lhs in grammar.t_rules_inv.items()]
    print("Built CNF grammar with %d nonterminal rules." % len(grammar.nt_rules), file=sys.stderr)
    print("Calculating inside probabilities...", file=sys.stderr)

    # ## grammar tests ##
    # with open("terminals_test.txt", 'w') as file:
    #     for term in grammar_terminals:
    #         file.write(f"{term}\n")

    # with open("t_rules.txt", 'w') as file:
    #     for rule in grammar.t_rules.items():
    #         file.write(f"{rule}\n")

    # with open("t_rules_inv.txt", 'w') as file:
    #     for rule in grammar.t_rules_inv.items():
    #         file.write(f"{rule}\n")
    # print("Done grammar tests\n")
    # ## grammar tests ##

    with open(text_filename) as infile:
        lines = infile.readlines()

    # Open CSV file for writing
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sentence", "WUGgedSentence", "Score", "LenFacScore", "CleanedParse"])  # CSV headers

        for line in tqdm.tqdm(lines):
            terminals = [
                "".join([TERMINAL_MARKER, terminal]) if terminal != NONE else terminal
                for terminal in line.strip().split()
            ]
            # check if first word capitalized is in grammar terminals
            captilazed_w1 = terminals[0][0] + terminals[0][1].capitalize() + terminals[0][2:]
            print(captilazed_w1)
            if terminals[0] not in grammar_terminals and captilazed_w1 in grammar_terminals: 
                terminals[0] = captilazed_w1

            terminals_w_wugs = [word if word in grammar_terminals else "_<WUG>" for word in terminals]
            print(terminals)
            print(terminals_w_wugs)
            score = grammar.score(terminals_w_wugs)
            print(score)
            lenfac_score = score/len(terminals_w_wugs)
            print(lenfac_score)
            best_parse = grammar.best_parse(terminals_w_wugs)
            # print(best_parse)
            cleaned_parse = remove_cnf_nodes(best_parse)
            # cleaned_parse = ""

            # Write the result to CSV
            writer.writerow([line.strip(), " ".join(terminals_w_wugs).replace("_",''), score, lenfac_score, cleaned_parse])

    print(f"Results saved to {output_csv}")


if __name__ == '__main__':
    pass
#     main(*sys.argv[1:])


# grammar = read_grammar("./sample1.0.FG-output.rank-1.txt")

# with open("./test_items.txt") as infile:
#     lines = infile.readlines()

# for line in tqdm.tqdm(lines):
#     terminals = [
#         "".join([TERMINAL_MARKER, terminal]) if terminal != NONE else terminal
#         for terminal in line.strip().split()
#     ]
#     score = grammar.score(terminals)
#     best_parse = grammar.best_parse(terminals)
