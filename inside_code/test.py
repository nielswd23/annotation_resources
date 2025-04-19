from inside import *


def test():
    '''

    Runs a variety of tests on the score function from inside.py.
    Notes:
        underscore denotes a lexical item (terminal node)
        getting -inf from score function means probability was 0
    '''
    # a^n b^n
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
            Rule('S1', ( '_b', 'S')): 1.0  # helper to enforce (a S b)
        }
        nt_rules, t_rules = convert_to_cnf(rules)
        pcfg = PCFG(nt_rules, t_rules, 'S')

        assert pcfg.score(['_a', '_b']) == math.log(1 - p_continue)
        assert round(pcfg.score(['_a', '_b', '_a', '_b']), 5) == round(math.log(p_continue * (1 - p_continue)), 5)
        assert round(pcfg.score(['_a', '_b', '_a', '_b', '_a', '_b']), 5) == round(math.log(p_continue ** 2 * (1 - p_continue)), 5)

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

        # a(bc)^n
        for i in range(100):
            p_continue = random.random() * .4
            rules = {
                Rule('S1', ('_a', 'S')): 1,
                Rule('S', ('_b', '_c', 'S')): p_continue,
                Rule('S', ('_b', '_c')): 1 - p_continue,
            }
            nt_rules, t_rules = convert_to_cnf(rules)
            pcfg = PCFG(nt_rules, t_rules, 'S1')
            # print("what it is:", pcfg.score(['_a', '_b', '_c']), "\nWhat it should be:", math.log(1 - p_continue))
            assert pcfg.score(['_a', '_b', '_c']) == math.log(1 - p_continue)
            # print("what it is:", round(pcfg.score(['_a', '_b', '_c', '_a', '_b', '_c']), 5), "\nWhat it should be:", round(math.log(p_continue * (1 - p_continue)), 5))
            assert round(pcfg.score(['_a', '_b', '_c', '_b', '_c']), 5) == round(math.log(p_continue * (1 - p_continue)), 5)
            # print("what it is:", round(pcfg.score(['_a', '_b', '_c', '_a', '_b', '_c', '_a', '_b', '_c']), 5), "\nWhat it should be:", round(math.log(p_continue ** 2 * (1 - p_continue)), 5))
            assert round(pcfg.score(['_a', '_b', '_c', '_b', '_c', '_b', '_c']), 5) == round(math.log(p_continue ** 2 * (1 - p_continue)), 5)

        # 2) Palindrome grammar for odd-length 'a'ⁿa'a'ⁿ
        #    S → _a                 w.p. (1-p)   (center)
        #    S → _a S _a            w.p. p       (wrap)
        for _ in range(50):
            p = random.random() * 0.4
            rules = {
                Rule('S', ('_a',)): 1 - p,
                Rule('S', ('_a', 'S', '_a')): p,
            }
            nt, tt = convert_to_cnf(rules)
            pcfg = PCFG(nt, tt, 'S')
            for n in range(4):  # test lengths 1,3,5,7
                seq = ['_a'] * (2 * n + 1)
                logp = pcfg.score(seq)
                # uses n wraps then stop: p^n * (1-p)
                expected = math.log(p ** n * (1 - p))
                assert round(logp, 5) == round(expected, 5), \
                    f"Palin failed n={n}: got {logp}, exp {expected}"

        # 3) WUG substitution sum: if grammar has two terminals under N,
        #    then `['_<WUG>']` should sum their probs
        rules = {
            Rule('S', ('N',)): 1.0,
            Rule('N', ('_dog',)): 0.6,
            Rule('N', ('_cat',)): 0.4,
        }
        nt, tt = convert_to_cnf(rules)
        pcfg = PCFG(nt, tt, 'S')
        scored = pcfg.score(['_<WUG>'])
        # should be log(0.6 + 0.4) == log(1.0)
        assert round(scored, 5) == round(math.log(1.0), 5), \
            f"WUG failed: got {scored}, exp {math.log(1.0)}"

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

    print('All tests passed!')


if __name__ == '__main__':
    test()