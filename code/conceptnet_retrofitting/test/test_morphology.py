from conceptnet_retrofitting.builders.morphological_rules import *
from nose.tools import eq_


vocab = [
    'happy', 'happier', 'dopey', 'dopier', 'grumpy', 'grumpier',
    'sneezy', 'sneezier', 'sleepy', 'sleepier', 'bashful', 'doc'
]


def test_candidate_rules():
    rules = list(generate_candidate_rules(vocab, min_examples=4))
    print(rules)
    eq_(len(rules), 2)
    for rule, support in rules:
        assert str(rule) in {'suffix:y:ier', 'suffix:ier:y'}
        assert support == {'happ', 'grump', 'sneez', 'sleep'}
