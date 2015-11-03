# http://www.aclweb.org/anthology/N15-1186.pdf
from conceptnet_retrofitting.word_vectors import WordVectors
from conceptnet_retrofitting.loaders import load_word_vectors
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist
import numpy as np
import pickle


def partition_from_start(n: int, word: str):
    """
    Get the first n characters of a word, and the characters that remain after
    those.
    """
    return word[:n], word[n:]


def partition_from_end(n: int, word: str):
    """
    Get the last n characters of a word (even if n=0), and the characters that
    remain before those.
    """
    if n == 0:
        return '', word
    else:
        return word[-n:], word[:-n]


class MorphologicalRule:
    rule_type = '?'

    def __init__(self, before: str, after: str,
                 vector_space: WordVectors=None, prototype: str=None,
                 strength: float=1.):
        self.before = before
        self.after = after
        self.vector_space = vector_space
        self.prototype = prototype
        self.vector = None
        self.strength = strength
        self.affix_length = len(self.before)

    def set_prototype(self, vector_space: WordVectors, prototype: str):
        return self.__class__(
            self.before, self.after, vector_space, prototype, self.strength
        )

    def applies_to(self, word: str) -> bool:
        raise NotImplementedError

    def apply(self, word: str) -> str:
        raise NotImplementedError

    def apply_base(self, base: str) -> str:
        raise NotImplementedError

    def apply_vector(self, word: str) -> (str, np.ndarray):
        if self.vector is None:
            v1 = self.vector_space.to_vector(self.prototype)
            v2 = self.vector_space.to_vector(self.apply(self.prototype))
            self.vector = v2 - v1
        vector = self.vector_space.to_vector(word)
        return self.apply(word), vector + self.vector

    def match_scores(self, word, max=100):
        if not self.applies_to(word):
            return max, 0.
        target, vector = self.apply_vector(word)
        norm = np.sqrt(np.sum(vector ** 2))
        vector /= norm
        if target in self.vector_space.labels:
            known_vector = self.vector_space.to_vector(target)
            sim = vector.dot(known_vector)
            return 0, sim
        else:
            return max, 0.

    def explains(self, word, rank_threshold=50, cosine_threshold=0.5):
        rank, sim = self.match_scores(word, rank_threshold)
        return rank < rank_threshold and sim > cosine_threshold

    def __str__(self):
        if self.prototype is None:
            return '%s:%s:%s' % (
                self.rule_type, self.before, self.after
            )
        else:
            return '%s:%s:%s (%s → %s)' % (
                self.rule_type, self.before, self.after, self.prototype, self.apply(self.prototype)
            )

    def __repr__(self):
        return '<MorphologicalRule: %s>' % self


class PrefixMorphologicalRule(MorphologicalRule):
    rule_type = 'prefix'

    def applies_to(self, word: str) -> bool:
        return word.startswith(self.before)

    def apply(self, word: str) -> str:
        prefix, base = partition_from_start(self.affix_length, word)
        assert prefix == self.before
        return self.after + base

    def apply_base(self, base: str) -> str:
        return self.before + base


class SuffixMorphologicalRule(MorphologicalRule):
    rule_type = 'suffix'

    def applies_to(self, word):
        return word.endswith(self.before)

    def apply(self, word):
        suffix, base = partition_from_end(self.affix_length, word)
        assert suffix == self.before
        return base + self.after

    def apply_base(self, base: str) -> str:
        return base + self.before


def affix_dicts(vocab, max_length=6, min_remaining=2, min_examples=10):
    prefixes = defaultdict(set)
    suffixes = defaultdict(set)
    for word in vocab:
        current_max = min(max_length, len(word) - min_remaining)
        for affix_length in range(current_max + 1):
            prefix, prefix_base = partition_from_start(affix_length, word)
            suffix, suffix_base = partition_from_end(affix_length, word)
            prefixes[prefix].add(prefix_base)
            suffixes[suffix].add(suffix_base)

    prefixes_out = {
        prefix: remainders
        for (prefix, remainders) in prefixes.items()
        if len(remainders) >= min_examples
    }
    suffixes_out = {
        suffix: remainders
        for (suffix, remainders) in suffixes.items()
        if len(remainders) >= min_examples
    }
    return prefixes_out, suffixes_out


def vector_median(vectors, nsamples=20):
    step_size = max(1, len(vectors) // nsamples)
    key_vectors = vectors[::step_size]
    dists = cdist(vectors, key_vectors)
    l1_dists = np.sum(np.abs(dists), axis=1)
    best = np.argmin(l1_dists)
    return vectors[best], best


def generate_candidate_rules(vocab, min_examples=10):
    prefixes, suffixes = affix_dicts(vocab, min_examples=min_examples)
    print("Built affix dictionaries")
    for dicts, rule_class in [
        (suffixes, SuffixMorphologicalRule),
        (prefixes, PrefixMorphologicalRule)
    ]:
        affixes = sorted([aff for aff in dicts if aff and aff[0].isalpha()])
        for aff1 in affixes:
            if aff1:
                for aff2 in affixes:
                    if aff1 != aff2:
                        values1 = dicts[aff1]
                        values2 = dicts[aff2]
                        current_min_examples = min_examples
                        if len(values1 & values2) >= min_examples:
                            yield (rule_class(aff1, aff2), values1 & values2)


def generate_vector_rules(wv, vocab, min_examples=10, n_clusters=5):
    rules = []
    for rule, examples in generate_candidate_rules(vocab, int(min_examples * 1.5)):
        ordered_examples = []
        for example in examples:
            untransformed = rule.apply_base(example)
            transformed = rule.apply(untransformed)
            if untransformed in wv.labels and transformed in wv.labels:
                vector = wv.to_vector(transformed) - wv.to_vector(untransformed)
                ordered_examples.append((untransformed, vector))

        vectors = np.vstack([vec for example, vec in ordered_examples])
        center, idx = vector_median(vectors)
        dist = np.sum(center ** 2)
        prototype, _ = ordered_examples[idx]
        new_rule = rule.set_prototype(wv, prototype)
        explained_count = 0
        for (example, vector) in ordered_examples:
            if new_rule.explains(example):
                explained_count += 1
            if explained_count >= min_examples:
                strength = explained_count / len(ordered_examples)
                if strength > 0.05:
                    new_rule.strength = strength
                    print('%4.4f %4.4f %s' % (dist, strength, new_rule))
                    rules.append(new_rule)
                    break

    return rules


def main():
    wv = load_word_vectors(
        'build-data/glove.840B.300d.lower.labels',
        'build-data/glove.840B.300d.l1.lower.npy'
    ).truncate(200000)
    print("Loaded vectors")
    vocab = wv.labels
    rules = generate_vector_rules(wv, vocab)
    with open('rules.pkl', 'wb') as out:
        pickle.dump(rules, out)


if __name__ == '__main__':
    main()
