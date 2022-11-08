import numpy as np

from ngram_vanilla import NGramVanilla


class NGramBackoff(NGramVanilla):
    def __init__(self, n, vsize):
        self.n = n
        self.vsize = vsize
        self.sub_models = [NGramVanilla(k, vsize) for k in range(1, n + 1)]

    def estimate(self, sequences):
        for sub_model in self.sub_models:
            sub_model.estimate(sequences)

    def ngram_prob(self, ngram):
        """Return the smoothed probability with backoff.
        
        That is, if the n-gram count of size self.n is defined, return that.
        Otherwise, check the n-gram of size self.n - 1, self.n - 2, etc. until you find one that is defined.
        
        Hint: Refer to ngram_prob in ngrams_vanilla.py.
        """
        # TODO: Your code here!
        for i in range(self.n, 0, -1):
            prefix = ngram[:-1]
            token = ngram[-1]
            if i == 1:
                return self.sub_models[i - 1].total[prefix] / self.vsize
            if self.sub_models[i - 1].count[prefix][token] == 0:
                ngram = ngram[1:]
            elif self.sub_models[i - 1].count[prefix][token] > 0:
                if self.sub_models[i - 1].total[prefix] == 0:
                    return 0
                return self.sub_models[i - 1].ngram_prob(ngram)
        # End of your code.