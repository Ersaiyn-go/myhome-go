from itertools import combinations

class eApriori:
    def __init__(self, min_support=0.5):
        self.min_support = min_support
        self.freq_itemsets = []

    def fit(self, transactions):
        item_counts = {}
        total = len(transactions)
        items = set(item for transaction in transactions for item in transaction)

        # Частотные 1-элементы
        for item in items:
            count = sum(1 for t in transactions if item in t)
            support = count / total
            if support >= self.min_support:
                self.freq_itemsets.append((frozenset([item]), support))

        k = 2
        current_freq = [fs for fs, _ in self.freq_itemsets]

        while current_freq:
            candidates = list(combinations(set().union(*current_freq), k))
            new_freq = []
            for candidate in candidates:
                candidate_set = frozenset(candidate)
                count = sum(1 for t in transactions if candidate_set.issubset(t))
                support = count / total
                if support >= self.min_support:
                    new_freq.append(candidate_set)
                    self.freq_itemsets.append((candidate_set, support))
            current_freq = new_freq
            k += 1

    def get_freq_itemsets(self):
        return self.freq_itemsets
