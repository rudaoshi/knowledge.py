__author__ = 'Sun'

import random

def roulette(cumsum_ratios):
    random_val = random.random()

    for idx, cumsum_ratio in enumerate(cumsum_ratios):
        if random_val < cumsum_ratio:
            return idx

    return -1