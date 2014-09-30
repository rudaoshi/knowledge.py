__author__ = 'sunmingming01'
from bidict import bidict



PADDING_WORD_STR = "$$$$"

"""
Word ID allocator
"""
word_id_map = bidict({PADDING_WORD_STR: 0})  # reserved for padding str

def alloc_word_id(word_str):
    if word_str not in word_id_map:
        cur_top_idx = len(word_id_map)
        word_id_map[word_str] = cur_top_idx

    return word_id_map[word_str]

