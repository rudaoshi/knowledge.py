__author__ = 'sunmingming01'



import bidict


PADDING_POS_STR = "$$$$"

pos_id_map = bidict({PADDING_POS_STR : 0})  # reserved for padding str

def alloc_pos_id(pos):
    if pos not in pos_id_map:
        cur_top_idx = len(pos_id_map)
        pos_id_map[pos] = cur_top_idx

    return pos_id_map[pos]


class Pos(object):

    def __init__(self, content):
        self.id = alloc_pos_id(content)
        self.content = content

    @classmethod
    def padding_pos(cls):

        padding_pos = Pos(PADDING_POS_STR)
        return padding_pos

    @classmethod
    def padding_pos2(cls):

        padding_pos = Pos(PADDING_POS_STR)
        return padding_pos