__author__ = 'sun'


class LocDiffToVerbTypes(object):

    diff_id_map = dict()

    @classmethod
    def get_locdiff_id(cls, loc_diff):
        if loc_diff not in cls.diff_id_map:
            cls.diff_id_map[loc_diff] = len(cls.diff_id_map)
        return cls.diff_id_map[loc_diff]

class LocDiffToThisTypes(object):

    diff_id_map = dict()

    @classmethod
    def get_locdiff_id(cls, loc_diff):
        if loc_diff not in cls.diff_id_map:
            cls.diff_id_map[loc_diff] = len(cls.diff_id_map)
        return cls.diff_id_map[loc_diff]