__author__ = 'sun'


class LocDiffToVerbTypes(object):

    DIFF_ID_MAP = dict()

    @classmethod
    def get_locdiff_id(cls, loc_diff):
        if loc_diff not in cls.DIFF_ID_MAP:
            cls.DIFF_ID_MAP[loc_diff] = len(cls.DIFF_ID_MAP)
        return cls.DIFF_ID_MAP[loc_diff]

class LocDiffToWordTypes(object):

    DIFF_ID_MAP = dict()

    @classmethod
    def get_locdiff_id(cls, loc_diff):
        if loc_diff not in cls.DIFF_ID_MAP:
            cls.DIFF_ID_MAP[loc_diff] = len(cls.DIFF_ID_MAP)
        return cls.DIFF_ID_MAP[loc_diff]


class LocTypes(object):

    LOC_ID_MAP = dict()

    @classmethod
    def get_loc_id(cls, loc):
        if loc not in cls.LOC_ID_MAP:
            cls.LOC_ID_MAP[loc] = len(cls.LOC_ID_MAP)
        return cls.LOC_ID_MAP[loc]