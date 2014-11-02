__author__ = 'Sun'


class NERTypes(object):

    PADDING_NER_TYPE = "#PAD#"
    NERTYPE_ID_MAP = dict((tag,id) for id,tag in enumerate([
        '*',
        'ORG',
        'PER',
        'LOC',
        'MISC',
        PADDING_NER_TYPE]))
