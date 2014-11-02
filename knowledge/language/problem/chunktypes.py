__author__ = 'Sun'





class ChunkTypes(object):

    PADDING_CHUNK_TYPE = "#PAD#"
    '''
    CHUNKTYPE_ID_MAP = dict((tag, id) for id, tag in enumerate([
                         'B-ADJP',
                         'B-ADVP',
                         'B-CONJP',
                         'B-INTJ',
                         'B-LST',
                         'B-NP',
                         'B-PP',
                         'B-PRT',
                         'B-SBAR',
                         'B-UCP',
                         'B-VP',
                         'I-ADJP',
                         'I-ADVP',
                         'I-CONJP',
                         'I-INTJ',
                         'I-LST',
                         'I-NP',
                         'I-PP',
                         'I-PRT',
                         'I-SBAR',
                         'I-UCP',
                         'I-VP',
                         'O',
                         PADDING_CHUNK_TYPE]))
    '''
    CHUNKTYPE_ID_MAP = dict((tag, id) for id, tag in enumerate([
            '*',
            'PP',
            'SBAR',
            'ADJP',
            'INTJ',
            'VP',
            'PRT',
            'LST',
            'NP',
            'CONJP',
            'ADVP',
            'UCP',
            PADDING_CHUNK_TYPE]))
