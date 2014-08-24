__author__ = 'Sun'


class PosTags(object):

    POSTAG_ID_MAP = dict((tag, id) for id, tag in enumerate("""
            #
            $
            ''
            (
            )
            ,
            --
            .
            :
            CC
            CD
            DT
            EX
            FW
            IN
            JJ
            JJR
            JJS
            LS
            MD
            NN
            NNP
            NNPS
            NNS
            PDT
            POS
            PRP
            PRP$
            RB
            RBR
            RBS
            RP
            SYM
            TO
            UH
            VB
            VBD
            VBG
            VBN
            VBP
            VBZ
            WDT
            WP
            WP$
            WRB
            ``
        """.split()))



class ChunkTypes(object):

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
                         'O']))

class SrlTypes(object):
    SRL_ID_MAP = dict((tag,id) for id,tag in enumerate([]))
