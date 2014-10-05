__author__ = 'Sun'


class PosTags(object):

    PADDING_POS_TAG = "#PAD#"
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
        """.split() + [PADDING_POS_TAG]))



class ChunkTypes(object):

    PADDING_CHUNK_TYPE = "#PAD#"

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


class NERTypes(object):

    PADDING_NER_TYPE = "#PAD#"
    NERTYPE_ID_MAP = dict((tag,id) for id,tag in enumerate([
        '#',
        PADDING_NER_TYPE]))

class SrlTypes(object):

    PADDING_SRL_TYPE = "#PAD#"
    SRLTYPE_ID_MAP = dict((tag,id) for id,tag in enumerate([
        '#',
        '*',
        'R-A4',
        'C-AM-DIR',
        'R-A0',
        'R-A1',
        'AM-MNR',
        'R-A3',
        'AM-MOD',
        'C-AM-MNR',
        'R-AM-MNR',
        'R-AM-TMP',
        'AM-PRD',
        'R-AM-DIR',
        'C-AM-CAU',
        'R-A2',
        'C-AM-TMP',
        'AM-EXT',
        '*',
        'R-AM-CAU',
        'A1',
        'A0',
        'A3',
        'A2',
        'A5',
        'A4',
        'R-AM-EXT',
        'C-V',
        'AM-DIR',
        'AM-DIS',
        'AM-TMP',
        'AM-REC',
        'AA',
        'C-AM-DIS',
        'AM-PNC',
        'AM-LOC',
        'C-A4',
        'AM',
        'R-AM-LOC',
        'C-AM-EXT',
        'V',
        'AM-CAU',
        'C-AM-LOC',
        'R-AM-ADV',
        'C-AM-PNC',
        'C-AM-NEG',
        'C-A3',
        'C-A2',
        'C-A1',
        'C-A0',
        'R-AA',
        'C-A5',
        'R-AM-PNC',
        'AM-ADV',
        'C-AM-ADV',
        'AM-NEG',
        PADDING_SRL_TYPE]))
