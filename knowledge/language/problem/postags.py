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
            AUX
            AUXG
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


