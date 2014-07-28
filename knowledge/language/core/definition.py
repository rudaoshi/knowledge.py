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