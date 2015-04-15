__author__ = 'Sun'




class SrlTypes(object):

    PADDING_SRL_TYPE = "#PAD#"
    SRLTYPE_LABEL_MAP = None
    OTHERTYPE_LABEL = None
    LABEL_SRLTYPE_MAP = None

    @classmethod
    def init(cls):
        """
        making labels according to IOBES role
        """

        srl_types = ['R-A4',
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
            'AM-NEG']

        begin_types = ["B_" + label for label in srl_types]
        in_types = ["I_" + label for label in srl_types]
        end_types = ["E_" + label for label in srl_types]
        single_types = ["S_" + label for label in srl_types]
        all_types = begin_types + in_types + end_types + single_types

        cls.SRLTYPE_LABEL_MAP = dict((srl_type, id) for id, srl_type in enumerate(all_types))
        cls.LABEL_SRLTYPE_MAP = dict((id, srl_type) for id, srl_type in enumerate(all_types))

        cls.OTHERTYPE_LABEL = len(cls.SRLTYPE_LABEL_MAP)
        other_types = ["#PAD#", "#", "*"]
        for other_type in other_types:
            cls.SRLTYPE_LABEL_MAP[other_type] = cls.OTHERTYPE_LABEL

        cls.LABEL_SRLTYPE_MAP[cls.OTHERTYPE_LABEL] = "*"


SrlTypes.init()