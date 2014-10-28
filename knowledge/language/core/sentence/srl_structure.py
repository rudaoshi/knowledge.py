__author__ = 'sunmingming01'

class Role(object):
    """
    Class for a role in srl problem
    """
    def __init__(self, type, start_pos, end_pos):

        self.__type = type
        self.__start_pos = start_pos
        self.__end_pos = end_pos

    @property
    def type(self):
        return self.__type

    @property
    def start_pos(self):
        return self.__start_pos

    @property
    def end_pos(self):
        return self.__end_pos

    def pos_shift(self, shift):
        self.__start_pos += shift
        self.__end_pos += shift

    def belongto(self, pos):

        return pos >= self.__start_pos and pos <= self.__end_pos



class SRLStructure(object):
    """
    Class for verbs in sentences
    Used in SRL task
    """

    def __init__(self, verb, verb_loc):

        self.__verb = verb
        self.__verb_loc = verb_loc

        self.__verb_sense = None
        self.__verb_infinitive = None

        self.__owner = None
        self.__roles = []

    def add_role(self, role):

        self.__roles.append(role)

    def roles(self):

        for role in self.__roles:
            yield role

    def pos_shift(self, shift):

        self.__verb_loc += shift

        for role in self.__roles:
            role.pos_shift(shift)


    @property
    def verb(self):
        return self.__verb

    @property
    def verb_loc(self):
        return self.__verb_loc

    @property
    def verb_sense(self):
        return self.__verb_sense

    @verb_sense.setter
    def verb_sense(self, value):
        self.__verb_sense = value

    @property
    def verb_infinitive(self):
        return self.__verb_infinitive

    @verb_infinitive.setter
    def verb_infinitive(self, value):
        self.__verb_infinitive = value

    @property
    def owner(self):
        return self.__owner

    @owner.setter
    def owner(self,value):
        self.__owner = value
