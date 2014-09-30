__author__ = 'sunmingming01'



class Verb(object):
    """
    Class for verbs in sentences
    Used in SRL task
    """

    def __init__(self, word):

        self.__word = word

        self.__verb_sense = None
        self.__verb_infinitive = None

        self.__owner = None
        self.__roles = dict()

    def add_role(self, name, pos):

        self.__roles[name] = pos

    def roles(self):

        for name, pos in self.__roles.iteritems():
            yield name, pos

    @property
    def word(self):
        return self.__word

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