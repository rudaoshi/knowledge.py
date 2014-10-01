__author__ = 'sunmingming01'


class Corpora(object):
    """
    Base class for Corporas
    """

    def words(self):
        """
        Get iterator of all words in the corpora
        :return: itertor of words
        """
        pass

    def get_word(self, word_id):
        """
        Get the word given the wordid
        :param word_id:
        :return: the word with the wordid
        """
        pass

    def sentences(self):
        """
        Get iterator of all sentences in the corpora
        :return: itertor of sentences
        """
        pass