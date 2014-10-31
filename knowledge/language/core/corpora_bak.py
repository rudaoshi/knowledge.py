__author__ = 'Sun','Huang'



from collections import defaultdict, Counter, deque
from knowledge.language.core.sentence.sentence import Sentence
from knowledge.language.core.document import Document
import csv

class Word(object):
    '''
    this word clas is for pos 
    '''
    def __init__(self,id,content):
        self.id = id
        self.content = content


class Corpora(object):

    PADDING_WORD = Word(0,'$$$$')

    def __init__(self):
        self.documents = []

        self.word_id_map = dict()
        self.id_word_map = dict()

        self.sentence_id_map = dict()

        self.word_id_map[self.PADDING_WORD.content] = 0
        self.id_word_map[0] = self.PADDING_WORD.content


        self.tns = defaultdict(Counter)        #term number in each document
        self.dns = Counter()                   #doc number containing each term

    def alloc_global_word_id(self, word):
        if word not in self.word_id_map:
            cur_top_idx = len(self.word_id_map)
            self.word_id_map[word] = cur_top_idx
            self.id_word_map[cur_top_idx] = word

        return self.word_id_map[word]


    def get_word_num(self):

        return len(self.word_id_map)


    def load_nltk_conll2000(self):

        import nltk

        document = Document(0)

        for idx, sentence in enumerate(nltk.corpus.conll2000.tagged_sents()):

            sentence_obj = Sentence()
            for word, tag in sentence:
                id = self.alloc_global_word_id(word)

                word_obj = Word(id, word)
                word_obj.tag = tag

                sentence_obj.add_word(word_obj)

            document.add_sentence(sentence_obj)

        self.documents.append(document)


    def add_document(self, document):

        self.documents.append(document)

        word_id_set = set()
        for sentence in document.sentences:
            sentence.id = len(self.sentences)
            self.sentences.append(sentence)

            for word in sentence.words():
                word.id = self.alloc_global_word_id(word)

                self.tns[document.id][word.id] += 1

                if word.id not in word_id_set:

                    self.dns[word.id] += 1
                    word_id_set.add(word.id)

    # def filter_words(self, min_occur = None, max_occur_ratio = None):
    #
    #     doc_num = len(self.tns)
    #
    #     word_id_remove = set()
    #
    #     for word in self.word_id_map:
    #         word_id = self.word_id_map[word]
    #
    #         if min_occur and self.dns[word_id] < min_occur:
    #             word_id_remove.add(word_id)
    #
    #             continue
    #
    #         if max_occur_ratio and self.dns[word_id] > max_occur_ratio * doc_num:
    #             word_id_remove.add(word_id)
    #             continue
    #
    #     for word_id in word_id_remove:
    #
    #         del self.word_id_map[self.id_word_map[word_id]]
    #         del self.id_word_map[word_id]
    #         del self.dns[word_id]
    #
    #     for doc_id in self.tns:
    #         self.tns[doc_id] = dict((k, v ) for k in self.tns[doc_id] if k not in word_id_remove)


    def save_word_dictionary(self, file_name):

        with open(file_name, 'w') as output:
            output.write('\n'.join(self.word_id_map.keys()))

    def save_doc_word_freq(self, file_name):

        with open(file_name, 'w') as output:

            for doc_id in self.tns:
                output.write('\n'.join("\t".join([doc_id, word_id, n] for word_id, n in self.tns[doc_id].iteritmes())))



