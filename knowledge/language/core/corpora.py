__author__ = 'Sun','Huang'



from collections import defaultdict, Counter
from knowledge.language.core.word import Word
from knowledge.language.core.sentence import Sentence
from knowledge.language.core.document import Document
import csv

class Corpora(object):

    PADDING_WORD = Word.padding_word()

    def __init__(self):
        self.documents = []

        self.word_id_map = dict()
        self.id_word_map = dict()

        self.sentence_id_map = dict()

        self.word_id_map[self.PADDING_WORD.content] = 0

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

            sentence_obj = Sentence(idx)
            for word, tag in sentence:
                id = self.alloc_global_word_id(word)

                word_obj = Word(id, word)
                word_obj.pos = tag

                sentence_obj.add_word(word_obj)

            document.add_sentence(sentence_obj)

        self.documents.append(document)


    def load_conll2005(self,random_order=False):
        pass


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


class Conll05(object):
    def __init__(self):
        pass

    @staticmethod
    def load(filename):
        '''
        load conll05 dataset
        for each sentence,output as such
        [((idx1,v1),[(w1,l1),(w2,l2),...,(wk,lk)]),
        ((idx2,v2),...),
        ...,
        ((idxn,vn),...)
        ]
        '''
        def is_begin(ss):
            if ss[0] == '(':
                return True,ss[1:]
            else:
                return False,None

        def is_end(ss):
            if ss[-1] == ')':
                return True
            else:
                return False


        with open(filename) as fr:
            reader = csv.reader(fr,delimiter=' ')
            tokens = list()
            prelabel = list()
            vidx = 0
            vlst = list()
            ret = list()
            for idx,line in enumerate(reader):
                if line[0] == '':
                    # this is a empty line
                    tokens = list()
                    prelabel = list()
                    vidx = 0
                    vlst = list()
                    continue
                cols = [i for i in line if not i == '']
                token = cols[0]
                pos = cols[1]
                labels = cols[6:]
                if pos[0] == 'V':
                    vlst.append(vidx)

                vidx += 1



                print line
                if idx >= 60:
                    break
            return ret

