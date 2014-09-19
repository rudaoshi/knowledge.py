__author__ = 'Sun','Huang'



from collections import defaultdict, Counter, deque
from knowledge.language.core.word import Word
from knowledge.language.core.sentence import Sentence
from knowledge.language.core.document import Document
import csv

class Corpora(object):

    PADDING_WORD = Word.padding_word()
    PADDING_WORD2 = Word.padding_word2()

    def __init__(self):
        self.documents = []

        self.word_id_map = dict()
        self.id_word_map = dict()

        self.sentence_id_map = dict()

        self.word_id_map[self.PADDING_WORD.content] = 0
        self.word_id_map[self.PADDING_WORD2] = 1

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


    def load_conll2005(self,filename):
        document = Document(0)
        for idx , rawsent in enumerate(Conll05.loadraw(filename)):
            sentence_obj = Sentence(idx)
            for verb,vidx,sent_algined in Conll05.sentence2iobsentece(rawsent):
                for word, tag in sent_algined:
                    id = self.alloc_global_word_id(word)

                    word_obj = Word(id, word)
                    word_obj.pos = tag

                    sentence_obj.add_word(word_obj)

                # we append verb idx in a sentence data
                document.add_sentence([vidx,sentence_obj])

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


class Conll05(object):
    def __init__(self):
        pass

    @staticmethod
    def loadraw(filename):
        # return corpus sentence by sentence
        raw_corpora = list()
        with open(filename) as fr:
            sentence = list()
            while True:
                line = fr.readline()
                if not line:
                    break
                cols = line.split()
                if len(cols) == 0:
                    continue
                if cols[1] == '.':
                    # sentence end here
                    raw_corpora.append(sentence)
                    sentence = list()
                sentence.append(cols[:1] + cols[6:])
        return raw_corpora

    @staticmethod
    def sentence2iobsentece(sentence):
        def is_begin(ss):
            if ss[0] == '(':
                return True
            else:
                return False

        def is_end(ss):
            if ss[-1] == ')':
                return True
            else:
                return False

        def is_verb(ss):
            if '(V' in ss[0:2]:
                return True
            else:
                return False

        def strip_label(ss):
            ss = ss.replace('*',' ')
            ss = ss.replace('(',' ')
            ss = ss.replace(')',' ')
            return ss.strip()

        iobsentences = list()
        vbnum = len(sentence[0][3:])
        for i in xrange(vbnum):
            iobsent = list()
            vbidx = -1
            pre_tag = None
            for idx,items in enumerate(sentence):
                word = items[0]
                pos = items[1]
                raw_srltag = items[i+2]
                if raw_srltag == '*' and pre_tag == None:
                    tag = '*'
                elif raw_srltag == '*' and pre_tag != None:
                    tag = pre_tag
                elif is_verb(raw_srltag):
                    vbidx = idx
                    tag = strip_label(raw_srltag)
                    pre_tag = tag
                elif is_begin(raw_srltag):
                    tag = strip_label(raw_srltag)
                    pre_tag = tag

                if is_end(raw_srltag):
                    tag = pre_tag
                    pre_tag = None

                iobsent.append([word,pos,tag])
            iobsentences.append([vbidx,iobsent])
        return iobsentences






