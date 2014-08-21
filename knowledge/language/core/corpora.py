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


    def load_conll2005(self,filename):
        document = Document(0)
        for idx , rawsent in enumerate(Conll05.loadraw(filename)):
            sentence_obj = Sentence(idx)
            for verb,vidx,sent_algined in Conll05.raw2align(rawsent):
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
        '''
        load conll05 dataset
        for each sentence,the output is:
        [[w1,pos,l1,l2,...],
        [w2,pos,l1,l2,...],
        ...
        [wn,pos,l1,l2,...]]
        '''

        with open(filename) as fr:
            reader = csv.reader(fr,delimiter=' ')
            sentence = list()
            for idx,line in enumerate(reader):
                if line[0] == '':
                    # this is a empty line
                    yield sentence
                    sentence = list()
                    continue
                cols = [i for i in line if not i == '']
                sentence.append(cols[:1] + cols[6:])

    @staticmethod
    def raw2align(sentence):
        def is_begin(ss):
            if ss[0] == '(':
                return True,strip_label(ss[1:])
            else:
                return False,None

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

        verb_sz = len(sentence[0][1:])
        if verb_sz > 0:
            tmp = zip(*sentence)
            tokens = tmp[0]
            sent_labels = tmp[1:]
            for idx in xrange(verb_sz):
                vidx_begin = -1
                vidx_end = -1
                pre_verb = False
                labels = sent_labels[idx]
                word_labels = list()
                pre_lable = None
                for widx,lb in enumerate(labels):
                    if is_verb(lb):
                        vidx_begin = widx
                        pre_verb = True
                    if pre_lable == None:
                        b,_lb = is_begin(lb)
                        if b:
                            pre_lable = _lb
                            word_labels.append((tokens[widx],_lb))
                            if is_end(lb):
                                pre_lable = None
                                if pre_verb:
                                    vidx_end = widx + 1
                                    pre_verb = False
                        else:
                            word_labels.append((tokens[widx],None))
                    else:
                        word_labels.append((tokens[widx],pre_lable))
                        if is_end(lb):
                            pre_lable = None
                            if pre_verb:
                                vidx_end = widx + 1
                                pre_verb = False

                yield tokens[vidx_begin:vidx_end], xrange(vidx_begin,vidx_end), word_labels




