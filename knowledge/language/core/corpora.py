__author__ = 'Sun','Huang'



from collections import defaultdict, Counter, deque
from knowledge.language.core.word import Word
from knowledge.language.core.sentence import Sentence
from knowledge.language.core.document import Document
import csv

class Corpora(object):

    PADDING_WORD = Word.padding_word()
    PADDING_WORD2 = Word.padding_word2()

    def __init__(self,pading_lst=[Word.padding_word()]):
        self.documents = []

        self.word_id_map = dict()
        self.id_word_map = dict()

        self.sentence_id_map = dict()

        #self.word_id_map[self.PADDING_WORD.content] = 0
        #self.word_id_map[self.PADDING_WORD2.content] = 1
        for idx,word in enumerate(pading_lst):
            self.word_id_map[word.content] = idx
            self.id_word_map[word.id] = word


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
            for vidx,sent_algined in Conll05.sentence2iobsentece(rawsent):
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
    words = set()
    pos = set()
    tags = set()

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
                #if cols[1] == '.':
                #    continue
                if len(cols) == 0:
                    # sentence end here
                    raw_corpora.append(sentence)
                    sentence = list()
                    continue
                sentence.append(cols[:2] + cols[6:])
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

        def mod_by_verb_phrase(wordlst,poslst,taglst):
            # if there exist verb pharse, we merge them into one tag and word pharse
            vbegin = -1
            vend = -1
            for idx,tag in enumerate(taglst):
                if is_verb(tag):
                    vbegin = idx
                if vbegin != -1 and is_end(tag):
                    vend = idx + 1
                    break
            if (vend - vbegin) == 1:
                return wordlst,poslst,taglst
            else:
                ret_wordlst = wordlst[:vbegin] + ['&'.join(wordlst[vbegin:vend])] + wordlst[vend:]
                ret_poslst = poslst[:vbegin] + ["VERB_PHRASE"] + poslst[vend:]
                ret_taglst = taglst[:vbegin] + ["(V*)"] + taglst[vend:]
                return ret_wordlst,ret_poslst,ret_taglst


        srl_sentences = list()
        vbnum = len(sentence[0][2:])
        wordlst = [i[0] for i in sentence]
        poslst = [i[1] for i in sentence]
        for i in xrange(vbnum):
            taglst = [item[i+2] for item in sentence]
            newwordlst,newposlst,newtaglst = mod_by_verb_phrase(wordlst,poslst,taglst)
            srl_sent = list()
            vbidx = -1
            pre_tag = None
            for idx,(word,pos,raw_srltag) in enumerate(zip(newwordlst,newposlst,newtaglst)):
                #print i,items
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

                Conll05.words.add(word)
                Conll05.pos.add(pos)
                Conll05.tags.add(tag)
                srl_sent.append([word,pos,tag])
            srl_sentences.append([vbidx,srl_sent])
        return srl_sentences






