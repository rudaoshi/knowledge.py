

class Document(object):

    def __init__(self, id, content = ""):
        self.id = id
        self.content = content
        self.sentences = []


    def add_sentence(self, sentence):

        self.sentences.append(sentence)


    def split_sentences(self, sentence_segmenter):
        self.sentences = list(sentence_segmenter.segment(self.content))

    def segement_words(self, word_segmenter):

        for sentence in self.sentences:
            sentence.segement_words(word_segmenter)