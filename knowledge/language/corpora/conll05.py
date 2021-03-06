
__author__ = 'sunmingming01'

import numpy as np
from knowledge.language.core.corpora import Corpora
from knowledge.language.core.word import word, word_repo
from knowledge.language.core.sentence.property import WordProperty
from knowledge.language.core.sentence.sentence import Sentence
from knowledge.language.core.sentence.srl_structure import SRLStructure, Role
from knowledge.language.core.sentence.chunk_structure import Chunk
from knowledge.language.core.sentence.ner_structure import Ner
from knowledge.language.problem import postags, srltypes, locdifftypes


def parse_start_end_components(tags):


    tag_name = ""
    tag_start_pos = 0
    tag_end_pos = 0


    for idx, tag in enumerate(tags):

        if tag.startswith('('):
            tag_name = tag[1:].replace("*","").replace(")","")
            tag_start_pos = idx
        if tag.endswith(')'):
            tag_end_pos =  idx

            if not tag_name:
                raise Exception('Bad tags :' + " ".join(map(str,tags)))

            yield (tag_name, (tag_start_pos, tag_end_pos))

def parse_components_to_tags(tags):
    tag_name = "*"
    tag_start_pos = 0
    tag_end_pos = 0

    for idx, tag in enumerate(tags):
        if tag.startswith('('):
            tag_name = tag[1:].replace("*","").replace(")","")
            tag_start_pos = idx

        yield tag_name

        if tag.endswith(')'):
            tag_end_pos =  idx
            if not tag_name:
                raise Exception('Bad tags :' + " ".join(map(str,tags)))
            tag_name = '*'






class Conll05Corpora(Corpora):
    """
    Class for Coll05 Corpora

    The format of Coll05 data set is as follows

    WORDS---->  NE--->  POS   PARTIAL_SYNT   FULL_SYNT------>   VS   TARGETS  PROPS------->

    The             *   DT    (NP*   (S*        (S(NP*          -    -        (A0*    (A0*
    $               *   $        *     *     (ADJP(QP*          -    -           *       *
    1.4             *   CD       *     *             *          -    -           *       *
    billion         *   CD       *     *             *))        -    -           *       *
    robot           *   NN       *     *             *          -    -           *       *
    spacecraft      *   NN       *)    *             *)         -    -           *)      *)
    faces           *   VBZ   (VP*)    *          (VP*          01   face      (V*)      *
    a               *   DT    (NP*     *          (NP*          -    -        (A1*       *
    six-year        *   JJ       *     *             *          -    -           *       *
    journey         *   NN       *)    *             *          -    -           *       *
    to              *   TO    (VP*   (S*        (S(VP*          -    -           *       *
    explore         *   VB       *)    *          (VP*          01   explore     *     (V*)
    Jupiter     (ORG*)  NNP   (NP*)    *       (NP(NP*)         -    -           *    (A1*
    and             *   CC       *     *             *          -    -           *       *
    its             *   PRP$  (NP*     *          (NP*          -    -           *       *
    16              *   CD       *     *             *          -    -           *       *
    known           *   JJ       *     *             *          -    -           *       *
    moons           *   NNS      *)    *)            *)))))))   -    -           *)      *)
    .               *   .        *     *)            *)         -    -           *       *


    There is one line for each token, and a blank line after the last token.
    The columns, separated by spaces, represent different annotations of the sentence with a tagging along words.
    For structured annotations (named entities, chunks, clauses, parse trees, arguments), we use the Start-End format.

    The Start-End format represents phrases (chunks, arguments, and syntactic constituents)
    that constitute a well-formed bracketing in a sentence (that is, phrases do not overlap, though they admit embedding).
    Each tag is of the form STARTS*ENDS, and represents phrases that start and end at the corresponding word.
    A phrase of type k places a (k parenthesis at the STARTS part of the first word, and a ) parenthesis at the END part of the last word.
    Scripts will be provided to transform a column in Start-End format into other standard formats (IOB1, IOB2, WSJ trees).
    The Start-End format used last year (that considered the phrase type in the start and end parts) will be compatible with the current software and scripts.


    The different annotations in a sentence are grouped in the following blocks:

    WORDS. The words of the sentence.
    NE. Named Entities.
    POS. PoS tags.
    PARTIAL SYNT. Partial syntax, namely chunks (1st column) and clauses (2nd column).
    FULL SYNT. Full syntactic tree. Note that this column represents the following WSJ tree:
     (S
        (NP (DT The)
          (ADJP
            (QP ($ $) (CD 1.4) (CD billion) ))
          (NN robot) (NN spacecraft) )
        (VP (VBZ faces)
          (NP (DT a) (JJ six-year) (NN journey)
            (S
              (VP (TO to)
                (VP (VB explore)
                  (NP
                    (NP (NNP Jupiter) )
                    (CC and)
                    (NP (PRP$ its) (CD 16) (JJ known) (NNS moons) )))))))
        (. .) )
    VS. VerbNet sense of target verbs. These are hand-crafted annotations that will be available only for training and development sets (not for the test set).
    TARGETS. The target verbs of the sentence, in infinitive form.
    PROPS. For each target verb, a column reprenting the arguments of the target verb.

    However, the data set we found is in following format

    WORDS---->                     POS   FULL_SYNT->  NE--->   VS  TARGETS          PROPS---->
    Ms.                            NNP   (S1(S(NP*         *    -   -                    (A0*
    Haag                           NNP           *)    (LOC*)   -   -                       *)
    plays                          VBZ        (VP*         *    02  play                  (V*)
    Elianti                        NNP        (NP*))       *    -   -                    (A1*)
    .                              .             *))       *    -   -                       *


    """

    def __init__(self):

        self.__sentences = []

    def sentences(self):

        for sentence in self.__sentences:
            yield sentence


    def __get_sentence_block(self, file_path):
        sentence_block = []
        with open(file_path,'r') as data_file:
            for line in data_file:
                if line.strip():
                    cols = line.split()
                    sentence_block.append(cols)
                else:
                    if sentence_block:
                        yield sentence_block

                        sentence_block = []

        if sentence_block:
            yield sentence_block

    def load(self, file_path, type=1):
        '''
        load corpora from Conll05 data file
        :param file_path:
        :param type: 1, the data we found in Github;2, append with conll2005/synt.upc
        :return:
        '''


        for sentence_info in self.__get_sentence_block(file_path):


            sentence = Sentence()

            sentence_array = np.array(sentence_info)


            for loc, word_info in enumerate(sentence_array):
                word_name, pos = word_info[:2]

                cur_word = word_repo.get_word(word_name)

                word_property = WordProperty()
                word_property.pos = pos


                if word_info[4] != "-":

                    srl = SRLStructure(cur_word, loc)
                    srl.verb_sense = word_info[4]
                    srl.verb_infinitive = word_repo.get_word(word_info[5])
                    sentence.add_srl_struct(srl)
                    sentence.add_word(srl.verb_infinitive, word_property)
                else:
                    sentence.add_word(cur_word, word_property)

            # parse ne
            for ne_type, (start_pos, end_pos) in parse_start_end_components(sentence_array[:,3]):
                ner = Ner(ne_type, start_pos, end_pos)
                sentence.add_ne(ner)
            if type == 2:
                # parse chunk
                for chunk_type, (start_pos, end_pos) in parse_start_end_components(sentence_array[:,-2]):
                    chunk = Chunk(chunk_type, start_pos, end_pos)
                    sentence.add_chunk(chunk)
            if type == 1:
                props = sentence_array[:,6:]
            else:
                props = sentence_array[:,6:-3]


            for verb_idx, srl in enumerate(sentence.srl_structs()):
                cur_prop = props[:,verb_idx]

                for role_type, (start_pos, end_pos) in parse_start_end_components(cur_prop):
                    role = Role(role_type, start_pos, end_pos)
                    srl.add_role(role)




            self.__sentences.append(sentence)


