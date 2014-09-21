from knowledge.language.core.corpora import Conll05

rawstr = '''
Free                           JJ     (S1(S(NP(NP*                                *    -   -                    (A0*               *               *               *               *               *               *
markets                        NNS               *)                               *    -   -                       *               *               *               *               *               *               *
,                              ,                 *                                *    -   -                       *               *               *               *               *               *               *
free                           JJ             (NP*                                *    -   -                       *               *               *               *               *               *               *
minds                          NNS               *)                               *    -   -                       *               *               *               *               *               *               *
and                            CC                *                                *    -   -                       *               *               *               *               *               *               *
free                           JJ             (NP*                                *    -   -                       *               *               *               *               *               *               *
elections                      NNS               *))                              *    -   -                       *)              *               *               *               *               *               *
have                           AUX            (VP*                                *    03  have                  (V*)              *               *               *               *               *               *
an                             DT          (NP(NP*                                *    -   -                    (A1*            (A1*            (A1*            (A1*            (A1*               *               *
appeal                         NN                *)                               *    -   -                       *               *               *)              *)              *)              *               *
that                           WDT     (SBAR(WHNP*)                               *    -   -                       *               *)         (R-A1*)         (R-A1*)         (R-A1*)              *               *
seems                          VBZ          (S(VP*                                *    01  seem                    *             (V*)              *               *               *               *               *
to                             TO           (S(VP*                                *    -   -                       *          (C-A1*               *               *               *               *               *
get                            VB             (VP*                                *    03  get                     *               *             (V*)              *               *               *               *
muddled                        VBN            (VP*                                *    01  muddle                  *               *            (A2*)            (V*)              *               *               *
only                           RB    (SBAR(WHADVP*                                *    -   -                       *               *        (AM-TMP*        (AM-TMP*       (R-AM-TMP*              *               *
when                           WRB               *)                               *    -   -                       *               *               *               *               *)              *               *
delivered                      VBN          (S(VP*                                *    01  deliver                 *               *               *               *             (V*)              *               *
through                        IN             (PP*                                *    -   -                       *               *               *               *        (AM-MNR*               *               *
U.N.                           NNP         (NP(NP*                            (ORG*)   -   -                       *               *               *               *               *            (A1*               *
organizations                  NNS               *)                               *    -   -                       *               *               *               *               *               *)              *
--                             :                 *                                *    -   -                       *               *               *               *               *               *               *
which                          WDT     (SBAR(WHNP*)                               *    -   -                       *               *               *               *               *          (R-A1*)              *
of                             IN           (S(PP*                                *    -   -                       *               *               *               *               *        (AM-DIS*               *
course                         NN             (NP*))                              *    -   -                       *               *               *               *               *               *)              *
are                            AUX            (VP*                                *    -   -                       *               *               *               *               *               *               *
made                           VBN            (VP*                                *    07  make                    *               *               *               *               *             (V*               *
up                             RP            (PRT*)                               *    -   -                       *               *               *               *               *               *)              *
largely                        RB             (PP*                                *    -   -                       *               *               *               *               *               *               *
of                             IN                *                                *    -   -                       *               *               *               *               *               *               *
governments                    NNS         (NP(NP*)                               *    -   -                       *               *               *               *               *            (A0*            (A0*)
that                           WDT     (SBAR(WHNP*)                               *    -   -                       *               *               *               *               *               *          (R-A0*)
fear                           VBP          (S(VP*                                *    01  fear                    *               *               *               *               *               *             (V*)
these                          DT             (NP*                                *    -   -                       *               *               *               *               *               *            (A1*
principles                     NNS               *)                               *    -   -                       *               *               *               *               *               *               *)
at                             IN             (PP*                                *    -   -                       *               *               *               *               *               *        (AM-LOC*
home                           NN             (NP*)))))))))))))))))))))))))       *    -   -                       *)              *)              *)              *)              *)              *)              *)
.                              .                 *))                              *    -   -                       *               *               *               *               *               *               *
'''


def test_Cornll05():
    print 'test cornll 05'
    raw_corpora = Conll05.loadraw('/home/kingsfield/data/conll05/training-set')
    max_sent_size = -1
    sum_sent_size = 0
    cnt = 0
    for sent in raw_corpora:
        cnt += 1
        sz = len(sent)
        sum_sent_size += sz
        if sz > max_sent_size:
            max_sent_size = sz
        #print sent
        #iobsent = Conll05.sentence2iobsentece(sent)
        #for i in iobsent:
        #   pass
    print '*' * 20
    print "maxium sentece size=%d" % (max_sent_size)
    print 'avg sentece size=%d' % (sum_sent_size/cnt)
    '''
    ss = rawstr.split('\n')
    sentence = list()
    for line in ss:
        sss = line.split()
        if len(sss) == 0:
            continue
        sss = sss[:2] + sss[6:]
        sentence.append(sss)
    for i in Conll05.sentence2iobsentece(sentence):
        print i
        print
    '''
    print 'test cornll 05 done'
    print '*' * 20

