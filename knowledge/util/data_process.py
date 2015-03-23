__author__ = 'Sun'


from itertools import islice

def moving_window(seq, window_size):
    """
    Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    :param seq:
    :param window_size:
    :return:
    """

    assert len(seq) >= window_size, " input sequence is shorter than window size"

    it = iter(seq)
    result = tuple(islice(it, window_size))

    yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]