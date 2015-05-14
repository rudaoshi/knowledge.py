__author__ = 'Sun'

def SIGN(a, b):
    if b >=0:
        return a if a >=0 else -a
    else:
        return -a if a >= 0 else a

import sys
def mnbrak(func,ax,bx):
    """\
    Bracket the minimum of func, given distinct points ax,bx.
    Return a triplet of points ax,bx,cx where ax < cx < bx,
    and f(cx) < f(ax) and f(cx) < f(bx).

    From NR recipe MNBRAK.

    Works in multidimensions provided ax and bx are numpy arrays
    (and func is a corresponding function of a numpy array).
    """

    GOLD = 1.618034
    GLIMIT = 100
    TINY=1e-20

    fa = func(ax)
    fb = func(bx)

    if fb > fa: # insure fa > fb
        ax,bx = bx,ax
        fa,fb = fb,fa

    cx = bx+GOLD*(bx-ax)
    fc = func(cx)

    while fb > fc:
        r = (bx-ax)*(fb-fc)
        q = (bx-cx)*(fb-fa)
        u = bx-((bx-cx)*q-(bx-ax)*r)/(2.0*SIGN(max(abs(q-r),TINY), q-r))
        ulim = bx+GLIMIT*(cx-bx)  # furthest point to test

        if (bx-u)*(u-cx) > 0.0:
            fu = func(u)
            if fu < fc: # min btw b and c
                ax = bx
                fa = fb
                bx = u
                fb = fu
                return ax,bx,cx
            elif fu > fb: # min btw a and u
                cx = u
                fc = fu
                return ax,bx,cx
            u = cx + GOLD*(cx-bx) # parabola fit was useless; continue
            fu = func(u)
        elif (cx-u)*(u-ulim) > 0.0:
            fu = func(u)
            if fu < fc:
                bx = cx
                cx = u
                u = cx + GOLD*(cx-bx)
                fb = fc
                fc = fu
                fu = func(u)
        elif (u-ulim)*(ulim-cx) > 0:
            u = ulim
            fu = func(u)
        else:
            u = cx+GOLD*(cx-bx)
            fu = func(u)
        #end if

        ax,bx,cx = bx,cx,u
        fa,fb,fc = fb,fc,fu
    # end while
    return ax,bx,cx

def brent(f, ax, bx, cx, tol):

    ITMAX = 100
    CGOLD = 0.3819660
    ZEPS = 1.0e-10

    e = 0.0
    a = min(ax, cx)
    b = max(ax, cx)
    v = bx
    w = v
    x = v
    fw = fv = fx = f(x)

    xm = 0.0
    d = 0.0
    u = 0.0

    for iter in range(ITMAX):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + ZEPS
        tol2 = 2.0 * tol1
        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            xmin = x
            return xmin, fx

        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp = e
            e = d
            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x):
                e = a - x if x >= xm else b - x
                d = CGOLD * e
            else:
                d = p / q
                u = x + d
            if u - a < tol2 or b - u < tol2:
                d = SIGN(tol1, xm - x)

        else:
            e = a - x if x >= xm else b - x
            d = CGOLD * e

        u = x + d if abs(d) >= tol1 else x + SIGN(tol1, d)
        fu = f(u)
        if fu <= fx:
            if (u >= x):
                a = x
            else:
                b = x

            v, w, x = w, x, u
            fv, fw, fx = fw, fx ,fu

        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    print >> sys.stderr, "Too many iterations in brent"
    xmin = x
    return xmin, fx


def linmin(func, p, x, ftol):
    """Given a n dimensional vector x and a direction dir, find the
    minimum along dir from x"""

    def func1d(s):
        return func(p+s*x)

    ax = 0.0
    bx = 1.0
    ax,bx,cx = mnbrak(func1d, ax,bx)
#    print >>sys.stderr, "mnbrak result", ax, bx, cx
    xmin, fx = brent(func1d, ax,bx,cx, ftol)
#    print >>sys.stderr, "brent result", xmin, fx

    return fx, p + xmin * x


import numpy as np
def cg_optimize(f, gf, x0, max_epoches, ftol):
    EPS = 1.0e-8

    n = len(x0)

    p = x0

    fp = f(x0)
    xi = gf(x0)

    fret = fp
    g = - xi
    xi = h = g


    for its in range(max_epoches):

        print "iteration :", its
        fret, p = linmin(f, p, xi, ftol)

        if 2.0 * abs(fret - fp) <= ftol * (abs(fret) + abs(fp) + EPS):
            break

        fp = fret
        xi = gf(p)

        gg = np.linalg.norm(g)**2
        dgg = np.dot(xi, xi + g)

        if gg == 0.0:
            break

        gam = dgg / gg

        g = -xi
        xi = h = g + gam * h

    return p
