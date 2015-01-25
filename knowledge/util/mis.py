import random, string
import sys,os
import subprocess

def randomword(length):
   return ''.join(random.choice(string.lowercase) for i in range(length))


def tmpfile(tag=''):
    pre = ''
    if sys.platform[:3] == 'win':
        pre = ''
    else:
        pre = '/tmp'
    filename = os.path.join(pre,'knowledge_' + tag + randomword(8))
    f = open(filename,'w')
    return f

def cleantmp(*args):
    for f in args:
        if not f.closed: f.close()
        filename = f.name
        subprocess.call(['rm',filename])


if __name__ == '__main__':
    f1 = tmpfile()
    f2 = tmpfile()
    f1.write("aaaa")
    f2.write("aaaa")
    cleantmp(f1,f2)