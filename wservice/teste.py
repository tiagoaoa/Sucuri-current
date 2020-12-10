from multiprocessing import Process, Queue, Value, Pipe
import sys
import os
sys.path.append(os.environ['HOME'] + "/xmlrpclib-1.0.1")

import xmlrpclib

from SimpleXMLRPCServer import SimpleXMLRPCServer
import pickle 

def soma(a, b):
    return a+b

def cria(q):
    print "Cria"

def recebe(q):
    print q.get()

q = Queue()
a = Process(target=cria, args=(q,))
b = Process(target=recebe, args=(q,))

p0, p1 = Pipe()

q.put('a')
print q.get()
