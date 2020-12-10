import sys, os
sys.path.append(os.environ['PYDFHOME'])
from pyDF import *

def imprime(args):
       print(args)
       return None

nworkers = int(sys.argv[1])
n = int(sys.argv[2])
graph = DFGraph()
sched = Scheduler(graph, nworkers, mpi_enabled = False)


s0 = Source(range(n))
s1 = Source(range(n))
s2 = Source(range(n))
s3 = Source(range(n))
s4 = Source(range(n))

p = FilterTagged(imprime,5)

graph.add(s0)
graph.add(s1)
graph.add(s2)
graph.add(s3)
graph.add(s4)
graph.add(p)


s0.add_edge(p,0)
s1.add_edge(p,1)
s3.add_edge(p,2)
s3.add_edge(p,3)
s4.add_edge(p,4)

sched.start()
