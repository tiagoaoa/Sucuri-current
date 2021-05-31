import sys
import os

sys.path.append(os.environ['PYDFHOME'])

from pyDF import *

nprocs = int(sys.argv[1])

graph = DFGraph()
sched = SchedulerWS(graph, nprocs, mpi_enabled = False)
req_node, resp_node = sched.set_wservice(("localhost", 8000))


def filter_function(args):
    return args[0]+1





filter_node = FilterTagged(filter_function,1)

graph.add(req_node)

graph.add(filter_node)

graph.add(resp_node)

req_node.add_edge(filter_node, 0)
filter_node.add_edge(resp_node,0)




sched.start()

