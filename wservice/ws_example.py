import sys
import os

sys.path.append(os.environ['PYDFHOME'])

from pyDF import *

nprocs = int(sys.argv[1])

graph = DFGraph()
sched = SchedulerWS(graph, nprocs, mpi_enabled = False)
req_node, resp_node = sched.set_wservice(("localhost", 8000))


graph.add(req_node)
graph.add(resp_node)

req_node.add_edge(resp_node, 0)


sched.start()

