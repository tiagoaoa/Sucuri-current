import os
import sys

sys.path.append(os.environ['PYDFHOME'])

from xmlrpc.server import SimpleXMLRPCServer


from pyDF import *

def addition(a, b):
        return a+b





class Request_Iter:
    def __init__(self,server):
        server = SimpleXMLRPCServer(server)
        print("Listening on port 8000.")
        server.register_function(lambda args: args)

        self.server = server
        pass
    def __iter__(self):
        return self
    def __next__(self):
        s = self.server
        r = self.server.get_request()
        #print s,r
        #return [[s,r]]
        return [[1,2]]



def print_args(args):
    print("Args were {}".format(args))

req_iter = Request_Iter(("localhost", 8000))


nprocs = int(sys.argv[1])

graph = DFGraph()
sched = Scheduler(graph, nprocs, mpi_enabled = False)#, wservice = ("localhost", 8000))


reqs = Source(req_iter)

print_node = Node(print_args, 1)
graph.add(reqs)

graph.add(print_node)

reqs.add_edge(print_node, 0)

#print(sched.queues[req[1]].put([1234]))
#print("Req {} {}".format(str(req[0]), str(req[1])))


sched.start()





