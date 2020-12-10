import os
import sys

sys.path.append(os.environ['PYDFHOME'])

sys.path.append(os.environ['HOME'] + "/xmlrpclib-1.0.1")


from pyDF import *

def soma(a, b):
        return a+b





class Request_Iter:
    def __init__(self,server):
        #server = SimpleXMLRPCServer(("localhost", 8000))
        #print "Listening on port 8000..."
        #   server.register_function(soma)"

        self.server = server
        pass
    def __iter__(self):
        return self
    def next(self):
        s = self.server
        r = self.server.get_request()
        #print s,r
        #return [[s,r]]
        return [[1,2]]








nprocs = int(sys.argv[1])

graph = DFGraph()
sched = Scheduler(graph, nprocs, mpi_enabled = False, wservice = ("localhost", 8000))

req= sched.queues["source"].get()
print sched.queues[req[1]].put([1234])
print "Req %s %s" %(str(req[0]), str(req[1]))


sched.start()






