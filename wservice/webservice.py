import os
import sys
from multiprocessing import Process, Queue
sys.path.append(os.environ['PYDFHOME'])

from socketserver import ThreadingMixIn

from xmlrpc.server import SimpleXMLRPCServer


from pyDF import *

class Iter_Queue():
    def __init__(self):
        self.q = Queue()

    def put(self, x):
        print("Enqueueing {}".format(args))
        self.q.put(x)

    def __iter__(self):
            yield self.q.get()


ResponseQueues = {}
class ThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


class WebService(Process):
    def __init__(self, server):
        Process.__init__(self)



        server = ThreadedXMLRPCServer(server)
        print("Listening on port 8000.")
        #server.register_function(lambda args: print("Args {}".format(args)), 'service')

        self.server = server
        self.req_queue = Iter_Queue()

        def service(args):
            resp_q = ResponseQueues
            print("Args in service {}".format(args))
            ResponseQueues[id(args)] = Queue()
            
            self.req_queue.put(args)
            rq = ResponseQueues.pop(id(args))
            
            print("Queues {}".format(ResponseQueues))
            return rq.get()


        server.register_function(service)

    def run(self):
        print("WebService Running")
        server = self.server
        while True:
            #req = server.get_request()
            #server.process_request(req)
            server.handle_request()


             

def response_enqueue(args):
    print("Args to enqueue {}".format(args))
    ResponseQueues[args.tag] = args.value
    return argdasdasdadsa

def print_args(args):
    print("Args were {}".format(args))


nprocs = int(sys.argv[1])

graph = DFGraph()
sched = Scheduler(graph, nprocs, mpi_enabled = False)#, wservice = ("localhost", 8000))


wservice = WebService(("localhost", 8000))
wservice.start()
req_iter = wservice.req_queue


reqs = Source(req_iter)

response = Node(response_enqueue,1)


print_node = Node(print_args, 1)
graph.add(reqs)

graph.add(print_node)

reqs.add_edge(print_node, 0)



sched.start()





