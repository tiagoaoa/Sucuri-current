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
        print("Enqueueing {}".format(x))
        self.q.put(x)

    def __iter__(self):
            yield self.q.get()



ResponseQueues = {}

class SourceWS(Source):
    def f(self, line, args):
        print("Receiving request {} {}".format(line, self.tagcounter))
        ResponseQueues[self.tagcounter] = Queue()
        return line
    

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
            
            self.req_queue.put(args)
            rq = ResponseQueues.pop(args.tag)
            
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
    print("Args to enqueue {}".format(args[0]))
    ResponseQueues[args[0].tag] = args[0].value
    print("Queue is {}".format(ResponseQueues))




nprocs = int(sys.argv[1])

graph = DFGraph()
sched = Scheduler(graph, nprocs, mpi_enabled = False)#, wservice = ("localhost", 8000))


wservice = WebService(("localhost", 8000))
wservice.start()

req_iter = wservice.req_queue


reqs = SourceWS(req_iter)

response = Node(response_enqueue, 1)


graph.add(reqs)

graph.add(response)

reqs.add_edge(response, 0)



sched.start()





