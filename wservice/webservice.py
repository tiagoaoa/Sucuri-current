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
        print("Iter Queue Enqueueing {} -- {}".format(x, self))
        self.q.put(x)

    def __iter__(self):
        return self
    def __next__(self):
        return self.q.get()

iter_q = Iter_Queue()
#for i in range(10):
#    q.put(i)
#for x in q:
#    print("Dequeued {}".format(x))



class SchedulerWS(Scheduler):
    def all_idle(self, workers):
        return False


class NodeDebug(Node):
    def __repr__(self):
        return "Node for Debugging {}".format(id(self))

ResponseQueues = {777: "Teste"}

class SourceWS(Source):
    def f(self, line, args):
        print("[SourceWS] Receiving request {} {}".format(line, self.tagcounter))
        #ResponseQueues[self.tagcounter] = Queue()
        print("[SourceWS] Waiting... {}".format(ResponseQueues))
        return line
    

class ThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


class WebService(Process):
    def __init__(self, server, resp_q):
        Process.__init__(self)
        self.resp_q = resp_q


        server = ThreadedXMLRPCServer(server)
        print("Listening on port 8000.")
        #server.register_function(lambda args: print("Args {}".format(args)), 'service')

        self.server = server
        self.req_queue = Iter_Queue()
        self.node = SourceWS(self.req_queue)


        def service(args):
            print("Args in service {} queue {} {}".format(args, ResponseQueues, self.req_queue))
            
            self.req_queue.put(args)

            rq = self.resp_q
            #rq = ResponseQueues[args[0].tag]
            print("[Service] Queues {}".format(self.resp_q))
            #return rq.get()
            return 7777


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
sched = SchedulerWS(graph, nprocs, mpi_enabled = False)#, wservice = ("localhost", 8000))


wservice = WebService(("localhost", 8000), ResponseQueues)
wservice.start()


#reqs = SourceWS(iter_q)
reqs = wservice.node
response = NodeDebug(response_enqueue, 1)


graph.add(reqs)

graph.add(response)

reqs.add_edge(response, 0)

reqs.pin([0])
response.pin([1])

sched.start()





