import os
import sys
from multiprocessing import Process, Queue
import threading

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

#for i in range(10):
#    q.put(i)
#for x in q:
#    print("Dequeued {}".format(x))



class NodeDebug(Node):
    def __repr__(self):
        return "Node for Debugging {}".format(id(self))
class NodeWS(Node):


class SourceWS(Source):
    def f(self, line, args):
        print("[SourceWS] Receiving request {} {}".format(line, self.tagcounter))

        #ResponseQueues[self.tagcounter] = Queue()
#        print("[SourceWS] Waiting... {}".format(ResponseQueues))
        return line
    

class ThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass

class ThreadedDict(dict):
    def __init__(self):
        dict.__init__(self)
        self.__lock__ = threading.Lock()
        self.__cond__ = threading.Condition(lock = self.__lock__)


    def lock(self):
        self.__lock__.acquire()

    def signalAll(self):
        self._lock__.release()
        self.__cond__.notify_all()


class WebService(Process):
    def __init__(self, server):
        Process.__init__(self)


        server = ThreadedXMLRPCServer(server)
        print("Listening on port 8000.")
        #server.register_function(lambda args: print("Args {}".format(args)), 'service')

        self.server = server
        self.req_queue = Iter_Queue()
        self.resp_queue = {777: "Teste"}
        self.d = ThreadedDict()
        self.node = SourceWS(self.req_queue)

    def resp_loop(self):
        d = self.d
        while True:
            x, val = self.resp_conn.get()
            d.lock()
            d[x] = val
            d.signalAll()



    def get_response(x):
        d = self.d
        resp = None
        d.lock()
         while resp == None:
            if d.has_key(x) 
                resp = d.pop(x)
                    d.unlock()
            else:
                d.wait()  #wait() releases and acquires the lock after the condition is notified

        return resp

    def service(args):
        thread_name = threading.currentThread().getName()
        print("Fila {}".format(self.resp_queue))
        self.req_queue.put(args)
            
        return self.get_response(thread_name)
            """rq = self.resp_q
            rq = self.resp_queue[thread_name] = Queue()
            print("[Service] Out {}".format(rq.get))
            print("[Service] Queues {} {}".format(self.resp_q, id(rq)))
            print("Args in service {} queue {} {} thread {}".format(args, self.resp_queue, self.req_queue, threading.currentThread().getName()))"""
        return 7777



    def run(self):
        print("WebService Running")
        threading.Thread(target = resp_loop)
        server = self.server
        server.register_function(service)
        count = 0
        while True:
            req, cl_addr = server.get_request()
            count += 1
            self.resp_queue[count] = Queue()
            server.process_request(req, cl_addr)
            #req.sendall(b'1234')
            #server.handle_request()


class NodeWS(Node):
    
    def __init__(self, resp_conn, number_of_input_ports=1):
        Node.__init__(self, None, number_of_input_ports)
        self.resp_conn = resp_conn
    def run(self, args, workerid, operq):
        self.resp_conn.put((args[0].tag, args[0].value))
      
        opers = self.create_oper(None, workerid, operq)
        self.sendops(opers, operq)
 




class SchedulerWS(Scheduler):
    def all_idle(self, workers):
        return False

    def set_wservice(self, wservice):
        self.ws = WebService(wservice)
        self.resp_q = {'Test': 'abc'}
        resp_conn, wservice_conn = Pipe()
        conn = self.workers[1].conn
        self.workers[1] = Worker(self.graph, self.operq, conn, 1)
        
        self.ws.resp_conn = wservice_conn

        resp_node = NodeWS(resp_conn)
        self.ws.start()

        self.add(resp_node)
        resp_node.pin([1])
        return resp_node






             

def response_enqueue(args):
    print("Resp:  {}".format(wservice.resp_queue))

    #    print("Args to enqueue {}".format(args[0#]))
#    ResponseQueues[args[0].value[0]].put[args[0].value[1]]



nprocs = int(sys.argv[1])

graph = DFGraph()
sched = SchedulerWS(graph, nprocs, mpi_enabled = False)#, wservice = ("localhost", 8000))
resp_node = sched.set_ws(("localhost", 8000))




#reqs = SourceWS(iter_q)


graph.add(reqs)


reqs.add_edge(resp_node, 0)

reqs.pin([0])

sched.start()





