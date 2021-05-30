from multiprocessing import Process, Queue
import threading


from pyDF import *


from socketserver import ThreadingMixIn

from xmlrpc.server import SimpleXMLRPCServer

from datetime import datetime

 
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





class SourceWS(Source):
    def f(self, line, args):
        print("[{}][SourceWS] Receiving request {} {}".format(datetime.now(),line, args))

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
        print("Got lock")
    def unlock(self):
        self.__lock__.release()

    def signalAll(self):
        self.__cond__.notify_all()
        self.__lock__.release()
        print("Release lock in signalAll")
    def wait(self):
        self.__cond__.wait()

class WebService(Process):
    def __init__(self, server):
        Process.__init__(self)


        server = ThreadedXMLRPCServer(server)
        print("Listening on port 8000.")
        #server.register_function(lambda args: print("Args {}".format(args)), 'service')
        self.req_queue = Iter_Queue()
        self.server = server
        self.d = ThreadedDict()

    def resp_loop(self):
        d = self.d
        while True:
            x, val = self.resp_conn.recv()
            print("Got a response {}".format((x, val)))
            d.lock()
            d[x] = val
            print("Got the lock to insert {}".format((d,x,val)))
            d.signalAll()



    def get_response(self, x):
        d = self.d
        resp = None
        x = x - 2
        d.lock()
        print("Got lock {}".format(x))
        while resp == None:
            print("resp is None {}".format(d))
            if x in d:
                print("Getting {}".format(x))
                resp = d.pop(x)
                d.unlock()
            else:
                print("Waiting for {}".format(x))
                d.wait()  #wait() releases and acquires the lock after the condition is notified
        print("Responding {}".format(resp)) 
        return resp

    def service(self, args):
        thread_name = threading.currentThread().getName()
        thread_id = int(thread_name.split('-')[1])
        
        self.req_queue.put(args)
        
        return self.get_response(thread_id)



    def run(self):
        print("WebService Running")
        threading.Thread(target = self.resp_loop).start()
        server = self.server
        server.register_function(self.service)
        while True:
            req, cl_addr = server.get_request()
            server.process_request(req, cl_addr)


class NodeWS(Node):
    
    def __init__(self, resp_conn, number_of_input_ports=1):
        Node.__init__(self, None, number_of_input_ports)
        self.resp_conn = resp_conn
    def run(self, args, workerid, operq):
        print("Args {}".format(args[0].val))
        self.resp_conn.send((args[0].val.tag, args[0].val.value))
      
      #  opers = self.create_oper(None, workerid, operq) #necessary to make worker request more tasks
      #  self.sendops(opers, operq)
 




class SchedulerWS(Scheduler):
    def all_idle(self, workers):
        return False

    def set_wservice(self, wservice):
        self.ws = WebService(wservice)
        resp_conn, wservice_conn = Pipe()
        resp_worker_id = self.n_workers
        self.n_workers += 1
        #conn = self.workers[resp_worker_id].conn
        
        sched_conn, worker_conn = Pipe()
        self.conn.append(sched_conn)
        self.workers.append(Worker(self.graph, self.operq, worker_conn, 1))
        self.pending_tasks.append(0) 
        
        self.ws.resp_conn = wservice_conn
        req_iter = self.ws.req_queue

        req_node = SourceWS(req_iter)
        resp_node = NodeWS(resp_conn)
        self.ws.start()

        resp_node.pin([resp_worker_id])
        req_node.pin([0])

        return req_node, resp_node






             








