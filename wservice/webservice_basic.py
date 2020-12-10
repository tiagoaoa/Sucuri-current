import os
import sys


#import xmlrpc

from xmlrpc.server import SimpleXMLRPCServer


def myhandler(server):
    while True:    
        req = server.get_request()
        #server.process_request(req[0], req[1])
        yield server,req

def soma(a, b):
        return a+b

def handler(server):
    print(server.get_request())
    

server = SimpleXMLRPCServer(("localhost", 8000))
print("Listening on port 8000...")
server.register_function(soma)
for i in myhandler(server):
    s,r = i
    s.process_request(r[0], r[1])
