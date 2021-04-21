import xmlrpc
import sys
proxy = xmlrpc.ServerProxy("http://localhost:8000/")
print "result %s" % proxy.handle_ws(eval(sys.argv[1]))
