import xmlrpclib
import sys
proxy = xmlrpclib.ServerProxy("http://localhost:8000/")
print "result %s" % proxy.handle_ws(eval(sys.argv[1]))
