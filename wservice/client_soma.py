import xmlrpclib
import sys
proxy = xmlrpclib.ServerProxy("http://localhost:8000/")
print "soma %d" % proxy.handle_ws(int(sys.argv[1]), int(sys.argv[2]))
