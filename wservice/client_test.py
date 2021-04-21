import xmlrpc.client
import sys
proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

print("Test client for WebService functionality")
#response = proxy.service(int(sys.argv[1]), int(sys.argv[2]))
response = proxy.service([2,3])

print(response)
