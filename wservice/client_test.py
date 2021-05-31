import xmlrpc.client
import sys
proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

print("Test client for WebService functionality")
response = proxy.service(123)

print(response)
