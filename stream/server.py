import socket
from comms import receive_array
import numpy as np
from PIL import Image

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 50000
    server.bind(('localhost', port))
    server.listen(1)
    
    print(f"Server listening on localhost:{port}")
    
    while True:
        client, addr = server.accept()
        print(f"Connection from {addr}")
        
        array = receive_array(client)
        print(f"Received: {type(array)}")
        Image.fromarray(array).save("minimap.png")
        
        client.close()

start_server()
