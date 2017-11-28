""" Program to send an image to a remote server (AWS EC2 instance) """
import socket
import re
import sys

HOST = "54.215.129.201"
PORT = 45400
IMAGE_PATH = "david16centerlight.jpg"

def open_session(username, password):
    os.system("putty.exe {}@{} -pw {}".format(username, HOST, password))

def main():
    image_file = open(IMAGE_PATH,'rb')

    # Regular expression to grab the labels from the pathname
    subject_name = re.findall(r'[^/]*$', IMAGE_PATH)[0].split(".")[0]

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect((HOST, PORT))

    message = "LABEL " + subject_name
    client_sock.sendall(message.encode())
    image_size = len(image_file.read())


    chunk = image_file.read(1024)
    while (chunk):
        print("Sending data...")
        client_sock.sendall(chunk)
        chunk = image_file.read(1024)
    image_file.close()
    print("File sent!")

    message = "FIN"
    client_sock.sendall(message.encode())
    response = client_sock.recv(1024)
    if response.decode() == "END":
        password = client_sock.recv(1024)
        password = password.decode()
        print("Got password: {}".format(password))
    else:
        print("Didn't get END")

    client_sock.shutdown(socket.SHUT_WR)
    print(client_sock.recv(1024))
    client_sock.close()


if __name__ == "__main__":
    main()
"""
try:
    print("Opening image")
    image = open(IMAGE_PATH, "rb")
    data = image.read()
    size = len(data)
    print("Sending size")
    message = "SIZE {}".format(size)
    client_sock.sendall(message.encode())
    print("Waiting for server response")
    server_response = client_sock.recv(BUFF_SIZE)
    print(server_response.decode())
    if server_response.decode() == "Size received":
        print("Server received size, sending data")
        client_sock.sendall(data)
        server_response = client_sock.recv(BUFF_SIZE)
        if server_response.decode() == "Image received":
            client_sock.sendall("BYE".encode())
            print("Server received {}, exiting now.".format(IMAGE_PATH))
    image.close()
finally:
    client_sock.close()
"""