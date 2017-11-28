""" Program to send an image to a remote server (AWS EC2 instance) """

import socket
import re
import os

IMAGE_PATH = "../../Data/SmallTest/davidposs.1.jpg"
INDENT = " " * 4


def open_session(username, password, host):
    os.system("putty.exe {}@{} -pw {}".format(username, host, password))


def connect_to_server(host, port):
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect((host, port))
    return client_sock


def send_label(label, client_sock):
    print("[*] Sending label to server...")
    message = "LABEL " + label
    client_sock.sendall(message.encode())
    print("{}[+] Sent label to server".format(INDENT))

def send_image_size(size, client_sock):
    """ Sends image size so server knows when to stop receiving """
    print("[*] Sending image size...")
    message = "SIZE " + str(size)
    client_sock.sendall(message.encode())
    print("{}[+] Sent image size of {}".format(INDENT, size))

def send_image(image_path,client_sock):
    try:
        print("[*] Sending image to server... ")
        with open(image_path, "rb") as image_file:
            chunk = image_file.read(1024)
            while (chunk):
                print("Sending data...")
                client_sock.sendall(chunk)
                chunk = image_file.read(1024)
        return True
    except:
        print("{}[!] Error sending file, stopping.".format(INDENT))
        return False


def main():
    host = "54.215.129.201"
    port = 45400
    # Regular expression to grab the labels from the pathname
    username = re.findall(r'[^/]*$', IMAGE_PATH)[0].split(".")[0]
    with open(IMAGE_PATH, "rb") as image:
        size = len(image.read())

    client_sock = connect_to_server(host, port)

    send_label(username, client_sock)
    send_image_size(size, client_sock)
    send_image(IMAGE_PATH, client_sock)

    password = client_sock.recv(1024)
    if password:
        print("Got my password! {}".format(password))
        open_session(username, password, host)
    return

if __name__ == "__main__":
    main()
