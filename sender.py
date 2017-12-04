""" Program to send an image to a remote server (AWS EC2 instance) """

import sys
import socket
import re
import os

INDENT = " " * 4


def open_session(username, password, host):
    """ Opens a putty session on the client computer. Linux only format right now"""
    os.system("putty -ssh -l {} {} -pw {}".format(username, host, password))


def connect_to_server(host, port):
    """ Connect to a server listening on the specified port """
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect((host, port))
    return client_sock


def send_label(label, client_sock):
    """ Label is in the image name, as in davidposs.12.jpg would have a label of davidposs """
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


def send_image(image_path, client_sock, size):
    """ Sends an image to the server """
    try:
        print("[*] Sending image to server... ")
        with open(image_path, "rb") as image_file:
            bytes_sent = 0
            while bytes_sent < size:
                chunk = image_file.read(1024)
                client_sock.sendall(chunk)
                bytes_sent += len(chunk)
                print("{}[+] Sending data, {} sent, {} total".format(INDENT, bytes_sent, size))

        print("{}[+] Finished sending".format(INDENT))
        return True
    except:
        print("{}[!] Error sending file, stopping.".format(INDENT))
        return False


def main():
    """ Main program to send a label and image to the server to authenticate the user. On success
        the program will launch a putty ssh session to the server under the user's account """
    host = "54.215.129.201"
    port = 45400
    image_path = sys.argv[1]

    if not image_path:
        print("[!] {} is not a valid image path".format(image_path))
        return

    # Regular expression to grab the labels from the pathname
    username = re.findall(r'[^/]*$', image_path)[0].split(".")[0]
    with open(image_path, "rb") as image:
        size = len(image.read())

    client_sock = connect_to_server(host, port)
    send_label(username, client_sock)
    send_image_size(size, client_sock)
    send_image(image_path, client_sock, size)

    print("[*] Waiting for server response...")
    password = client_sock.recv(1024).decode()
    if password:
        print("{}[+] received credentials!".format(INDENT))
        open_session(username, password, host)
        client_sock.shutdown(socket.SHUT_WR)
        client_sock.close()
    else:
        print("{}[!] Could not get credentials".format(INDENT))
        client_sock.shutdown(socket.SHUT_WR)
        client_sock.close()


if __name__ == "__main__":
    main()
