"""
    Server side program to listen for images sent to this server on port 45400
    Responsibilities:
        * Listen for images
        * Run received images on facial recognition model
        * If face is recognized, send back to client ssh password for their account
        * If face is not recognized, send an error to the client
"""

import os
import socket
import authenticate
INDENT = " " * 4


def connect_to_client(host, port):
    """ Establishes a connection with a client who connects to this server """
    print("[*] Listening for a client...")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(10)
    client_sock, addr = server_sock.accept()
    print("{}[+] Connected to {}".format(INDENT, addr[0]))
    return server_sock, client_sock, addr


def receive_label(client_sock):
    """ Receieves a label from the client_sock connection. When receiving, the label
        must be preceeded by "LABEL ", so that the program can correctly separate it. """
    print("[*] Listening for username...")
    try:
        data = client_sock.recv(1024)
        if data.decode().startswith("LABEL "):
            label = data.decode().split(" ")[1]
            print("{}[+] Got username: {}".format(INDENT, label))
            return label
        else:
            print("{}[!] No username received".format(INDENT))
            return None
    except:
        print("{}[!] There was an error getting a username".format(INDENT))
        return None


def receive_size(client_sock):
    """ Receives the size of the image to be sent so that the program knows when to stop listening
        for data being sent. """
    print("[*] Listening for image size...")
    try:
        data = client_sock.recv(9)
        print(data)
        if data.decode().startswith("SIZE "):
            size = data.decode().split(" ")[1]
            print("{}[+] Got size: {}".format(INDENT, size))
            return int(size)
        else:
            print("{}[!] Could not retrieve the image size".format(INDENT))
            return None
    except:
        print("{}[!] There was an error getting the image size".format(INDENT))
        return None


def receive_image(client_sock, file_name, size):
    """ Receives an image from client_sock and stores it in file_name
        Returns true on success """
    print("[*] Listening for image...")
    try:
        image = open(file_name, "wb")
        bytes_received = 0
        data = ""
        while data and bytes_received <= size:
            data = client_sock.recv(1024)
            bytes_received += len(data)
            print("{}[+] Received {} bytes, {} total".format(INDENT, len(data), bytes_received))
            image.write(data)
        print("{}[+] Finished receiving data".format(INDENT))
        image.close()
        return True
    except KeyboardInterrupt:
        print("{}[!] User quitting, closing connection".format(INDENT))
        try:
            client_sock.close()
            return False
        except:
            print("{}[!] There was an error, quitting".format(INDENT))
            return False


def retrieve_credentials(username):
    """ THIS IS VERY BAD PRACTICE AND IF YOU'RE READING THIS CODE, KNOW THAT I HATE IT WITH
        EVERY FIBER OF MY BEING. I plan on fixing this in the future, but for now it is just a
        temporary place holder until I figure out how to generate attribute-based credentials. """
    os.system("bash decrypt.sh")
    password = None
    try:
        with open("passwords.txt", "r") as pwords:
            print("Looking for credentials...")
            for line in pwords:
                if line.startswith(username):
                    password = line.split(":")[1]
                    break
    finally:
        os.system("bash encrypt.sh")
        return password


def send_credentials(client_sock, password):
    """ Along with above this is VERY BAD PRACTICE, and I hate it, but this
        function sends the user's password in PLAINTEXT back to them
        For any potential employers, I would not code like this professionally. """

    client_sock.sendall(password.encode())

def handle_client(host, port, file_type):
    """ Handles 1 client at a time.
    TODO: enable multithreading of client requests """
    _, client_sock, addr = connect_to_client(host, port)
    username = receive_label(client_sock)
    file_name = username + file_type
    image_size = receive_size(client_sock)

    if receive_image(client_sock, file_name, image_size):
        print("[*] Wrote data to {}".format(file_name))
    else:
        print("[*] Error receiving image")
        client_sock.shutdown(socket.SHUT_WR)
        client_sock.close()

    print("\n\n[*************** Authenticating ***************]")
    if authenticate.is_valid_user(username):
        if authenticate.authenticate(username, file_name):
            print("[*] Welcome back {}".format(username))
            password = retrieve_credentials(username)
            if password:
                send_credentials(client_sock, password)
                print("{}[+] Sent credentials!\n".format(INDENT))
            else:
                print("[!] Could not retrieve password\n")
    else:
        print("[!] Not success\n")

    client_sock.shutdown(socket.SHUT_WR)
    client_sock.close()

def main():
    """ Loops handle_client infinitely to handle client requests """
    host = "0.0.0.0"
    port = 45400
    file_type = ".jpg"

    while True:
        try:
            handle_client(host, port, file_type)
        #except KeyboardInterrupt as user_quit:
        #    print("[!] <ctrl+C> pressed, user quitting...")
        #    return
        except:
            # Do nothing for now?
            print("There was an error handling previous client")

if __name__ == "__main__":
    main()
