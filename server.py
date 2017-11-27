
import sys
import socket

def receive_image(server_sock):
    #while True:
    label = ""
    try:
        print("[*] Listening for connections")
        client_sock, addr = server_sock.accept()
        print("[+] Established connection with {}".format(addr[0]))
        print("[+] Listening for data...")
        data = client_sock.recv(1024)
        if data.decode().startswith("LABEL "):
            label = data.decode().split(" ")[1]
            file_name = label + ".jpg"
            print("Got {}".format(file_name))
            image_file = open(file_name, "ab+")
        else: return None, None, None
        while data:
            print ("receiving")
            image_file.write(data)
            data = client_sock.recv(1024)
        print ("finished receiving data")
        msg = "FINACK"
        client_sock.sendall(msg.encode())
        print("FINACK sent")
        return client_sock, addr, label
    except KeyboardInterrupt as term_signal:
        print("User quitting, closing connection")
        try:
            client_sock.close()
        finally:
            return None, None, None


def main():

    host = "0.0.0.0"
    port = 45400
    filename = "new_face.jpg"

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((host, port))

    server_sock.listen(5)
    client_sock, addr, label = receive_image(server_sock)
    if client_sock is not None:
        print ("did you pass?")
        if label == "davidposs":
            password = "letmein"
            client_sock.sendall(password.encode())
            print("Sent password for davidposs")
            

if __name__ == "__main__":
    main()
