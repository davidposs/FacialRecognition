"""
Created on Thu Nov 16 16:27:29 2017
@author: David Poss

Program to send an image to a remote server (AWS EC2 instance)
"""

import os
import cv2
import socket
import subprocess

INDENT = " " * 4


def valid_input(args):
    """ Checks command line arguments for validity """
    if len(args) == 1:
        print("[!] Please supply an image to use")
        return False
    if len(args) > 2:
        print("[!] Too many arguments, please only supply one")
        return False
    if not os.path.exists(args[1]):
        print("[!] {} is not a valid file path".format(args[1]))
        return False
    if os.path.exists(args[1]) and args[1].endswith(".jpg"):
        return True
    return False


def open_session(username, password, host):
    """ Opens a putty session on the client computer. Linux only format right now"""
    password = password.split("\n")[0]
    subprocess.Popen(["gnome-terminal", "-e",
                      "plink {}@{} -pw {}".format(username, host, password)])


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


def take_picture(username, out_type):
    """ Takes a picture of the user to send to the server """
    cascades = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)
    print("Taking a picture to send to the server")
    while True:
        try:
            _, frame = video_capture.read()
            grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascades.detectMultiScale(grayed, scaleFactor=1.1, minNeighbors=5)
            # Get x, y coordinates and width and height of face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.imshow('Webcam', frame)

            key_press = cv2.waitKey(1)
            # Press 's' to save detected face
            if key_press & 0xFF == ord('q'):
                return False
            elif key_press & 0xFF == ord('s'):
                face_to_save = frame[y:y + h, x:x + w]
                cropped_face = cv2.resize(face_to_save, (180, 180), interpolation=cv2.INTER_AREA)
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                # Add a ".1" to keep file naming convention
                image_path = username + "." + str(1) + out_type
                cv2.imwrite(image_path, cropped_face)
                return image_path
        except KeyboardInterrupt:
            print("[!] User stopped, quitting")
            return None


def main():
    """ Main program to send a label and image to the server to authenticate the user. On success
        the program will launch a putty ssh session to the server under the user's account """
    host = "54.215.129.201"
    port = 45400

    username = input("Enter your username: ")
    image_path = take_picture(username, out_type=".jpg")
    if not image_path:
        print("[!] Error taking the picture, quitting")
        return
    # Regular expression to grab the labels from the pathname
    # username = re.findall(r'[^/]*$', image_path)[0].split(".")[0]
    with open(image_path, "rb") as image:
        size = len(image.read())

    client_sock = connect_to_server(host, port)
    send_label(username, client_sock)
    send_image_size(size, client_sock)
    send_image(image_path, client_sock, size)

    print("[*] Waiting for authentication...")
    password = client_sock.recv(1024).decode()
    if password:
        print("{}[+] Access granted!".format(INDENT))
        open_session(username, password, host)
        try:
            client_sock.close()
        finally:
            return
    else:
        print("{}[!] Could not get credentials".format(INDENT))
        client_sock.shutdown(socket.SHUT_WR)
        client_sock.close()


if __name__ == "__main__":
    main()
